"""
SVD 音频降噪增强版（断音修复 + 混合阈值降噪）
修复特性：
  - 稳健窗补偿（限制最大增益）
  - 智能归一化（避免压死弱信号）
  - 数值健康检查（NaN/Inf修复）
  - 混合阈值降噪（软阈值+硬截断）
  - 独立残差分析图
依赖：numpy, scipy, matplotlib, sounddevice, tqdm (可选)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat, wavfile
import sounddevice as sd
import warnings
from pathlib import Path

# ---------------------- 可选进度条 ----------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    print("提示：安装 tqdm 可显示进度条 (pip install tqdm)")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 配置参数（可修改）======================
CONFIG = {
    'fs': 8000,                      # 采样率（Hz）
    'duration': 10,                 # 生成音频时长（秒）
    'noise_level': 0.1,            # 噪声强度
    'frame_length': 256,           # 帧长
    'overlap': 128,               # 重叠长度
    'energy_thresh': 0.95,        # 能量保留阈值
    'noise_tail_len': 20,         # 估计噪声的尾部奇异值个数
    'fixed_k': [5, 10, 20, 30, 50],  # 固定候选k
    'fine_search_range': 10,      # 精细搜索范围 ±
    'svd_mode': 'hybrid',         # 降噪模式：'hard'硬截断 / 'hybrid'混合（推荐）
    'soft_thresh_factor': 0.9,    # 软阈值因子（相对于噪声标准差）
    'soft_thresh_max': 0.15,      # 软阈值上限（防止过大）
    'plot_enabled': True,         # 是否显示图形
    'plot_residual_alone': True,  # 是否单独绘制残差分析图
    'play_enabled': True,         # 是否播放音频
    'save_wav': True,            # 是否保存WAV文件
}

# ====================== 核心函数 ======================
def ensure_mono(y, Fs):
    """确保信号为单声道，立体声取平均"""
    if y.ndim > 1:
        if y.shape[1] == 2:
            y = np.mean(y, axis=1)
            print("立体声已合并为单声道（取平均）")
        else:
            raise ValueError(f"不支持的声道数: {y.shape[1]}")
    return y

def generate_test_signal(fs, duration):
    """生成 440/880/1320 Hz 三音测试信号 + 包络"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + \
        0.3 * np.sin(2 * np.pi * 880 * t) + \
        0.2 * np.sin(2 * np.pi * 1320 * t)
    attack = int(0.1 * fs)
    release = int(0.2 * fs)
    envelope = np.ones_like(y)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    return y * envelope

def overlap_add_with_window(frames, window_sum, overlap, target_len, max_gain=5.0):
    """
    带窗补偿的重叠相加法（增强版）
    参数:
        frames: 加窗后的帧矩阵 (frame_len, n_frames)
        window_sum: 窗函数累积和向量 (长度 >= target_len)
        overlap: 重叠样本数
        target_len: 目标信号长度
        max_gain: 最大允许补偿增益（防止静音段爆炸）
    """
    frame_len, n_frames = frames.shape
    step = frame_len - overlap
    total_len = frame_len + (n_frames - 1) * step
    signal = np.zeros(total_len)
    for i in range(n_frames):
        start = i * step
        signal[start:start + frame_len] += frames[:, i]
    
    # 裁剪至目标长度
    window_sum = window_sum[:target_len]
    signal = signal[:target_len]
    
    # 稳健窗补偿
    eps = 1e-6
    safe_divisor = np.maximum(window_sum, eps)
    gain = signal / safe_divisor
    
    # 限制最大增益
    max_gain_mask = safe_divisor < (1.0 / max_gain)
    gain[max_gain_mask] = signal[max_gain_mask] * max_gain
    
    return gain

def svd_denoise(y_noisy, y_clean, Fs, config):
    """
    SVD降噪主流程：分帧、SVD、候选k遍历、精细搜索、窗补偿重建
    支持硬截断和混合阈值降噪（软阈值+硬截断）
    """
    fl = config['frame_length']
    ov = config['overlap']
    step = fl - ov
    mode = config['svd_mode']

    # ----- 分帧（加汉宁窗）并计算窗累积和 -----
    n_frames = int(np.ceil((len(y_noisy) - fl) / step)) + 1
    padding = (n_frames - 1) * step + fl - len(y_noisy)
    y_padded = np.pad(y_noisy, (0, max(padding, 0)), 'constant')
    
    win = np.hanning(fl)
    frames = np.zeros((fl, n_frames))
    window_sum = np.zeros(len(y_padded))
    for i in range(n_frames):
        start = i * step
        segment = y_padded[start:start + fl]
        frames[:, i] = segment * win
        window_sum[start:start + fl] += win
    
    window_sum = window_sum[:len(y_noisy)]
    A = frames.T  # (n_frames, fl)

    # ----- SVD -----
    U, S_diag, Vt = np.linalg.svd(A, full_matrices=False)
    sing_vals = S_diag.copy()
    total_energy = np.sum(sing_vals ** 2)
    energy_cum = np.cumsum(sing_vals ** 2) / total_energy

    # ----- 噪声水平估计（带上限）-----
    noise_std = np.std(sing_vals[-config['noise_tail_len']:])
    soft_thresh = min(config['soft_thresh_factor'] * noise_std,
                      config.get('soft_thresh_max', 0.2))

    # ----- 候选k生成（粗筛）-----
    energy_thresh_idx = np.where(energy_cum >= config['energy_thresh'])[0][0] + 1
    hard_thresh_idx = np.sum(sing_vals > 3 * noise_std)
    k_candidates = set(config['fixed_k'])
    k_candidates.add(min(fl, energy_thresh_idx + 20))
    k_candidates.add(min(fl, hard_thresh_idx))
    k_candidates = sorted(k_candidates)
    print(f'粗筛候选 k: {k_candidates}')

    # ----- 粗筛遍历 -----
    results = {}
    print('\n===== 粗筛阶段 =====')
    print('k值\tSNR(dB)\t能量(%)\tΔSNR(dB)\t模式')
    for k in tqdm(k_candidates, desc='粗筛'):
        # --- 根据模式处理奇异值 ---
        if mode == 'hard':
            A_recon = U[:, :k] @ (S_diag[:k, np.newaxis] * Vt[:k, :])
        elif mode == 'hybrid':
            S_hybrid = np.zeros_like(S_diag)
            S_hybrid[:k] = np.maximum(S_diag[:k] - soft_thresh, 0)
            A_recon = U @ (S_hybrid[:, np.newaxis] * Vt)
        else:
            raise ValueError(f"未知模式: {mode}")

        # 带窗补偿的重叠相加（使用增强版）
        y_recon = overlap_add_with_window(A_recon.T, window_sum, ov, len(y_clean), max_gain=5.0)
        
        # 数值健康检查
        if np.any(np.isnan(y_recon)) or np.any(np.isinf(y_recon)):
            y_recon = np.nan_to_num(y_recon, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 智能归一化
        max_abs = np.max(np.abs(y_recon))
        if max_abs > 0.01:
            y_recon /= max_abs
        elif max_abs > 0:
            y_recon = np.zeros_like(y_recon)  # 极弱信号直接静音

        snr, delta, ener = _calc_metrics(y_clean, y_recon, energy_cum, k, SNR_original)
        results[k] = {'y': y_recon, 'snr': snr, 'ener': ener, 'delta': delta}
        print(f'{k:2d}\t{snr:.2f}\t{ener:.1f}\t+{delta:.2f}\t{mode}')

    # ----- 精细搜索（围绕最优k）-----
    best_k_coarse = max(results, key=lambda x: results[x]['snr'])
    print(f'\n粗筛最优 k = {best_k_coarse}, SNR = {results[best_k_coarse]["snr"]:.2f} dB')

    low = max(1, best_k_coarse - config['fine_search_range'])
    high = min(fl, best_k_coarse + config['fine_search_range'])
    fine_candidates = [k for k in range(low, high+1) if k not in results]
    if fine_candidates:
        print(f'\n===== 精细搜索 (k={low}~{high}) =====')
        print('k值\tSNR(dB)\t能量(%)\tΔSNR(dB)\t模式')
        for k in tqdm(fine_candidates, desc='精细搜索'):
            if mode == 'hard':
                A_recon = U[:, :k] @ (S_diag[:k, np.newaxis] * Vt[:k, :])
            elif mode == 'hybrid':
                S_hybrid = np.zeros_like(S_diag)
                S_hybrid[:k] = np.maximum(S_diag[:k] - soft_thresh, 0)
                A_recon = U @ (S_hybrid[:, np.newaxis] * Vt)

            y_recon = overlap_add_with_window(A_recon.T, window_sum, ov, len(y_clean), max_gain=5.0)
            if np.any(np.isnan(y_recon)) or np.any(np.isinf(y_recon)):
                y_recon = np.nan_to_num(y_recon, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = np.max(np.abs(y_recon))
            if max_abs > 0.01:
                y_recon /= max_abs
            elif max_abs > 0:
                y_recon = np.zeros_like(y_recon)

            snr, delta, ener = _calc_metrics(y_clean, y_recon, energy_cum, k, SNR_original)
            results[k] = {'y': y_recon, 'snr': snr, 'ener': ener, 'delta': delta}
            print(f'{k:2d}\t{snr:.2f}\t{ener:.1f}\t+{delta:.2f}\t{mode}')

    # 最终最优
    best_k = max(results, key=lambda x: results[x]['snr'])
    best = results[best_k]
    print(f'\n✅ 全局最优 k = {best_k}, SNR = {best["snr"]:.2f} dB, '
          f'能量保留率 = {best["ener"]:.1f}%, ΔSNR = +{best["delta"]:.2f} dB')

    return {
        'best_k': best_k,
        'best_y': best['y'],
        'best_snr': best['snr'],
        'best_ener': best['ener'],
        'k_candidates': sorted(results.keys()),
        'snr_values': [results[k]['snr'] for k in sorted(results.keys())],
        'energy_values': [results[k]['ener'] for k in sorted(results.keys())],
        'sing_vals': sing_vals,
        'energy_cum': energy_cum,
        'U': U, 'S_diag': S_diag, 'Vt': Vt,
        'window_sum': window_sum,
        'noise_std': noise_std,
        'soft_thresh': soft_thresh,
        'mode': mode,
    }

def _calc_metrics(y_clean, y_recon, energy_cum, k, SNR_original):
    """计算SNR、ΔSNR、能量保留率"""
    snr = 20 * np.log10(np.linalg.norm(y_clean) / (np.linalg.norm(y_recon - y_clean) + 1e-12))
    delta = snr - SNR_original
    ener = energy_cum[min(k, len(energy_cum)) - 1] * 100
    return snr, delta, ener

# ====================== 可视化模块 ======================
def plot_results(t, y_clean, y_noisy, y_denoised, Fs, sing_vals,
                 k_best, energy_cum, k_list, snr_list, ener_list, SNR_original, mode):
    """绘制9子图核心分析"""
    plt.figure(figsize=(16, 12))

    # (1) 奇异值分布
    plt.subplot(3, 3, 1)
    plt.semilogy(sing_vals, 'b-', lw=2)
    plt.plot(k_best, sing_vals[k_best-1], 'ro', ms=8)
    plt.xlabel('奇异值序号'); plt.ylabel('奇异值大小')
    plt.title(f'奇异值分布（对数）· 模式:{mode}'); plt.grid(alpha=0.3)
    plt.legend(['奇异值', f'最优k={k_best}'])

    # (2) SNR vs k
    plt.subplot(3, 3, 2)
    plt.plot(k_list, snr_list, 'b-o', lw=2, ms=6)
    plt.axhline(y=SNR_original, color='r', ls='--', lw=1.5)
    plt.xlabel('k值'); plt.ylabel('SNR (dB)')
    plt.title('SNR vs k'); plt.grid(alpha=0.3)
    plt.legend(['重构SNR', '含噪SNR'])

    # (3) 累积能量
    plt.subplot(3, 3, 3)
    plt.plot(np.arange(1, len(energy_cum)+1), energy_cum*100, 'b-', lw=2)
    plt.plot(k_best, energy_cum[k_best-1]*100, 'ro', ms=8)
    plt.xlabel('奇异值个数'); plt.ylabel('能量保留率(%)')
    plt.title('累积能量曲线'); plt.ylim([0, 105]); plt.grid(alpha=0.3)
    plt.legend(['累积能量', f'k={k_best}'])

    # (4) 原始波形
    plt.subplot(3, 3, 4)
    plt.plot(t, y_clean, 'b', lw=1)
    plt.xlabel('时间(s)'); plt.ylabel('振幅')
    plt.title('原始音频'); plt.xlim([0, t[-1]]); plt.grid(alpha=0.3)

    # (5) 含噪波形
    plt.subplot(3, 3, 5)
    plt.plot(t, y_noisy, 'r', lw=0.8)
    plt.xlabel('时间(s)'); plt.ylabel('振幅')
    plt.title(f'含噪音频 (noise={CONFIG["noise_level"]})')
    plt.xlim([0, t[-1]]); plt.grid(alpha=0.3)

    # (6) 降噪波形对比
    plt.subplot(3, 3, 6)
    plt.plot(t, y_denoised, 'g', lw=1.2, label='降噪后')
    plt.plot(t, y_clean, 'b--', lw=0.8, label='原始')
    plt.xlabel('时间(s)'); plt.ylabel('振幅')
    plt.title(f'最优降噪 (k={k_best}, SNR={snr_list[k_list.index(k_best)]:.1f}dB)')
    plt.legend(); plt.xlim([0, t[-1]]); plt.grid(alpha=0.3)

    # (7) 频谱对比
    plt.subplot(3, 3, 7)
    N = len(y_clean)
    f = np.fft.rfftfreq(N, d=1/Fs)
    Y_orig = 20 * np.log10(np.abs(np.fft.rfft(y_clean)/N) + 1e-12)
    Y_recon = 20 * np.log10(np.abs(np.fft.rfft(y_denoised)/N) + 1e-12)
    plt.plot(f, Y_orig, 'b--', lw=1.2, label='原始')
    plt.plot(f, Y_recon, 'g-', lw=1, label='降噪后')
    plt.xlabel('频率(Hz)'); plt.ylabel('幅度(dB)')
    plt.title('频谱对比'); plt.xlim([0, 4000])
    plt.legend(); plt.grid(alpha=0.3)

    # (8) 残差波形
    plt.subplot(3, 3, 8)
    residual = y_denoised - y_clean
    plt.plot(t, residual, 'purple', lw=0.8)
    plt.xlabel('时间(s)'); plt.ylabel('振幅')
    plt.title(f'残差 (RMS={np.sqrt(np.mean(residual**2)):.4f})')
    plt.xlim([0, t[-1]]); plt.grid(alpha=0.3)

    # (9) SNR & 能量保留率双轴图
    plt.subplot(3, 3, 9)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(k_list, snr_list, 'b-o', label='SNR')
    ax2.plot(k_list, ener_list, 'r-s', label='能量%')
    ax1.set_xlabel('k值')
    ax1.set_ylabel('SNR (dB)', color='b')
    ax2.set_ylabel('能量保留率 (%)', color='r')
    ax1.grid(alpha=0.3)
    plt.title('SNR vs 能量保留率')

    plt.tight_layout()
    plt.show()

def plot_residual_analysis(t, y_clean, y_denoised, Fs):
    """独立残差分析图：差值波形 + 差值频谱 + 直方图"""
    residual = y_denoised - y_clean
    rms = np.sqrt(np.mean(residual**2))
    max_abs = np.max(np.abs(residual))

    fig = plt.figure(figsize=(14, 6))
    
    # 残差波形
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(t, residual, 'purple', lw=0.8)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('振幅')
    ax1.set_title(f'残差波形 (RMS={rms:.4f}, Max|A|={max_abs:.3f})')
    ax1.set_xlim([0, t[-1]])
    ax1.grid(alpha=0.3)

    # 残差频谱
    ax2 = fig.add_subplot(1, 3, 2)
    N = len(y_clean)
    f = np.fft.rfftfreq(N, d=1/Fs)
    Y_res = 20 * np.log10(np.abs(np.fft.rfft(residual)/N) + 1e-12)
    ax2.plot(f, Y_res, 'orange', lw=1)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.set_title('残差频谱')
    ax2.set_xlim([0, 4000])
    ax2.grid(alpha=0.3)

    # 残差幅度分布直方图
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(residual, bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('振幅')
    ax3.set_ylabel('频次')
    ax3.set_title('残差幅度分布')

    plt.tight_layout()
    plt.show()
    return fig

# ====================== 主程序 ======================
if __name__ == '__main__':
    # ---------- 1. 加载/生成纯净音频 ----------
    mat_file = Path('train.mat')
    if mat_file.exists():
        data = loadmat(mat_file)
        y = data['y'].flatten()
        Fs = int(data['Fs'])
        print(f'已加载 {mat_file} (Fs={Fs} Hz)')
    else:
        print(f'{mat_file} 不存在，生成测试信号...')
        Fs = CONFIG['fs']
        y = generate_test_signal(Fs, CONFIG['duration'])
        savemat(mat_file, {'y': y, 'Fs': Fs})
        print('已生成 train.mat')

    y = ensure_mono(y, Fs)
    t = np.arange(len(y)) / Fs
    print(f'采样率: {Fs} Hz | 时长: {len(y)/Fs:.2f} s | 样本数: {len(y)}')

    # ---------- 2. 加噪 + 直流偏移修复 ----------
    np.random.seed(42)
    noise = CONFIG['noise_level'] * np.random.randn(len(y))
    y_noisy = y + noise
    y_noisy = y_noisy - np.mean(y_noisy)  # 自动修复直流偏移
    SNR_original = 20 * np.log10(np.linalg.norm(y) / (np.linalg.norm(y_noisy - y) + 1e-12))
    print(f'含噪信号 SNR (直流修正后): {SNR_original:.2f} dB')

    # ---------- 3. 播放（可选）----------
    if CONFIG['play_enabled']:
        print('播放原始音频...')
        sd.play(y, Fs); sd.wait()
        print('播放含噪音频...')
        sd.play(y_noisy, Fs); sd.wait()

    # ---------- 4. SVD 降噪 ----------
    result = svd_denoise(y_noisy, y, Fs, CONFIG)

    # ---------- 5. 可视化 ----------
    if CONFIG['plot_enabled']:
        plot_results(t, y, y_noisy, result['best_y'], Fs,
                     result['sing_vals'], result['best_k'],
                     result['energy_cum'],
                     result['k_candidates'],
                     result['snr_values'],
                     result['energy_values'],
                     SNR_original,
                     result['mode'])
        
        if CONFIG['plot_residual_alone']:
            plot_residual_analysis(t, y, result['best_y'], Fs)

    # ---------- 6. 播放降噪音频 ----------
    if CONFIG['play_enabled']:
        print(f'播放最优降噪音频 (k={result["best_k"]})...')
        sd.play(result['best_y'], Fs); sd.wait()

    # ---------- 7. 保存WAV ----------
    if CONFIG['save_wav']:
        wavfile.write('original.wav', Fs, y)
        wavfile.write('noisy.wav', Fs, y_noisy)
        wavfile.write(f'denoised_k{result["best_k"]}_{result["mode"]}.wav', Fs, result['best_y'])
        print('\nWAV文件已保存至当前目录。')

    # ---------- 8. 效果验证 ----------
    residual = result['best_y'] - y
    rms_res = np.sqrt(np.mean(residual**2))
    print('\n===== 最终降噪效果 =====')
    print(f'降噪模式: {result["mode"]}')
    print(f'软阈值因子: {CONFIG["soft_thresh_factor"]} (噪声标准差={result["noise_std"]:.4f}, 实际阈值={result["soft_thresh"]:.4f})')
    print(f'最优 k = {result["best_k"]}')
    print(f'降噪后 SNR = {result["best_snr"]:.2f} dB')
    print(f'信噪比提升 = {result["best_snr"] - SNR_original:.2f} dB')
    print(f'残差 RMS = {rms_res:.4f}')