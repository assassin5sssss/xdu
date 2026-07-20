"""
KSVD & SVD 综合应用平台
整合功能：
1. KSVD图像修复（划痕/污渍）
2. SVD音频降噪（增强版）
3. SVD图像去噪评估
"""

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import sys
import tempfile
import warnings
from pathlib import Path
import os
from skimage import color

# ==================== 导入各模块功能 ====================
try:
    import scratch
    import stain
except ImportError:
    st.error("❌ 缺少必要模块 scratch.py 或 stain.py，请确保它们在同一目录下")
    st.stop()

warnings.filterwarnings('ignore')
# 图像修复模块
import stain
import scratch

# 音频处理模块
from scipy.io import wavfile
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="矩阵分析与应用",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #667eea;
    }
    .training-log {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: monospace;
        padding: 15px;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.85em;
    }
    .badge { 
        padding: 4px 8px; 
        border-radius: 4px; 
        border: 1px solid; 
        background: #1E1E1E; 
        margin-right: 10px; 
        font-family: monospace; 
        font-size: 0.9em; 
        color: #FFF; 
        display: inline-block; 
        margin-bottom: 5px; 
    }
</style>
""", unsafe_allow_html=True)

# ==================== 标题区域 ====================
st.markdown("""
<div class="main-header">
    <h1>🎯 矩阵分析与应用</h1>
</div>
""", unsafe_allow_html=True)

# ==================== 侧边栏导航 ====================
with st.sidebar:
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top:0; color:#000000">🎓 课程设计项目</h3>
        <p style="color:#000000"><b>课程名称:</b> 矩阵分析与计算（X2MS1012） </p>
        <p style="color:#000000"><b>指导老师:</b> 尹小艳 </p>
        <p style="color:#000000"><b>论文题目:</b> 《SVD在多场景数据处理中的应用与结果分析》 </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📌 功能导航")
    app_mode = st.radio(
        "选择应用模块",
        ["🖼️ KSVD图像修复", "🎵 SVD音频降噪", "📊 SVD图像去噪"],
        help="选择要使用的功能模块"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ 通用设置")
    
    # GPU设置
    use_gpu = st.checkbox("启用GPU加速 (CUDA)", value=torch.cuda.is_available())
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        st.success(f"✅ GPU已启用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        st.info("ℹ️ 使用CPU模式")
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    with st.expander("查看说明"):
        st.markdown("""
        **图像修复模块**：
        - 支持划痕修复和不规则污渍修复
        - 基于KSVD字典学习算法
        - 包含边缘精修功能
        
        **音频降噪模块**：
        - 基于SVD分帧处理
        - 支持混合阈值降噪
        - 包含残差分析
        
        **图像去噪评估**：
        - 支持多种噪声类型
        - SVD与中值滤波对比
        - PSNR/SSIM指标评估
        """)

# ==================== 音频处理函数（从appex.py替换） ====================

def ensure_mono(y, Fs):
    """确保信号为单声道"""
    if y.ndim > 1:
        if y.shape[1] == 2:
            y = np.mean(y, axis=1)
        else:
            raise ValueError(f"不支持的声道数: {y.shape[1]}")
    return y

def generate_test_signal(fs=8000, duration=10):
    """生成测试信号"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + \
        0.3 * np.sin(2 * np.pi * 880 * t) + \
        0.2 * np.sin(2 * np.pi * 1320 * t)
    attack = int(0.1 * fs)
    release = int(0.2 * fs)
    envelope = np.ones_like(y)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    return y * envelope, fs

def overlap_add_with_window(frames, window_sum, overlap, target_len, max_gain=5.0):
    """重叠相加法"""
    frame_len, n_frames = frames.shape
    step = frame_len - overlap
    total_len = frame_len + (n_frames - 1) * step
    signal = np.zeros(total_len)
    for i in range(n_frames):
        start = i * step
        signal[start:start + frame_len] += frames[:, i]
    window_sum = window_sum[:target_len]
    signal = signal[:target_len]
    eps = 1e-6
    safe_divisor = np.maximum(window_sum, eps)
    gain = signal / safe_divisor
    max_gain_mask = safe_divisor < (1.0 / max_gain)
    gain[max_gain_mask] = signal[max_gain_mask] * max_gain
    return gain

def svd_denoise_audio(y_noisy, y_clean, Fs, params, SNR_original):
    """SVD音频降噪主流程"""
    fl = params['frame_length']
    ov = params['overlap']
    step = fl - ov
    mode = params['svd_mode']

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
    A = frames.T

    U, S_diag, Vt = np.linalg.svd(A, full_matrices=False)
    sing_vals = S_diag.copy()
    total_energy = np.sum(sing_vals ** 2)
    energy_cum = np.cumsum(sing_vals ** 2) / total_energy

    noise_std = np.std(sing_vals[-params['noise_tail_len']:])
    soft_thresh = min(params['soft_thresh_factor'] * noise_std,
                      params.get('soft_thresh_max', 0.2))
    noise_thresh = 3 * noise_std

    energy_thresh_idx = np.where(energy_cum >= params['energy_thresh'])[0][0] + 1
    hard_thresh_idx = np.sum(sing_vals > noise_thresh)
    k_candidates = set(params['fixed_k'])
    k_candidates.add(min(fl, energy_thresh_idx + 20))
    k_candidates.add(min(fl, hard_thresh_idx))
    k_candidates = sorted(k_candidates)

    results = {}
    for k in k_candidates:
        if mode == 'hard':
            A_recon = U[:, :k] @ (S_diag[:k, np.newaxis] * Vt[:k, :])
        else:
            S_hybrid = np.zeros_like(S_diag)
            S_hybrid[:k] = np.maximum(S_diag[:k] - soft_thresh, 0)
            A_recon = U @ (S_hybrid[:, np.newaxis] * Vt)

        y_recon = overlap_add_with_window(A_recon.T, window_sum, ov, len(y_clean), max_gain=5.0)
        y_recon = np.nan_to_num(y_recon, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = np.max(np.abs(y_recon))
        if max_abs > 0.01:
            y_recon /= max_abs
        elif max_abs > 0:
            y_recon = np.zeros_like(y_recon)

        snr = 20 * np.log10(np.linalg.norm(y_clean) / (np.linalg.norm(y_recon - y_clean) + 1e-12))
        delta = snr - SNR_original
        ener = energy_cum[min(k, len(energy_cum)) - 1] * 100
        results[k] = {'y': y_recon, 'snr': snr, 'ener': ener, 'delta': delta}

    best_k_coarse = max(results, key=lambda x: results[x]['snr'])
    low = max(1, best_k_coarse - params['fine_search_range'])
    high = min(fl, best_k_coarse + params['fine_search_range'])
    fine_candidates = [k for k in range(low, high+1) if k not in results]
    
    for k in fine_candidates:
        if mode == 'hard':
            A_recon = U[:, :k] @ (S_diag[:k, np.newaxis] * Vt[:k, :])
        else:
            S_hybrid = np.zeros_like(S_diag)
            S_hybrid[:k] = np.maximum(S_diag[:k] - soft_thresh, 0)
            A_recon = U @ (S_hybrid[:, np.newaxis] * Vt)

        y_recon = overlap_add_with_window(A_recon.T, window_sum, ov, len(y_clean), max_gain=5.0)
        y_recon = np.nan_to_num(y_recon, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = np.max(np.abs(y_recon))
        if max_abs > 0.01:
            y_recon /= max_abs
        elif max_abs > 0:
            y_recon = np.zeros_like(y_recon)

        snr = 20 * np.log10(np.linalg.norm(y_clean) / (np.linalg.norm(y_recon - y_clean) + 1e-12))
        delta = snr - SNR_original
        ener = energy_cum[min(k, len(energy_cum)) - 1] * 100
        results[k] = {'y': y_recon, 'snr': snr, 'ener': ener, 'delta': delta}

    best_k = max(results, key=lambda x: results[x]['snr'])
    best = results[best_k]

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
        'noise_std': noise_std,
        'soft_thresh': soft_thresh,
        'noise_thresh': noise_thresh,
        'mode': mode,
    }

def numpy_to_wav_bytes(y, Fs):
    """将numpy数组转换为WAV格式"""
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, Fs, y.astype(np.float32))
    bytes_io.seek(0)
    return bytes_io

def create_waveform_plot(t, y, title, color):
    """创建波形图"""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(t, y, color=color, linewidth=1)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('振幅')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2)
    return fig

def create_spectrum_plot(y, Fs, title, color):
    """创建频谱图"""
    N = len(y)
    f = np.fft.rfftfreq(N, d=1/Fs)
    Y = 20 * np.log10(np.abs(np.fft.rfft(y)/N) + 1e-12)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(f, Y, color=color, linewidth=1)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('幅度 (dB)')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0, 4000])
    ax.grid(True, alpha=0.2)
    return fig

def create_singular_value_plot(sing_vals, k_best, thresh=None):
    """创建奇异值分布图"""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(sing_vals, 'b-', linewidth=2, label='奇异值')
    ax.axvline(x=k_best-1, color='r', linestyle='--', linewidth=1, label=f'最优k={k_best}')
    if thresh:
        ax.axhline(y=thresh, color='orange', linestyle=':', linewidth=1, label=f'3σ阈值={thresh:.3f}')
    ax.set_xlabel('奇异值序号')
    ax.set_ylabel('奇异值大小')
    ax.set_title('奇异值分布（对数）', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig

def plot_residual_analysis(t, y_clean, y_denoised, Fs):
    """残差分析图"""
    residual = y_denoised - y_clean
    rms = np.sqrt(np.mean(residual**2))

    fig = plt.figure(figsize=(14, 4))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(t, residual, 'purple', lw=0.8)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('振幅')
    ax1.set_title(f'残差波形 (RMS={rms:.4f})', fontweight='bold')
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(1, 3, 2)
    N = len(y_clean)
    f = np.fft.rfftfreq(N, d=1/Fs)
    Y_res = 20 * np.log10(np.abs(np.fft.rfft(residual)/N) + 1e-12)
    ax2.plot(f, Y_res, 'orange', lw=1)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.set_title('残差频谱', fontweight='bold')
    ax2.set_xlim([0, 4000])
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(residual, bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('振幅')
    ax3.set_ylabel('频次')
    ax3.set_title('残差幅度分布', fontweight='bold')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig

# ==================== 根据选择加载不同模块 ====================
# ========== 1. KSVD图像修复模块（完全参照repair.py，划痕和污渍完全分开）==========
if app_mode == "🖼️ KSVD图像修复":
    st.header("🖼️ KSVD图像强化修复系统")
    st.caption("基于KSVD字典学习 + 动态稀疏度衰减 + 边缘精修算法 | 支持手动绘制划痕和自动生成污渍")
    
    # ==================== 初始化session state ====================
    if 'repair_stage' not in st.session_state:
        st.session_state.repair_stage = 'mode_selection'  # mode_selection, scratch_mode, stain_mode
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'scratch_stage' not in st.session_state:
        st.session_state.scratch_stage = 'upload'  # upload, damage, repair, result
    if 'stain_stage' not in st.session_state:
        st.session_state.stain_stage = 'upload'  # upload, damage, repair, result
    if 'img_original' not in st.session_state:
        st.session_state.img_original = None  # 存储彩色图像 (RGB)
    if 'img_gray' not in st.session_state:
        st.session_state.img_gray = None      # 存储灰度图像 (用于处理)
    if 'img_damaged' not in st.session_state:
        st.session_state.img_damaged = None   # 存储灰度损坏图像
    if 'img_damaged_color' not in st.session_state:
        st.session_state.img_damaged_color = None  # 存储彩色损坏图像
    if 'mask' not in st.session_state:
        st.session_state.mask = None
    if 'repair_result' not in st.session_state:
        st.session_state.repair_result = None  # 存储灰度修复结果
    if 'repair_result_color' not in st.session_state:
        st.session_state.repair_result_color = None  # 存储彩色修复结果
    
    # ==================== 重置状态函数 ====================
    def reset_all():
        st.session_state.repair_stage = 'mode_selection'
        st.session_state.mode = None
        st.session_state.scratch_stage = 'upload'
        st.session_state.stain_stage = 'upload'
        st.session_state.img_original = None
        st.session_state.img_gray = None
        st.session_state.img_damaged = None
        st.session_state.img_damaged_color = None
        st.session_state.mask = None
        st.session_state.repair_result = None
        st.session_state.repair_result_color = None
    
    def reset_scratch_state():
        st.session_state.scratch_stage = 'upload'
        st.session_state.img_original = None
        st.session_state.img_gray = None
        st.session_state.img_damaged = None
        st.session_state.img_damaged_color = None
        st.session_state.mask = None
        st.session_state.repair_result = None
        st.session_state.repair_result_color = None
    
    def reset_stain_state():
        st.session_state.stain_stage = 'upload'
        st.session_state.img_original = None
        st.session_state.img_gray = None
        st.session_state.img_damaged = None
        st.session_state.img_damaged_color = None
        st.session_state.mask = None
        st.session_state.repair_result = None
        st.session_state.repair_result_color = None
    
    # ==================== 图像加载函数 ====================
    def load_image_for_repair(uploaded_file):
        if uploaded_file is None:
            return None, None
        
        # 加载彩色图像
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        
        # 转换为灰度图用于处理
        from skimage import color
        img_gray = color.rgb2gray(img_np)
        
        if use_gpu:
            img_gray = torch.from_numpy(img_gray).float().to(device)
            img_np_tensor = torch.from_numpy(img_np).float().to(device)
            return img_gray, img_np_tensor
        else:
            return img_gray, img_np
    
    # ==================== 辅助函数：确保图像为彩色显示 ====================
    def ensure_color_display(img, is_tensor=False):
        """确保图像以彩色RGB格式显示"""
        if img is None:
            return None
        
        # 如果是tensor，转换为numpy
        if is_tensor or (use_gpu and torch.is_tensor(img)):
            img = img.cpu().numpy()
        
        # 确保图像在0-1范围内
        if img.max() > 1.0:
            img = img / 255.0
        
        # 如果是2D（灰度图），转换为RGB
        if len(img.shape) == 2:
            return np.stack([img, img, img], axis=2)
        # 如果是3D但通道数为1
        elif len(img.shape) == 3 and img.shape[2] == 1:
            return np.concatenate([img, img, img], axis=2)
        # 如果是RGBA，取RGB
        elif len(img.shape) == 3 and img.shape[2] == 4:
            return img[:, :, :3]
        # 已经是RGB
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return img
        # 其他情况
        else:
            # 尝试转换为灰度再转RGB
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            return np.stack([gray, gray, gray], axis=2)
    
    # ==================== 辅助函数：转换为灰度图用于计算 ====================
    def to_grayscale_for_metric(img):
        if img is None:
            return None
        if use_gpu and torch.is_tensor(img):
            img = img.cpu().numpy()
        if len(img.shape) == 3:
            if img.shape[0] == 3:  # (C, H, W)
                return np.mean(img, axis=0)
            else:  # (H, W, C)
                return np.mean(img, axis=2)
        else:
            return img
    
    # ==================== 模式选择界面 ====================
    if st.session_state.repair_stage == 'mode_selection':
        st.markdown("### 🔧 请选择修复模式")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✏️ 划痕修复")
            st.markdown("适合修复线条状、细长的图像损伤")
            st.markdown("- 手动绘制划痕区域")
            st.markdown("- 适用于划痕、折痕等")
            st.markdown("- 使用scratch模块")
            if st.button("选择划痕修复", use_container_width=True, type="primary"):
                st.session_state.repair_stage = 'scratch_mode'
                st.session_state.mode = 'scratch'
                st.session_state.scratch_stage = 'upload'
                st.rerun()
        
        with col2:
            st.markdown("#### 🧪 污渍修复")
            st.markdown("适合修复面状、区域性的图像污渍")
            st.markdown("- 自动生成随机污渍")
            st.markdown("- 适用于污渍、斑点等")
            st.markdown("- 使用stain模块")
            if st.button("选择污渍修复", use_container_width=True, type="primary"):
                st.session_state.repair_stage = 'stain_mode'
                st.session_state.mode = 'stain'
                st.session_state.stain_stage = 'upload'
                st.rerun()
    
    # ==================== 划痕修复流程 ====================
    elif st.session_state.repair_stage == 'scratch_mode':
        
        # 步骤1: 上传图像
        if st.session_state.scratch_stage == 'upload':
            st.markdown("### ✏️ 划痕修复 - 步骤1/4: 上传图像")
            
            uploaded_file = st.file_uploader("选择图像文件", type=['png', 'jpg'], key="scratch_upload")
            
            if uploaded_file is not None:
                img_gray, img_color = load_image_for_repair(uploaded_file)
                st.session_state.img_gray = img_gray
                st.session_state.img_original = img_color
                
                st.success("✅ 图像上传成功！")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➡️ 下一步：绘制划痕", use_container_width=True, type="primary"):
                        st.session_state.scratch_stage = 'damage'
                        st.rerun()
                with col2:
                    if st.button("🔙 返回模式选择"):
                        reset_all()
                        st.rerun()
            
            # 显示预览
            if st.session_state.img_original is not None:
                img_display = ensure_color_display(st.session_state.img_original)
                st.image(img_display, caption="已上传图像 (彩色)", use_container_width=True)
        
        # 步骤2: 绘制划痕
        elif st.session_state.scratch_stage == 'damage':
            st.markdown("### ✏️ 划痕修复 - 步骤2/4: 绘制划痕")
            
            # 导入必要的模块
            import matplotlib
            try:
                matplotlib.use('TkAgg')
            except:
                pass
            
            # 显示彩色原始图像
            img_color_display = ensure_color_display(st.session_state.img_original)
            st.image(img_color_display, caption="原始图像 (彩色)", use_container_width=True)
            
            st.markdown("---")
            st.info("点击下方按钮打开独立画布窗口，在弹出窗口中用鼠标左键拖动绘制划痕，按Enter或ESC键保存")
            
            # 获取灰度图用于绘制
            img_gray = st.session_state.img_gray
            if use_gpu and torch.is_tensor(img_gray):
                img_gray = img_gray.cpu().numpy()
            
            # 手动绘制划痕按钮
            if st.button("🖌️ 打开画布绘制划痕 (按Enter保存)", use_container_width=True, type="primary"):
                try:
                    # 处理图像格式
                    from skimage import color
                    if len(img_gray.shape) == 3:
                        if img_gray.shape[2] == 4:  # RGBA
                            display_img = color.rgb2gray(img_gray[:, :, :3])
                        elif img_gray.shape[2] == 3:  # RGB
                            display_img = color.rgb2gray(img_gray)
                        else:
                            display_img = np.mean(img_gray, axis=2)
                    else:
                        display_img = img_gray
                    
                    # 调用scratch模块的交互式绘制函数
                    mask, area, percentage = scratch.interactive_draw_scratches(display_img, use_gpu, device)
                    
                    # 保存掩码
                    st.session_state.mask = mask
                    
                    # 创建损坏图像（彩色版本）
                    if use_gpu and torch.is_tensor(st.session_state.img_original):
                        img_damaged_color = st.session_state.img_original.clone()
                        # 在彩色图像的每个通道上应用掩码
                        for c in range(img_damaged_color.shape[2]):
                            channel = img_damaged_color[:, :, c]
                            channel[mask] = 0
                            img_damaged_color[:, :, c] = channel
                        
                        # 同时创建灰度损坏图像用于处理
                        img_damaged_gray = st.session_state.img_gray.clone()
                        img_damaged_gray[mask] = 0
                    else:
                        img_damaged_color = st.session_state.img_original.copy()
                        for c in range(img_damaged_color.shape[2]):
                            img_damaged_color[:, :, c][mask] = 0
                        
                        img_damaged_gray = st.session_state.img_gray.copy()
                        img_damaged_gray[mask] = 0
                    
                    st.session_state.img_damaged = img_damaged_gray
                    st.session_state.img_damaged_color = img_damaged_color
                    
                    st.success(f"✅ 划痕绘制完成！面积: {area} 像素 ({percentage:.2f}%)")
                    st.session_state.scratch_stage = 'repair'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"绘制失败: {e}")
                    st.exception(e)
            
            # 显示当前划痕预览（如果已经绘制过）
            if st.session_state.get('mask') is not None and st.session_state.get('img_damaged_color') is not None:
                st.markdown("---")
                st.subheader("当前划痕预览")
                
                img_damaged_display = ensure_color_display(st.session_state.img_damaged_color)
                
                # 计算划痕面积和占比
                mask_np = st.session_state.mask
                if use_gpu and torch.is_tensor(mask_np):
                    mask_np = mask_np.cpu().numpy()
                
                area = np.sum(mask_np)
                total_pixels = mask_np.size
                percentage = 100 * area / total_pixels
                
                st.image(img_damaged_display, caption=f"带划痕图像 (彩色, 面积: {area} 像素, {percentage:.2f}%)", use_container_width=True)
                
                if st.button("➡️ 下一步：执行修复", use_container_width=True, type="primary"):
                    st.session_state.scratch_stage = 'repair'
                    st.rerun()
            
            # 返回按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔙 返回上一步", use_container_width=True):
                    st.session_state.scratch_stage = 'upload'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择", use_container_width=True):
                    reset_all()
                    st.rerun()
        
        # 步骤3: 执行修复
        elif st.session_state.scratch_stage == 'repair':
            st.markdown("### ✏️ 划痕修复 - 步骤3/4: 执行修复")
            
            # 显示图像对比
            col1, col2 = st.columns(2)
            with col1:
                img_orig_display = ensure_color_display(st.session_state.img_original)
                st.image(img_orig_display, caption="原始图像 (彩色)", use_container_width=True)
            with col2:
                if st.session_state.img_damaged_color is not None:
                    img_damaged_display = ensure_color_display(st.session_state.img_damaged_color)
                else:
                    img_damaged_display = ensure_color_display(st.session_state.img_damaged)
                st.image(img_damaged_display, caption="带划痕图像", use_container_width=True)
            
            st.markdown("---")
            
            # 修复参数设置
            st.subheader("🔧 ")
            
            patch_size = 8
            dict_size = 64
            sparsity = 4
            max_iter = 1
            step = 6
            repair_iterations = 6
            
            if st.button("🚀 开始划痕修复", use_container_width=True, type="primary"):
                with st.spinner("正在修复划痕..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 准备数据
                    img_damaged_gray = st.session_state.img_damaged
                    mask_scratches = st.session_state.mask
                    
                    # 创建修复掩码 - 完好区域为True，划痕区域为False
                    mask = ~mask_scratches
                    
                    # 步骤1: 提取完好区域的图像块
                    status_text.text("1/6: 提取图像块...")
                    patches, patch_pos = scratch.extract_background_patches_scratches(
                        img_damaged_gray, mask, patch_size, step, use_gpu, device
                    )
                    progress_bar.progress(15)
                    
                    if patches.size == 0:
                        st.error("错误：未能提取到图像块！")
                        st.stop()
                    
                    # 如果提取的块太少，调整步长
                    if patches.shape[1] < dict_size * 2:
                        st.caption("完好块不足，放宽提取条件")
                        step = max(1, step - 1)
                        patches, patch_pos = scratch.extract_background_patches_scratches(
                            img_damaged_gray, mask, patch_size, step, use_gpu, device
                        )
                    progress_bar.progress(30)
                    
                    # 步骤2: KSVD字典训练
                    status_text.text("2/6: 训练字典...")
                    start_time = time.time()
                    
                    dictionary, alpha = scratch.improved_ksvd_for_scratches(
                        patches, dict_size, sparsity, max_iter, use_gpu, device
                    )
                    
                    training_time = time.time() - start_time
                    st.caption(f"字典训练完成，耗时 {training_time:.2f} 秒")
                    progress_bar.progress(50)
                    
                    # 步骤3: 初始化填充
                    status_text.text("3/6: 初始化填充...")
                    img_curr_gray = scratch.fill_scratches_with_neighbors_gpu(
                        img_damaged_gray, mask_scratches, use_gpu, device
                    )
                    progress_bar.progress(65)
                    
                    # 步骤4: 迭代修复（带稀疏度演变）
                    status_text.text("4/6: 迭代修复...")
                    
                    # 设置稀疏度演变逻辑
                    start_sparsity = sparsity
                    end_sparsity = max(1, sparsity // 2)
                    
                    for iter_idx in range(repair_iterations):
                        # 动态计算当前轮次的稀疏度
                        if repair_iterations > 1:
                            current_sparsity = int(start_sparsity - (iter_idx / (repair_iterations - 1)) * (start_sparsity - end_sparsity))
                        else:
                            current_sparsity = start_sparsity
                        
                        # 步长逐渐减小以提高精细度
                        current_step = max(1, step - iter_idx // 2)
                        
                        st.caption(f"  第 {iter_idx+1}/{repair_iterations} 轮: 稀疏度={current_sparsity}, 步长={current_step}")
                        
                        # 调用修复函数
                        img_curr_gray = scratch.repair_scratches_iterative_gpu(
                            img_curr_gray, mask_scratches, dictionary, patch_size,
                            current_sparsity, current_step, use_gpu, device, iter_idx + 1
                        )
                        
                        # 更新进度
                        progress_bar.progress(65 + int(25 * (iter_idx + 1) / repair_iterations))
                    
                    progress_bar.progress(90)
                    
                    # 步骤5: 最终结果
                    status_text.text("5/6: 后处理...")
                    img_repaired_gray = img_curr_gray
                    
                    # 创建彩色修复结果
                    if use_gpu and torch.is_tensor(st.session_state.img_original):
                        img_repaired_color = st.session_state.img_original.clone()
                        if torch.is_tensor(img_repaired_gray):
                            repaired_gray_np = img_repaired_gray.cpu().numpy()
                        else:
                            repaired_gray_np = img_repaired_gray
                        
                        img_original_np = st.session_state.img_original.cpu().numpy()
                        
                        # 对于彩色图像，将修复的灰度值应用到每个通道
                        for c in range(3):
                            channel = img_repaired_color[:, :, c].cpu().numpy()
                            if mask_scratches.sum() > 0:
                                # 使用灰度值作为亮度，保持原始颜色的比例
                                channel[mask_scratches] = repaired_gray_np[mask_scratches]
                            img_repaired_color[:, :, c] = torch.from_numpy(channel).float().to(device)
                    else:
                        img_repaired_color = st.session_state.img_original.copy()
                        repaired_gray_np = img_repaired_gray if not use_gpu else img_repaired_gray.cpu().numpy()
                        for c in range(3):
                            img_repaired_color[:, :, c][mask_scratches] = repaired_gray_np[mask_scratches]
                    
                    progress_bar.progress(100)
                    status_text.text("修复完成！")
                    time.sleep(0.5)
                    
                    st.session_state.repair_result = img_repaired_gray
                    st.session_state.repair_result_color = img_repaired_color
                    st.session_state.scratch_stage = 'result'
                    st.rerun()
            
            # 返回按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔙 返回上一步", use_container_width=True):
                    st.session_state.scratch_stage = 'damage'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择", use_container_width=True):
                    reset_all()
                    st.rerun()
        
        # 步骤4: 显示结果
        elif st.session_state.scratch_stage == 'result':
            st.markdown("### ✏️ 划痕修复 - 步骤4/4: 修复结果")
            
            # 准备数据用于显示
            if use_gpu:
                img_original_cpu = st.session_state.img_gray.cpu().numpy() if torch.is_tensor(st.session_state.img_gray) else st.session_state.img_gray
                img_repaired_gray = st.session_state.repair_result.cpu().numpy() if torch.is_tensor(st.session_state.repair_result) else st.session_state.repair_result
                mask_scratches_cpu = st.session_state.mask.cpu().numpy() if torch.is_tensor(st.session_state.mask) else st.session_state.mask
            else:
                img_original_cpu = st.session_state.img_gray
                img_repaired_gray = st.session_state.repair_result
                mask_scratches_cpu = st.session_state.mask
            
            # 处理掩码
            if len(mask_scratches_cpu.shape) == 3:
                mask_np = mask_scratches_cpu[:, :, 0] > 0
            else:
                mask_np = mask_scratches_cpu > 0
            
            # 计算划痕统计
            scratch_area = np.sum(mask_np)
            total_pixels = mask_np.size
            scratch_percentage = 100 * scratch_area / total_pixels
            
            # 显示三列彩色对比
            col1, col2, col3 = st.columns(3)
            with col1:
                img_orig_display = ensure_color_display(st.session_state.img_original)
                st.image(img_orig_display, caption="原始彩色图像", use_container_width=True)
            with col2:
                img_damaged_display = ensure_color_display(st.session_state.img_damaged_color)
                st.image(img_damaged_display, caption=f"带划痕图像 ({scratch_percentage:.2f}% 像素损坏)", use_container_width=True)
            with col3:
                img_result_display = ensure_color_display(st.session_state.repair_result_color)
                st.image(img_result_display, caption="KSVD修复结果", use_container_width=True)
            
            st.markdown("---")
            
            # ========== 计算质量指标（使用灰度图计算）==========
            st.subheader("📊 质量评估报告")
            
            if scratch_area > 0:
                # 获取修复区域的像素值
                repaired_values = img_repaired_gray[mask_np]
                original_values = img_original_cpu[mask_np]
                
                # 验证：检查修复区域是否有变化
                max_diff_scratch = np.max(np.abs(repaired_values - original_values))
                
                # 计算修复区域差异
                diff_values = repaired_values - original_values
                abs_diff_values = np.abs(diff_values)
                
                # 修复区域统计
                mae_repair = np.mean(abs_diff_values)
                max_ae_repair = np.max(abs_diff_values)
                mse_repair = np.mean(diff_values ** 2)
                
                # 修复区域PSNR
                if mse_repair > 0:
                    psnr_repair = 10 * np.log10(1.0 / mse_repair)
                else:
                    psnr_repair = float('inf')
                
                # 计算相对误差
                original_norm = np.linalg.norm(original_values)
                if original_norm > 0:
                    relative_error = 100 * np.linalg.norm(diff_values) / original_norm
                else:
                    relative_error = 0
                
                # 计算SSIM指标
                try:
                    from skimage.metrics import structural_similarity as ssim_skimage
                    
                    h, w = img_repaired_gray.shape
                    
                    # 整体SSIM计算
                    if h >= 7 and w >= 7:
                        data_range = max(img_original_cpu.max(), img_repaired_gray.max()) - min(img_original_cpu.min(), img_repaired_gray.min())
                        if data_range <= 0:
                            data_range = 1.0
                        
                        win_size = min(7, h, w)
                        if win_size % 2 == 0:
                            win_size -= 1
                        if win_size < 3:
                            win_size = 3
                        
                        ssim_overall = ssim_skimage(
                            img_original_cpu, img_repaired_gray,
                            data_range=data_range,
                            win_size=win_size,
                            gaussian_weights=True,
                            sigma=1.5,
                            use_sample_covariance=False,
                            channel_axis=None
                        )
                    else:
                        ssim_overall = np.nan
                    
                    # 修复区域SSIM计算
                    if scratch_area > 0:
                        rows, cols = np.where(mask_np > 0)
                        if len(rows) > 0:
                            r_min = max(0, np.min(rows) - 10)
                            r_max = min(img_original_cpu.shape[0], np.max(rows) + 10)
                            c_min = max(0, np.min(cols) - 10)
                            c_max = min(img_original_cpu.shape[1], np.max(cols) + 10)
                            
                            patch_h = r_max - r_min
                            patch_w = c_max - c_min
                            
                            if patch_h >= 7 and patch_w >= 7:
                                patch_ref = img_original_cpu[r_min:r_max, c_min:c_max]
                                patch_comp = img_repaired_gray[r_min:r_max, c_min:c_max]
                                
                                patch_data_range = max(patch_ref.max(), patch_comp.max()) - min(patch_ref.min(), patch_comp.min())
                                if patch_data_range <= 0:
                                    patch_data_range = 1.0
                                
                                win_size = min(7, patch_h, patch_w)
                                if win_size % 2 == 0:
                                    win_size -= 1
                                if win_size < 3:
                                    win_size = 3
                                
                                ssim_patch = ssim_skimage(
                                    patch_ref, patch_comp,
                                    data_range=patch_data_range,
                                    win_size=win_size,
                                    gaussian_weights=True,
                                    sigma=1.5,
                                    use_sample_covariance=False,
                                    channel_axis=None
                                )
                            else:
                                ssim_patch = ssim_overall
                        else:
                            ssim_patch = ssim_overall
                    else:
                        ssim_patch = ssim_overall
                    
                    # 合理性修正
                    if not np.isnan(ssim_overall) and not np.isnan(ssim_patch):
                        if ssim_patch > ssim_overall:
                            ssim_overall, ssim_patch = ssim_patch, ssim_overall
                            
                except Exception as e:
                    print(f"SSIM计算出错: {e}")
                    ssim_overall = np.nan
                    ssim_patch = np.nan
                
                # 显示指标卡片
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("划痕像素", f"{scratch_area}")
                    st.metric("划痕占比", f"{scratch_percentage:.2f}%")
                
                with col2:
                    if not np.isinf(psnr_repair):
                        st.metric("修复区域PSNR", f"{psnr_repair:.2f} dB")
                    else:
                        st.metric("修复区域PSNR", "∞")
                    st.metric("修复区域SSIM", f"{ssim_patch:.4f}")
                
                with col3:
                    st.metric("平均绝对误差", f"{mae_repair:.4f}")
                    st.metric("最大绝对误差", f"{max_ae_repair:.4f}")
                
                with col4:
                    st.metric("相对误差", f"{relative_error:.2f}%")
                    st.metric("修复区域MSE", f"{mse_repair:.6f}")
                
                # 质量评级
                st.markdown("---")
                if not np.isinf(psnr_repair):
                    if psnr_repair > 30:
                        st.success("⭐⭐⭐⭐⭐ 修复质量: 优秀 (PSNR > 30dB)")
                    elif psnr_repair > 25:
                        st.success("⭐⭐⭐⭐ 修复质量: 良好 (PSNR 25-30dB)")
                    elif psnr_repair > 20:
                        st.info("⭐⭐⭐ 修复质量: 中等 (PSNR 20-25dB)")
                    else:
                        st.warning("⭐⭐ 修复质量: 需改进 (PSNR < 20dB)")
                
                # 验证信息
                if psnr_repair > 45:
                    st.warning(f"⚠️ PSNR异常偏高，最大像素差异: {max_diff_scratch:.6f}")
                    if max_diff_scratch < 0.01:
                        st.warning("⚠️ 修复图像与原始图像几乎相同，请检查划痕掩码是否正确应用！")
                
                if max_diff_scratch < 0.01:
                    st.warning("⚠️ 修复区域几乎没有变化，请检查修复算法或划痕掩码！")
            
            else:
                st.info("没有划痕区域，无需修复")
            
            st.markdown("---")
            st.success("✅ 划痕修复完成！")
            
            # 下载按钮
            result_display = ensure_color_display(st.session_state.repair_result_color)
            if result_display is not None:
                if result_display.max() <= 1.0:
                    result_display = (result_display * 255).astype(np.uint8)
                res_pil = Image.fromarray(result_display)
                buf = io.BytesIO()
                res_pil.save(buf, format="PNG")
                st.download_button(
                    "📥 下载修复结果 (彩色)",
                    buf.getvalue(),
                    "repaired_scratch.png",
                    "image/png",
                    use_container_width=True
                )
            
            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 重新绘制划痕", use_container_width=True):
                    st.session_state.mask = None
                    st.session_state.img_damaged = None
                    st.session_state.img_damaged_color = None
                    st.session_state.repair_result = None
                    st.session_state.repair_result_color = None
                    st.session_state.scratch_stage = 'damage'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择 (开始新修复)", use_container_width=True, type="primary"):
                    reset_all()
                    st.rerun()
    
    # ==================== 污渍修复流程 ====================
    elif st.session_state.repair_stage == 'stain_mode':
        
        # 步骤1: 上传图像
        if st.session_state.stain_stage == 'upload':
            st.markdown("### 🧪 污渍修复 - 步骤1/4: 上传图像")
            
            uploaded_file = st.file_uploader("选择图像文件", type=['png', 'jpg'], key="stain_upload")
            
            if uploaded_file is not None:
                img_gray, img_color = load_image_for_repair(uploaded_file)
                st.session_state.img_gray = img_gray
                st.session_state.img_original = img_color
                
                st.success("✅ 图像上传成功！")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➡️ 下一步：生成污渍", use_container_width=True, type="primary"):
                        st.session_state.stain_stage = 'damage'
                        st.rerun()
                with col2:
                    if st.button("🔙 返回模式选择"):
                        reset_all()
                        st.rerun()
            
            # 显示预览
            if st.session_state.img_original is not None:
                img_display = ensure_color_display(st.session_state.img_original)
                st.image(img_display, caption="已上传图像 (彩色)", use_container_width=True)
        
        # 步骤2: 生成污渍
        elif st.session_state.stain_stage == 'damage':
            st.markdown("### 🧪 污渍修复 - 步骤2/4: 生成污渍")
            
            # 显示彩色原始图像
            img_color_display = ensure_color_display(st.session_state.img_original)
            st.image(img_color_display, caption="原始图像 (彩色)", use_container_width=True)
            
            # 获取灰度图用于处理
            img_gray = st.session_state.img_gray
            if use_gpu and torch.is_tensor(img_gray):
                img_gray = img_gray.cpu().numpy()
            
            if st.button("🎲 生成随机污渍", use_container_width=True, type="primary"):
                try:
                    # 调用stain模块
                    mask, regions, area, percentage, damaged_gray, _ = stain.generate_random_stains(
                        img_gray, use_gpu, device, show_plots=False
                    )
                    
                    # 创建彩色损坏图像
                    if use_gpu and torch.is_tensor(st.session_state.img_original):
                        img_damaged_color = st.session_state.img_original.clone()
                        for c in range(img_damaged_color.shape[2]):
                            channel = img_damaged_color[:, :, c]
                            channel[mask] = 0
                            img_damaged_color[:, :, c] = channel
                    else:
                        img_damaged_color = st.session_state.img_original.copy()
                        for c in range(img_damaged_color.shape[2]):
                            img_damaged_color[:, :, c][mask] = 0
                    
                    st.session_state.mask = mask
                    st.session_state.img_damaged = damaged_gray
                    st.session_state.img_damaged_color = img_damaged_color
                    
                    st.success(f"✅ 生成 {len(regions)} 个污渍，占比 {percentage:.2f}%")
                    st.session_state.stain_stage = 'repair'
                    st.rerun()
                except Exception as e:
                    st.error(f"生成失败: {e}")
            
            # 返回按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔙 返回上一步", use_container_width=True):
                    st.session_state.stain_stage = 'upload'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择", use_container_width=True):
                    reset_all()
                    st.rerun()
        
        # 步骤3: 执行修复
        elif st.session_state.stain_stage == 'repair':
            st.markdown("### 🧪 污渍修复 - 步骤3/4: 执行修复")
            
            # 显示图像对比
            col1, col2 = st.columns(2)
            with col1:
                img_orig_display = ensure_color_display(st.session_state.img_original)
                st.image(img_orig_display, caption="原始图像 (彩色)", use_container_width=True)
            with col2:
                if st.session_state.img_damaged_color is not None:
                    img_damaged_display = ensure_color_display(st.session_state.img_damaged_color)
                else:
                    img_damaged_display = ensure_color_display(st.session_state.img_damaged)
                st.image(img_damaged_display, caption="带污渍图像", use_container_width=True)
            
            st.markdown("---")
            
            # 修复参数
            st.subheader("🔧 ")
            
            patch_size = 6
            dict_size = 64
            sparsity = 5
            max_iter = 25
            step = 1
            
            if st.button("🚀 开始污渍修复", use_container_width=True, type="primary"):
                try:
                    with st.spinner("正在修复污渍..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 准备数据
                        img_damaged_gray = st.session_state.img_damaged
                        mask_stains = st.session_state.mask
                        
                        # 获取mask的numpy数组
                        if use_gpu and torch.is_tensor(mask_stains):
                            mask_np = mask_stains.cpu().numpy()
                        else:
                            mask_np = mask_stains
                        
                        # 创建修复掩码（完好区域）
                        mask = ~mask_np
                        
                        # 步骤1: 提取完好区域图像块
                        status_text.text("1/8: 提取图像块...")
                        patches, patch_pos = stain.extract_patches_for_psnr20(
                            img_damaged_gray, mask, patch_size, use_gpu, device
                        )
                        progress_bar.progress(10)
                        
                        if patches.size == 0:
                            st.error("无法提取图像块")
                            st.stop()
                        
                        # 步骤2: 数据增强
                        status_text.text("2/8: 数据增强...")
                        patches_aug = stain.augment_patches_for_training(patches, use_gpu, device)
                        progress_bar.progress(20)
                        
                        # 步骤3: KSVD字典训练
                        status_text.text("3/8: 训练字典...")
                        if use_gpu and torch.is_tensor(patches_aug):
                            patches_np = patches_aug.cpu().numpy()
                        else:
                            patches_np = patches_aug
                        
                        dict_trained = stain.professional_ksvd_training_correct(
                            patches_np, dict_size, sparsity, max_iter, use_gpu, device
                        )
                        progress_bar.progress(35)
                        
                        # 步骤4: 掩膜预处理（膨胀处理）
                        status_text.text("4/8: 掩膜预处理...")
                        from skimage.morphology import disk, binary_dilation
                        
                        selem = disk(3)
                        mask_dilated = binary_dilation(mask_np, selem)
                        
                        if use_gpu:
                            mask_dilated = torch.from_numpy(mask_dilated).to(device)
                        progress_bar.progress(45)
                        
                        # 步骤5: 初始化填充
                        status_text.text("5/8: 初始化填充...")
                        img_current_gray = stain.professional_initialization(
                            img_damaged_gray, mask_dilated, use_gpu, device
                        )
                        
                        # 计算初始PSNR
                        init_psnr = stain.compute_local_psnr(img_current_gray, st.session_state.img_gray, mask_stains, use_gpu, device)
                        st.caption(f"初始化填充后PSNR: {init_psnr:.2f} dB")
                        
                        # 扩散填充策略（如果初始效果极差）
                        if init_psnr < 5:
                            st.caption("初始PSNR过低，执行扩散填充...")
                            img_current_gray = stain.diffusion_filling(img_current_gray, mask_dilated, use_gpu, device)
                        
                        progress_bar.progress(55)
                        
                        # 步骤6: 分区域深度修复
                        status_text.text("6/8: 分区域深度修复...")
                        
                        # 获取mask的CPU数组用于区域分析
                        if use_gpu and torch.is_tensor(mask_dilated):
                            mask_cpu = mask_dilated.cpu().numpy()
                        else:
                            mask_cpu = mask_dilated
                        
                        from skimage import measure
                        labeled_mask = measure.label(mask_cpu, connectivity=2)
                        regions = measure.regionprops(labeled_mask)
                        num_regions = len(regions)
                        
                        if num_regions > 0:
                            st.caption(f"检测到 {num_regions} 个污渍连通区域")
                            
                            # 按面积降序排列
                            region_sizes = [r.area for r in regions]
                            size_order = np.argsort(region_sizes)[::-1]
                            
                            # 处理前10个最大区域
                            for region_idx in range(min(10, num_regions)):
                                region_id = size_order[region_idx]
                                region = regions[region_id]
                                
                                # 扩大边界以捕捉周围纹理
                                min_row, min_col, max_row, max_col = region.bbox
                                padding = 20
                                min_row = max(0, min_row - padding)
                                max_row = min(mask_cpu.shape[0], max_row + padding)
                                min_col = max(0, min_col - padding)
                                max_col = min(mask_cpu.shape[1], max_col + padding)
                                
                                # 提取子区域
                                if use_gpu:
                                    if torch.is_tensor(img_current_gray):
                                        sub_img = img_current_gray[min_row:max_row, min_col:max_col].cpu().numpy()
                                    else:
                                        sub_img = img_current_gray[min_row:max_row, min_col:max_col]
                                    
                                    if torch.is_tensor(st.session_state.img_gray):
                                        sub_original = st.session_state.img_gray[min_row:max_row, min_col:max_col].cpu().numpy()
                                    else:
                                        sub_original = st.session_state.img_gray[min_row:max_row, min_col:max_col]
                                else:
                                    sub_img = img_current_gray[min_row:max_row, min_col:max_col]
                                    sub_original = st.session_state.img_gray[min_row:max_row, min_col:max_col]
                                
                                sub_mask = mask_dilated[min_row:max_row, min_col:max_col]
                                if use_gpu and torch.is_tensor(sub_mask):
                                    sub_mask = sub_mask.cpu().numpy()
                                
                                # 区域修复
                                sub_img_repaired = stain.repair_region_aggressively(
                                    sub_img, sub_mask, dict_trained, patch_size, use_gpu, device
                                )
                                
                                # 无缝融合处理
                                from scipy.ndimage import gaussian_filter
                                blend_mask = gaussian_filter(sub_mask.astype(np.float32), sigma=2)
                                
                                if use_gpu:
                                    sub_img_gpu = torch.from_numpy(sub_img).float().to(device)
                                    blend_mask_gpu = torch.from_numpy(blend_mask).float().to(device)
                                    sub_img_repaired = sub_img_gpu * (1 - blend_mask_gpu) + sub_img_repaired * blend_mask_gpu
                                    sub_img_repaired_np = sub_img_repaired.cpu().numpy()
                                else:
                                    sub_img_repaired = sub_img * (1 - blend_mask) + sub_img_repaired * blend_mask
                                    sub_img_repaired_np = sub_img_repaired
                                
                                # 放回原图
                                if use_gpu:
                                    img_current_gray[min_row:max_row, min_col:max_col] = torch.from_numpy(sub_img_repaired_np).float().to(device)
                                else:
                                    img_current_gray[min_row:max_row, min_col:max_col] = sub_img_repaired_np
                                
                                # 计算区域PSNR（每5个区域显示一次）
                                if (region_idx + 1) % 5 == 0:
                                    region_psnr = stain.compute_region_psnr(
                                        sub_img_repaired_np, sub_original, sub_mask, use_gpu, device
                                    )
                                    st.caption(f"  已完成 {region_idx + 1}/{num_regions} 个区域修复，当前PSNR: {region_psnr:.2f} dB")
                                
                                # 更新进度
                                progress_bar.progress(55 + int(25 * (region_idx + 1) / min(10, num_regions)))
                        
                        # 步骤7: 边缘精修
                        status_text.text("7/8: 边缘精修...")
                        
                        # 执行边缘精修
                        img_current_gray = stain.edge_refinement_expert(
                            img_current_gray, mask_dilated, dict_trained, patch_size, use_gpu, device
                        )
                        progress_bar.progress(85)
                        
                        # 步骤8: 全局一致性后处理
                        status_text.text("8/8: 后处理...")
                        
                        # 自适应平滑
                        if use_gpu:
                            img_current_cpu = img_current_gray.cpu().numpy()
                            mask_dilated_cpu = mask_dilated.cpu().numpy()
                        else:
                            img_current_cpu = img_current_gray
                            mask_dilated_cpu = mask_dilated
                        
                        img_current_cpu = stain.adaptive_smoothing(img_current_cpu, mask_dilated_cpu)
                        
                        # 智能后处理
                        img_current_cpu = stain.intelligent_postprocessing(
                            img_current_cpu, mask_dilated_cpu, use_gpu, device
                        )
                        
                        if use_gpu:
                            img_current_gray = torch.from_numpy(img_current_cpu).float().to(device)
                        else:
                            img_current_gray = img_current_cpu
                        
                        progress_bar.progress(95)
                        
                        # 步骤9: 最终结果
                        final_psnr = stain.compute_optimized_psnr(
                            img_current_gray, st.session_state.img_gray, mask_stains, use_gpu, device
                        )
                        st.caption(f"最终PSNR: {final_psnr:.2f} dB")
                        
                        img_repaired_gray = img_current_gray
                        
                        # 创建彩色修复结果
                        if use_gpu and torch.is_tensor(st.session_state.img_original):
                            img_repaired_color = st.session_state.img_original.clone()
                            if torch.is_tensor(img_repaired_gray):
                                repaired_gray_np = img_repaired_gray.cpu().numpy()
                            else:
                                repaired_gray_np = img_repaired_gray
                            
                            for c in range(3):
                                img_repaired_color[:, :, c][mask_stains] = torch.from_numpy(repaired_gray_np[mask_stains]).float().to(device)
                        else:
                            img_repaired_color = st.session_state.img_original.copy()
                            repaired_gray_np = img_repaired_gray if not use_gpu else img_repaired_gray.cpu().numpy()
                            for c in range(3):
                                img_repaired_color[:, :, c][mask_stains] = repaired_gray_np[mask_stains]
                        
                        progress_bar.progress(100)
                        status_text.text("修复完成！")
                        time.sleep(0.5)
                        
                        st.session_state.repair_result = img_repaired_gray
                        st.session_state.repair_result_color = img_repaired_color
                        st.session_state.stain_stage = 'result'
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"修复过程中出现错误: {str(e)}")
                    st.exception(e)
            
            # 返回按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔙 返回上一步", use_container_width=True):
                    st.session_state.stain_stage = 'damage'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择", use_container_width=True):
                    reset_all()
                    st.rerun()
        
        # 步骤4: 显示结果
        elif st.session_state.stain_stage == 'result':
            st.markdown("### 🧪 污渍修复 - 步骤4/4: 修复结果")
            
            # 准备数据用于显示
            if use_gpu:
                img_original_cpu = st.session_state.img_gray.cpu().numpy() if torch.is_tensor(st.session_state.img_gray) else st.session_state.img_gray
                img_repaired_gray = st.session_state.repair_result.cpu().numpy() if torch.is_tensor(st.session_state.repair_result) else st.session_state.repair_result
                mask_stains_cpu = st.session_state.mask.cpu().numpy() if torch.is_tensor(st.session_state.mask) else st.session_state.mask
            else:
                img_original_cpu = st.session_state.img_gray
                img_repaired_gray = st.session_state.repair_result
                mask_stains_cpu = st.session_state.mask
            
            # 处理掩码
            if len(mask_stains_cpu.shape) == 3:
                mask_stains_cpu = mask_stains_cpu[:, :, 0] > 0
            elif mask_stains_cpu.dtype != bool:
                mask_stains_cpu = mask_stains_cpu > 0
            
            # 计算污渍统计
            stain_area = np.sum(mask_stains_cpu)
            total_pixels = img_original_cpu.size
            stain_percentage = 100 * stain_area / total_pixels if total_pixels > 0 else 0
            
            # 显示三列彩色对比
            col1, col2, col3 = st.columns(3)
            with col1:
                img_orig_display = ensure_color_display(st.session_state.img_original)
                st.image(img_orig_display, caption="原始图像 (彩色)", use_container_width=True)
            with col2:
                img_damaged_display = ensure_color_display(st.session_state.img_damaged_color)
                st.image(img_damaged_display, caption=f"带污渍图像 ({stain_percentage:.2f}%)", use_container_width=True)
            with col3:
                img_result_display = ensure_color_display(st.session_state.repair_result_color)
                st.image(img_result_display, caption="修复结果 (彩色)", use_container_width=True)
            
            st.markdown("---")
            
            # ========== 计算质量指标 ==========
            st.subheader("📊 质量评估报告")
            
            if stain_area > 0:
                # 获取修复区域的像素值
                repaired_values = img_repaired_gray[mask_stains_cpu]
                original_values = img_original_cpu[mask_stains_cpu]
                
                # 验证：检查修复区域是否有变化
                max_diff_scratch = np.max(np.abs(repaired_values - original_values))
                
                # 计算修复区域差异
                diff_values = repaired_values - original_values
                abs_diff_values = np.abs(diff_values)
                
                # 修复区域统计
                mae_repair = np.mean(abs_diff_values)
                max_ae_repair = np.max(abs_diff_values)
                mse_repair = np.mean(diff_values ** 2)
                
                # 修复区域PSNR
                if mse_repair > 1e-10:
                    psnr_repair = 10 * np.log10(1.0 / mse_repair)
                else:
                    psnr_repair = float('inf')
                
                # 计算相对误差
                original_norm = np.linalg.norm(original_values)
                if original_norm > 1e-10:
                    relative_error = 100 * np.linalg.norm(diff_values) / original_norm
                else:
                    relative_error = 0
                
                # 计算SSIM指标
                try:
                    from skimage.metrics import structural_similarity as ssim_skimage
                    
                    h, w = img_repaired_gray.shape
                    
                    # 整体SSIM计算
                    if h >= 7 and w >= 7:
                        data_range = max(img_original_cpu.max(), img_repaired_gray.max()) - min(img_original_cpu.min(), img_repaired_gray.min())
                        if data_range <= 0:
                            data_range = 1.0
                        
                        win_size = min(7, h, w)
                        if win_size % 2 == 0:
                            win_size -= 1
                        if win_size < 3:
                            win_size = 3
                        
                        ssim_overall = ssim_skimage(
                            img_original_cpu, img_repaired_gray,
                            data_range=data_range,
                            win_size=win_size,
                            gaussian_weights=True,
                            sigma=1.5,
                            use_sample_covariance=False,
                            channel_axis=None
                        )
                    else:
                        ssim_overall = np.nan
                    
                    # 修复区域SSIM计算
                    if stain_area > 0:
                        rows, cols = np.where(mask_stains_cpu > 0)
                        if len(rows) > 0:
                            r_min = max(0, np.min(rows) - 10)
                            r_max = min(img_original_cpu.shape[0], np.max(rows) + 10)
                            c_min = max(0, np.min(cols) - 10)
                            c_max = min(img_original_cpu.shape[1], np.max(cols) + 10)
                            
                            patch_h = r_max - r_min
                            patch_w = c_max - c_min
                            
                            if patch_h >= 7 and patch_w >= 7:
                                patch_ref = img_original_cpu[r_min:r_max, c_min:c_max]
                                patch_comp = img_repaired_gray[r_min:r_max, c_min:c_max]
                                
                                patch_data_range = max(patch_ref.max(), patch_comp.max()) - min(patch_ref.min(), patch_comp.min())
                                if patch_data_range <= 0:
                                    patch_data_range = 1.0
                                
                                win_size = min(7, patch_h, patch_w)
                                if win_size % 2 == 0:
                                    win_size -= 1
                                if win_size < 3:
                                    win_size = 3
                                
                                ssim_patch = ssim_skimage(
                                    patch_ref, patch_comp,
                                    data_range=patch_data_range,
                                    win_size=win_size,
                                    gaussian_weights=True,
                                    sigma=1.5,
                                    use_sample_covariance=False,
                                    channel_axis=None
                                )
                            else:
                                ssim_patch = ssim_overall
                        else:
                            ssim_patch = ssim_overall
                    else:
                        ssim_patch = ssim_overall
                    
                    # 合理性修正
                    if not np.isnan(ssim_overall) and not np.isnan(ssim_patch):
                        if ssim_patch > ssim_overall:
                            ssim_overall, ssim_patch = ssim_patch, ssim_overall
                            
                except Exception as e:
                    print(f"SSIM计算出错: {e}")
                    ssim_overall = np.nan
                    ssim_patch = np.nan
                
                # 显示指标卡片
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("污渍像素", f"{stain_area}")
                    st.metric("污渍占比", f"{stain_percentage:.2f}%")
                
                with col2:
                    if not np.isinf(psnr_repair):
                        st.metric("修复区域PSNR", f"{psnr_repair:.2f} dB")
                    else:
                        st.metric("修复区域PSNR", "∞")
                    st.metric("修复区域SSIM", f"{ssim_patch:.4f}")
                
                with col3:
                    st.metric("平均绝对误差", f"{mae_repair:.4f}")
                    st.metric("最大绝对误差", f"{max_ae_repair:.4f}")
                
                with col4:
                    st.metric("相对误差", f"{relative_error:.2f}%")
                    st.metric("修复区域MSE", f"{mse_repair:.6f}")
                
                # 质量评级
                st.markdown("---")
                if not np.isinf(psnr_repair):
                    if psnr_repair > 30:
                        st.success("⭐⭐⭐⭐⭐ 修复质量: 优秀 (PSNR > 30dB)")
                    elif psnr_repair > 25:
                        st.success("⭐⭐⭐⭐ 修复质量: 良好 (PSNR 25-30dB)")
                    elif psnr_repair > 20:
                        st.info("⭐⭐⭐ 修复质量: 中等 (PSNR 20-25dB)")
                    else:
                        st.warning("⭐⭐ 修复质量: 需改进 (PSNR < 20dB)")
                
                # 验证信息
                if psnr_repair > 45:
                    st.warning(f"⚠️ PSNR异常偏高，最大像素差异: {max_diff_scratch:.6f}")
                    if max_diff_scratch < 0.01:
                        st.warning("⚠️ 修复图像与原始图像几乎相同，请检查污渍掩码是否正确应用！")
                
                if max_diff_scratch < 0.01:
                    st.warning("⚠️ 修复区域几乎没有变化，请检查修复算法或污渍掩码！")
            
            else:
                st.info("没有污渍区域，无需修复")
            
            st.markdown("---")
            st.success("✅ 污渍修复完成！")
            
            # 下载按钮
            result_display = ensure_color_display(st.session_state.repair_result_color)
            if result_display is not None:
                if result_display.max() <= 1.0:
                    result_display = (result_display * 255).astype(np.uint8)
                res_pil = Image.fromarray(result_display)
                buf = io.BytesIO()
                res_pil.save(buf, format="PNG")
                st.download_button(
                    "📥 下载修复结果 (彩色)",
                    buf.getvalue(),
                    "repaired_stain.png",
                    "image/png",
                    use_container_width=True
                )
            
            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 重新生成污渍", use_container_width=True):
                    st.session_state.mask = None
                    st.session_state.img_damaged = None
                    st.session_state.img_damaged_color = None
                    st.session_state.repair_result = None
                    st.session_state.repair_result_color = None
                    st.session_state.stain_stage = 'damage'
                    st.rerun()
            with col2:
                if st.button("🏠 返回模式选择 (开始新修复)", use_container_width=True, type="primary"):
                    reset_all()
                    st.rerun()
# ========== 2. SVD音频降噪模块（完全分开的流程：默认信号和上传文件）==========
elif app_mode == "🎵 SVD音频降噪":
    st.header("🎵 SVD音频降噪系统")
    st.caption("基于分帧SVD与自适应混合阈值的音频降噪系统")
    
    # ==================== 初始化session state ====================
    if 'audio_stage' not in st.session_state:
        st.session_state.audio_stage = 'mode_selection'  # mode_selection, test_signal_mode, upload_mode
    if 'test_signal_stage' not in st.session_state:
        st.session_state.test_signal_stage = 'params'  # params, process, result
    if 'upload_stage' not in st.session_state:
        st.session_state.upload_stage = 'upload'  # upload, params, process, result
    if 'audio_preset' not in st.session_state:
        st.session_state.audio_preset = "medium"  # light, medium, strong
    
    # ==================== 重置状态函数 ====================
    def reset_audio_all():
        st.session_state.audio_stage = 'mode_selection'
        st.session_state.test_signal_stage = 'params'
        st.session_state.upload_stage = 'upload'
        if 'y_clean' in st.session_state:
            del st.session_state.y_clean
        if 'fs' in st.session_state:
            del st.session_state.fs
        if 'y_noisy' in st.session_state:
            del st.session_state.y_noisy
        if 'result' in st.session_state:
            del st.session_state.result
    
    def reset_test_signal_state():
        st.session_state.test_signal_stage = 'params'
        if 'y_clean' in st.session_state:
            del st.session_state.y_clean
        if 'y_noisy' in st.session_state:
            del st.session_state.y_noisy
        if 'result' in st.session_state:
            del st.session_state.result
    
    def reset_upload_state():
        st.session_state.upload_stage = 'upload'
        if 'y_clean' in st.session_state:
            del st.session_state.y_clean
        if 'y_noisy' in st.session_state:
            del st.session_state.y_noisy
        if 'result' in st.session_state:
            del st.session_state.result
    
    # ==================== 模式选择界面 ====================
    if st.session_state.audio_stage == 'mode_selection':
        st.markdown("### 🔧 请选择音频来源")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎵 默认测试信号")
            st.markdown("使用系统生成的测试信号（440Hz+880Hz+1320Hz正弦波）")
            st.markdown("- 时长: 10秒")
            st.markdown("- 采样率: 8000 Hz")
            st.markdown("- 包含起振和衰减包络")
            if st.button("选择测试信号", use_container_width=True, type="primary"):
                st.session_state.audio_stage = 'test_signal_mode'
                st.session_state.test_signal_stage = 'params'
                st.rerun()
        
        with col2:
            st.markdown("#### 📁 上传WAV文件")
            st.markdown("上传自己的音频文件进行处理")
            st.markdown("- 支持格式: WAV")
            st.markdown("- 自动转换为单声道")
            st.markdown("- 自动归一化处理")
            if st.button("选择上传文件", use_container_width=True, type="primary"):
                st.session_state.audio_stage = 'upload_mode'
                st.session_state.upload_stage = 'upload'
                st.rerun()
    
    # ==================== 默认测试信号流程 ====================
    elif st.session_state.audio_stage == 'test_signal_mode':
        
        # 步骤1: 参数设置
        if st.session_state.test_signal_stage == 'params':
            st.markdown("### 🎵 测试信号 - 步骤1/3: 参数设置")
            
            # 降噪强度选择
            st.markdown("#### ⚡ 降噪强度选择")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🌱 轻度降噪", use_container_width=True):
                    st.session_state.audio_preset = "light"
                    st.rerun()
            with col2:
                if st.button("🌿 中度降噪", use_container_width=True, type="primary"):
                    st.session_state.audio_preset = "medium"
                    st.rerun()
            with col3:
                if st.button("🍂 强力降噪", use_container_width=True):
                    st.session_state.audio_preset = "strong"
                    st.rerun()
            
            # 根据预设设置参数
            if st.session_state.audio_preset == "light":
                noise_level = 0.05
                frame_length = 512
                overlap_ratio = 0.5
                energy_thresh = 0.95
                soft_thresh_factor = 0.4
                preset_desc = "轻度降噪：保留最多细节，适合轻微噪声"
                st.success(f"🌱 {preset_desc}")
                
            elif st.session_state.audio_preset == "medium":
                noise_level = 0.10
                frame_length = 256
                overlap_ratio = 0.6
                energy_thresh = 0.92
                soft_thresh_factor = 0.6
                preset_desc = "中度降噪：平衡效果和细节"
                st.info(f"🌿 {preset_desc}")
                
            else:  # strong
                noise_level = 0.15
                frame_length = 128
                overlap_ratio = 0.75
                energy_thresh = 0.88
                soft_thresh_factor = 0.8
                preset_desc = "强力降噪：降噪效果好，但可能损失细节"
                st.warning(f"🍂 {preset_desc}")
            
            # 显示参数详情
            with st.expander("🔧 详细参数", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**噪声强度**: {noise_level}")
                    st.markdown(f"**帧长**: {frame_length}")
                    st.markdown(f"**重叠率**: {overlap_ratio}")
                with col2:
                    st.markdown(f"**能量阈值**: {energy_thresh}")
                    st.markdown(f"**软阈值因子**: {soft_thresh_factor}")
                    st.markdown(f"**降噪模式**: 混合阈值 (固定)")
            
            # 候选k值（根据帧长自动设置）
            if frame_length == 128:
                fixed_k = [5, 10, 15, 20, 25]
            elif frame_length == 256:
                fixed_k = [10, 15, 20, 25, 30]
            else:  # 512
                fixed_k = [15, 20, 25, 30, 35]
            
            # 保存参数到session state
            st.session_state.noise_level = noise_level
            st.session_state.frame_length = frame_length
            st.session_state.overlap_ratio = overlap_ratio
            st.session_state.energy_thresh = energy_thresh
            st.session_state.soft_thresh_factor = soft_thresh_factor
            st.session_state.fixed_k = fixed_k
            st.session_state.svd_mode = "hybrid"
            st.session_state.soft_thresh_max = 0.18
            
            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 生成并降噪", use_container_width=True, type="primary"):
                    st.session_state.test_signal_stage = 'process'
                    st.rerun()
            with col2:
                if st.button("🔙 返回模式选择", use_container_width=True):
                    reset_audio_all()
                    st.rerun()
        
        # 步骤2: 处理过程
        elif st.session_state.test_signal_stage == 'process':
            st.markdown("### 🎵 测试信号 - 步骤2/3: 降噪处理")
            
            # 生成测试信号
            fs = 8000
            duration = 10
            y_clean, fs = generate_test_signal(fs=fs, duration=duration)
            t = np.arange(len(y_clean)) / fs
            
            # 添加噪声
            np.random.seed(42)
            noise = st.session_state.noise_level * np.random.randn(len(y_clean))
            y_noisy = y_clean + noise
            y_noisy = y_noisy - np.mean(y_noisy)
            
            # 计算原始SNR
            noise_energy = np.linalg.norm(y_noisy - y_clean) + 1e-12
            signal_energy = np.linalg.norm(y_clean)
            snr_original = 20 * np.log10(signal_energy / noise_energy)
            
            overlap = int(st.session_state.frame_length * st.session_state.overlap_ratio)
            
            # 执行降噪
            with st.spinner("正在进行SVD降噪处理..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("步骤1/3: 参数配置...")
                progress_bar.progress(20)
                
                params = {
                    'frame_length': st.session_state.frame_length,
                    'overlap': overlap,
                    'energy_thresh': st.session_state.energy_thresh,
                    'fixed_k': st.session_state.fixed_k,
                    'svd_mode': st.session_state.svd_mode,
                    'soft_thresh_factor': st.session_state.soft_thresh_factor,
                    'soft_thresh_max': st.session_state.soft_thresh_max,
                    'noise_tail_len': 20,
                    'fine_search_range': 10,
                }
                
                status_text.text("步骤2/3: SVD降噪处理中...")
                progress_bar.progress(50)
                
                try:
                    result = svd_denoise_audio(y_noisy, y_clean, fs, params, snr_original)
                except Exception as e:
                    st.error(f"降噪处理失败: {str(e)}")
                    st.exception(e)
                    st.stop()
                
                status_text.text("步骤3/3: 生成结果...")
                progress_bar.progress(80)
                
                # 保存结果到session state
                st.session_state.y_clean = y_clean
                st.session_state.y_noisy = y_noisy
                st.session_state.fs = fs
                st.session_state.t = t
                st.session_state.snr_original = snr_original
                st.session_state.result = result
                
                progress_bar.progress(100)
                status_text.success("✅ 处理完成!")
            
            st.session_state.test_signal_stage = 'result'
            st.rerun()
        
        # 步骤3: 显示结果
        elif st.session_state.test_signal_stage == 'result':
            st.markdown("### 🎵 测试信号 - 步骤3/3: 降噪结果")
            
            # 从session state获取数据
            y_clean = st.session_state.y_clean
            y_noisy = st.session_state.y_noisy
            fs = st.session_state.fs
            t = st.session_state.t
            snr_original = st.session_state.snr_original
            result = st.session_state.result
            
            improvement = result['best_snr'] - snr_original
            
            # 显示结果摘要
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最优k值", result['best_k'])
            with col2:
                st.metric("降噪后SNR", f"{result['best_snr']:.2f} dB", 
                         delta=f"{improvement:+.2f} dB")
            with col3:
                st.metric("能量保留率", f"{result['best_ener']:.1f}%")
            
            # 降噪效果评价
            if improvement > 8:
                st.success(f"✨ **降噪效果显著！** SNR提升了 {improvement:.1f} dB")
            elif improvement > 5:
                st.success(f"✅ **降噪效果良好**，SNR提升了 {improvement:.1f} dB")
            elif improvement > 3:
                st.info(f"📊 **降噪效果中等**，SNR提升了 {improvement:.1f} dB")
            else:
                st.warning(f"⚠️ 降噪效果一般，提升 {improvement:.1f} dB")
            
            # 创建标签页
            tab_audio1, tab_audio2, tab_audio3 = st.tabs(["🎧 波形对比", "📈 矩阵分析", "🧹 残差分析"])
            
            with tab_audio1:
                st.subheader("音频波形对比")
                col_w1, col_w2, col_w3 = st.columns(3)
                
                with col_w1:
                    fig1 = create_waveform_plot(t, y_clean, "原始音频", '#00AAFF')
                    st.pyplot(fig1)
                    plt.close(fig1)
                    st.audio(numpy_to_wav_bytes(y_clean, fs), format='audio/wav')
                
                with col_w2:
                    fig2 = create_waveform_plot(t, y_noisy, f"含噪音频 (SNR={snr_original:.1f}dB)", '#FF4B4B')
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.audio(numpy_to_wav_bytes(y_noisy, fs), format='audio/wav')
                
                with col_w3:
                    fig3 = create_waveform_plot(t, result['best_y'], 
                                                f"降噪音频 (k={result['best_k']}, SNR={result['best_snr']:.1f}dB)", 
                                                '#00CC00')
                    st.pyplot(fig3)
                    plt.close(fig3)
                    st.audio(numpy_to_wav_bytes(result['best_y'], fs), format='audio/wav')
                
                st.divider()
                
                # 频谱对比
                st.subheader("频谱分析")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    fig_s1 = create_spectrum_plot(y_clean, fs, "原始音频频谱", '#00AAFF')
                    st.pyplot(fig_s1)
                    plt.close(fig_s1)
                with col_s2:
                    fig_s2 = create_spectrum_plot(result['best_y'], fs, "降噪音频频谱", '#00CC00')
                    st.pyplot(fig_s2)
                    plt.close(fig_s2)
            
            with tab_audio2:
                st.subheader("矩阵数值分析")
                
                col_ana1, col_ana2, col_ana3 = st.columns(3)
                
                with col_ana1:
                    fig_sv = create_singular_value_plot(result['sing_vals'], result['best_k'], result['noise_thresh'])
                    st.pyplot(fig_sv)
                    plt.close(fig_sv)
                
                with col_ana2:
                    # SNR vs k曲线
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(result['k_candidates'], result['snr_values'], 'b-o', linewidth=2, markersize=5)
                    ax.axhline(y=snr_original, color='r', linestyle='--', linewidth=1.5, 
                              label=f'含噪SNR={snr_original:.1f}dB')
                    ax.set_xlabel('k值')
                    ax.set_ylabel('SNR (dB)')
                    ax.set_title('SNR vs k')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col_ana3:
                    # 累积能量曲线
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(np.arange(1, len(result['energy_cum'])+1), result['energy_cum']*100, 'b-', linewidth=2)
                    ax.plot(result['best_k'], result['energy_cum'][result['best_k']-1]*100, 'ro', markersize=6)
                    ax.set_xlabel('奇异值个数')
                    ax.set_ylabel('能量保留率 (%)')
                    ax.set_title('累积能量曲线')
                    ax.set_ylim([0, 105])
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.divider()
                
                # 奇异值统计
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                sing_vals = result['sing_vals']
                with col_stats1:
                    st.metric("最大奇异值", f"{sing_vals[0]:.3f}")
                with col_stats2:
                    st.metric("最小奇异值", f"{sing_vals[-1]:.3f}")
                with col_stats3:
                    st.metric("奇异值总和", f"{np.sum(sing_vals):.2f}")
                with col_stats4:
                    st.metric("条件数", f"{sing_vals[0]/sing_vals[-1]:.1f}")
            
            with tab_audio3:
                st.subheader("残差分析")
                fig_res = plot_residual_analysis(t, y_clean, result['best_y'], fs)
                st.pyplot(fig_res)
                plt.close(fig_res)
                
                # 残差统计
                residual = result['best_y'] - y_clean
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("残差RMS", f"{np.sqrt(np.mean(residual**2)):.4f}")
                with col_r2:
                    st.metric("最大残差", f"{np.max(np.abs(residual)):.4f}")
                with col_r3:
                    st.metric("残差标准差", f"{np.std(residual):.4f}")
            
            # 下载按钮
            st.divider()
            
            denoised_audio = result['best_y']
            if np.max(np.abs(denoised_audio)) > 1.0:
                denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 下载降噪音频",
                    numpy_to_wav_bytes(denoised_audio, fs).getvalue(),
                    f"test_signal_{st.session_state.audio_preset}_k{result['best_k']}.wav",
                    "audio/wav",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 重新选择参数", use_container_width=True):
                    reset_test_signal_state()
                    st.rerun()

# ==================== 上传WAV文件流程（音乐模式专用，只保留波形对比）====================
    elif st.session_state.audio_stage == 'upload_mode':
        
        # 步骤1: 上传文件
        if st.session_state.upload_stage == 'upload':
            st.markdown("### 📁 上传文件 - 步骤1/3: 上传音频文件")
            
            uploaded_file = st.file_uploader("选择WAV文件", type=["wav"], key="audio_upload_music")
            
            if uploaded_file is not None:
                try:
                    fs, y_clean = wavfile.read(uploaded_file)
                    y_clean = ensure_mono(y_clean, fs)
                    
                    # 归一化
                    if y_clean.dtype == 'int16':
                        y_clean = y_clean.astype(np.float32) / 32768.0
                    elif y_clean.dtype == 'int32':
                        y_clean = y_clean.astype(np.float32) / 2147483648.0
                    elif y_clean.dtype == 'uint8':
                        y_clean = (y_clean.astype(np.float32) - 128) / 128.0
                    elif y_clean.dtype == 'float32' or y_clean.dtype == 'float64':
                        max_val = np.max(np.abs(y_clean))
                        if max_val > 1.0:
                            y_clean = y_clean / max_val
                    
                    # 限制长度（处理前10秒，加快处理速度）
                    max_duration = 10
                    if len(y_clean) > max_duration * fs:
                        st.warning(f"音频过长，将只处理前{max_duration}秒")
                        y_clean = y_clean[:max_duration * fs]
                    
                    # 保存到session state
                    st.session_state.y_clean = y_clean
                    st.session_state.fs = fs
                    st.session_state.uploaded_filename = uploaded_file.name
                    
                    st.success(f"✅ 已加载音频: {uploaded_file.name}")
                    st.info(f"采样率: {fs} Hz, 时长: {len(y_clean)/fs:.2f} 秒")
                    
                    # 显示波形预览
                    t_preview = np.arange(min(10000, len(y_clean))) / fs
                    y_preview = y_clean[:len(t_preview)]
                    
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(t_preview, y_preview, color='#00AAFF', linewidth=0.8)
                    ax.set_xlabel('时间 (s)')
                    ax.set_ylabel('振幅')
                    ax.set_title('音频波形预览')
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # 操作按钮
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("➡️ 下一步：降噪", key="upload_to_params", use_container_width=True, type="primary"):
                            st.session_state.upload_stage = 'params'
                            st.rerun()
                    with col2:
                        if st.button("🔙 返回模式选择", key="upload_back_to_mode", use_container_width=True):
                            reset_audio_all()
                            st.rerun()
                
                except Exception as e:
                    st.error(f"读取音频文件失败: {str(e)}")
                    st.exception(e)
            else:
                if st.button("🔙 返回模式选择", key="upload_mode_back_no_file", use_container_width=True):
                    reset_audio_all()
                    st.rerun()
        
        # 步骤2: 参数设置（固定为音乐模式）
        elif st.session_state.upload_stage == 'params':
            st.markdown("### 📁 上传文件 - 步骤2/3: 降噪模式参数")
            
            st.info(f"当前音频: {st.session_state.uploaded_filename}")
            
            # ==================== 固定音乐模式参数 ====================
            st.markdown("#### 🎵 降噪模式参数")
            
            # 保留细节，温和降噪
            noise_level = 0.12      # 噪声强度系数
            frame_length = 1024      # 帧长
            overlap_ratio = 0.75     # 重叠率
            energy_thresh = 0.96     # 能量阈值
            soft_thresh_factor = 0.45 # 软阈值因子
            fixed_k = [20, 30, 40, 50, 60]  # 候选k值
            
            preset_desc = "🎵 保留音乐细节，温和去除背景噪声"
            st.success(f"🎵 {preset_desc}")
            
            # 计算预计加噪后SNR
            signal_power = np.mean(st.session_state.y_clean ** 2)
            noise_power = (noise_level ** 2)
            estimated_snr = 10 * np.log10(signal_power / noise_power)
            
            # 显示参数信息
            st.markdown("#### 📊 参数信息")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**噪声强度系数**: {noise_level}")
                st.info(f"**信号能量**: {signal_power:.4f}")
                st.info(f"**噪声能量**: {noise_power:.4f}")
                st.info(f"**帧长**: {frame_length}")
            with col2:
                st.success(f"**预计加噪后SNR**: {estimated_snr:.1f} dB")
                st.info(f"**重叠率**: {overlap_ratio}")
                st.info(f"**能量阈值**: {energy_thresh}")
                st.info(f"**软阈值因子**: {soft_thresh_factor}")
            
            with st.expander("🔧 详细参数", expanded=False):
                st.markdown(f"**候选k值**: {fixed_k}")
                st.markdown(f"**降噪模式**: 混合阈值")
                st.markdown(f"**软阈值上限**: 0.20")
            
            # 保存优化参数到session state
            st.session_state.noise_level = noise_level
            st.session_state.frame_length = frame_length
            st.session_state.overlap_ratio = overlap_ratio
            st.session_state.energy_thresh = energy_thresh
            st.session_state.soft_thresh_factor = soft_thresh_factor
            st.session_state.fixed_k = fixed_k
            st.session_state.svd_mode = "hybrid"
            st.session_state.soft_thresh_max = 0.20
            st.session_state.estimated_snr = estimated_snr
            st.session_state.audio_preset = "music"
            
            # 操作按钮
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚀 开始降噪", key="params_to_process", use_container_width=True, type="primary"):
                    st.session_state.upload_stage = 'process'
                    st.rerun()
            with col2:
                if st.button("🔙 返回上一步", key="params_back", use_container_width=True):
                    st.session_state.upload_stage = 'upload'
                    st.rerun()
            with col3:
                if st.button("🏠 返回模式选择", key="params_to_mode", use_container_width=True):
                    reset_audio_all()
                    st.rerun()
        
        # 步骤3: 处理过程
        elif st.session_state.upload_stage == 'process':
            st.markdown("### 📁 上传文件 - 步骤3/3: 降噪处理")
            
            # 从session state获取数据
            y_clean = st.session_state.y_clean
            fs = st.session_state.fs
            noise_level = st.session_state.noise_level
            
            t = np.arange(len(y_clean)) / fs
            
            # 添加噪声
            np.random.seed(42)
            noise = noise_level * np.random.randn(len(y_clean))
            y_noisy = y_clean + noise
            y_noisy = y_noisy - np.mean(y_noisy)
            
            # 计算原始SNR
            noise_energy = np.linalg.norm(y_noisy - y_clean) + 1e-12
            signal_energy = np.linalg.norm(y_clean)
            snr_original = 20 * np.log10(signal_energy / noise_energy)
            

            overlap = int(st.session_state.frame_length * st.session_state.overlap_ratio)
            
            # 执行降噪
            with st.spinner("正在进行降噪处理..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("步骤1/3: 加载参数...")
                progress_bar.progress(20)
                
                params = {
                    'frame_length': st.session_state.frame_length,
                    'overlap': overlap,
                    'energy_thresh': st.session_state.energy_thresh,
                    'fixed_k': st.session_state.fixed_k,
                    'svd_mode': st.session_state.svd_mode,
                    'soft_thresh_factor': st.session_state.soft_thresh_factor,
                    'soft_thresh_max': st.session_state.soft_thresh_max,
                    'noise_tail_len': 20,
                    'fine_search_range': 10,
                }
                
                status_text.text("步骤2/3: SVD降噪处理中...")
                progress_bar.progress(50)
                
                try:
                    result = svd_denoise_audio(y_noisy, y_clean, fs, params, snr_original)
                except Exception as e:
                    st.error(f"降噪处理失败: {str(e)}")
                    st.exception(e)
                    st.stop()
                
                status_text.text("步骤3/3: 生成结果...")
                progress_bar.progress(80)
                
                # 保存结果到session state
                st.session_state.y_noisy = y_noisy
                st.session_state.t = t
                st.session_state.snr_original = snr_original
                st.session_state.result = result
                
                progress_bar.progress(100)
                status_text.success("✅ 降噪完成!")
            
            st.session_state.upload_stage = 'result'
            st.rerun()
        
        # 步骤4: 显示结果（只保留波形对比）
        elif st.session_state.upload_stage == 'result':
            st.markdown("### 📁 上传文件 - 降噪结果")
            
            st.info(f"当前音频: {st.session_state.uploaded_filename}")
            
            # 从session state获取数据
            y_clean = st.session_state.y_clean
            y_noisy = st.session_state.y_noisy
            fs = st.session_state.fs
            t = st.session_state.t
            snr_original = st.session_state.snr_original
            result = st.session_state.result
            
            improvement = result['best_snr'] - snr_original
            
            # 只保留波形对比标签页
            st.subheader("🎧 音频波形对比")
            
            col_w1, col_w2, col_w3 = st.columns(3)
            
            with col_w1:
                st.write("**原始音频**")
                fig1 = create_waveform_plot(t, y_clean, "原始音频", '#00AAFF')
                st.pyplot(fig1)
                plt.close(fig1)
                st.audio(numpy_to_wav_bytes(y_clean, fs), format='audio/wav')
            
            with col_w2:
                st.write(f"**加噪音频** ")
                fig2 = create_waveform_plot(t, y_noisy, f"含噪音频", '#FF4B4B')
                st.pyplot(fig2)
                plt.close(fig2)
                st.audio(numpy_to_wav_bytes(y_noisy, fs), format='audio/wav')
            
            with col_w3:
                st.write(f"**降噪音频** ")
                fig3 = create_waveform_plot(t, result['best_y'], f"降噪音频", '#00CC00')
                st.pyplot(fig3)
                plt.close(fig3)
                st.audio(numpy_to_wav_bytes(result['best_y'], fs), format='audio/wav')
            
            # 下载按钮
            st.divider()
            
            denoised_audio = result['best_y']
            if np.max(np.abs(denoised_audio)) > 1.0:
                denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 下载降噪音频",
                    numpy_to_wav_bytes(denoised_audio, fs).getvalue(),
                    f"music_denoised_k{result['best_k']}_snr{result['best_snr']:.0f}.wav",
                    "audio/wav",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 重新处理", use_container_width=True):
                    st.session_state.upload_stage = 'upload'
                    st.rerun()
            with col3:
                if st.button("🏠 返回模式选择", use_container_width=True, type="primary"):
                    reset_audio_all()
                    st.rerun()    
# ========== 3. SVD图像去噪评估模块==========
else:  # app_mode == "📊 SVD图像去噪"
    st.header("📊 SVD图像去噪评估系统")
    st.caption("矩阵分析与图像处理：综合去噪评估")
    
    st.markdown("""
    <div class="feature-card">
        <h4>📚 算法说明</h4>
        <ul>
            <li><b>高斯噪声</b>：使用改进的SVD矩阵分解（低秩近似）</li>
            <li><b>椒盐噪声/像素丢失</b>：使用中值滤波（空间域处理）</li>
            <li><b>评估指标</b>：PSNR (峰值信噪比) + SSIM (结构相似性)</li>
            <li><i>注：椒盐噪声和像素丢失的处理放在此处仅为对比参考</i></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 文件上传
    uploaded_file = st.file_uploader("上传测试图像", type=["jpg", "png", "jpeg"])
    
    # 参数设置
    with st.expander("🔧 处理参数设置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            noise_kind = st.selectbox("噪声类型", ["高斯噪声", "椒盐噪声", "像素丢失"])
            noise_lvl = st.slider("噪声强度", 0, 150, 60)
        with col2:
            algo = st.radio("处理算法", ["软阈值收缩", "硬截断"], horizontal=True)
            k_param = st.slider("处理参数 (K值/滤波强度)", 1, 200, 30)
    
    # 核心处理函数
    def svd_denoise_core(channel, k_value, method, noise_kind):
        """针对不同噪声类型定制处理"""
        if noise_kind == "椒盐噪声":
            size = 3 if k_value < 100 else 5
            return median_filter(channel, size=size)
        
        A = channel.astype(np.float64)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        
        if method == "软阈值收缩":
            idx = min(k_value, len(s) - 1)
            tau = s[idx] * 0.7
            s_new = np.maximum(s - tau, 0) + s * 0.08
        else:
            s_new = s.copy()
            if k_value < len(s):
                s_new[k_value:] = s[k_value:] * 0.15
        
        denoised = (U * s_new) @ Vt
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def add_noise(img_np, n_type, n_val):
        """噪声生成"""
        if n_type == "高斯噪声":
            noise = np.random.normal(0, n_val, img_np.shape)
            noisy = img_np.astype(np.float64) + noise
        elif n_type == "椒盐噪声":
            noisy = img_np.copy()
            prob = n_val / 500.0
            mask = np.random.random(img_np.shape[:2])
            noisy[mask < prob/2] = 0
            noisy[mask > 1 - prob/2] = 255
        else:  # 像素丢失
            noisy = img_np.copy()
            prob = n_val / 300.0
            mask = np.random.random(img_np.shape[:2])
            noisy[mask < prob] = 0
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    if uploaded_file:
        # 加载图像
        pil_img = Image.open(uploaded_file).convert('RGB')
        pil_img.thumbnail((512, 512))
        raw_np = np.array(pil_img)
        
        st.image(raw_np, caption="原始图像", width=300)
        
        if st.button("🚀 开始处理", type="primary", use_container_width=True):
            with st.spinner("处理中..."):
                # 加噪
                noisy_np = add_noise(raw_np, noise_kind, noise_lvl)
                
                # 去噪
                denoised_channels = []
                for i in range(3):
                    channel = noisy_np[:, :, i]
                    if noise_kind == "像素丢失":
                        f_size = 3 if k_param < 100 else 5
                        ch_res = median_filter(channel, size=f_size)
                    else:
                        ch_res = svd_denoise_core(channel, k_param, algo, noise_kind)
                    denoised_channels.append(ch_res)
                
                final_denoised = np.stack(denoised_channels, axis=2)
                
                # 计算指标
                p_noisy = psnr_func(raw_np, noisy_np, data_range=255)
                s_noisy = ssim_func(raw_np, noisy_np, data_range=255, channel_axis=2)
                p_denoised = psnr_func(raw_np, final_denoised, data_range=255)
                s_denoised = ssim_func(raw_np, final_denoised, data_range=255, channel_axis=2)
                
                # 显示指标
                st.subheader("📊 质量评估指标")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("PSNR (加噪后)", f"{p_noisy:.2f} dB")
                with col_m2:
                    st.metric("SSIM (加噪后)", f"{s_noisy:.4f}")
                with col_m3:
                    st.metric("PSNR (处理后)", f"{p_denoised:.2f} dB", 
                             delta=f"{p_denoised - p_noisy:.2f}")
                with col_m4:
                    st.metric("SSIM (处理后)", f"{s_denoised:.4f}", 
                             delta=f"{s_denoised - s_noisy:.4f}")
                
                # 显示图像对比
                st.subheader("🖼️ 图像对比")
                col_img1, col_img2, col_img3 = st.columns(3)
                with col_img1:
                    st.image(raw_np, caption="原始图像", use_container_width=True)
                with col_img2:
                    st.image(noisy_np, caption="加噪图像", use_container_width=True)
                with col_img3:
                    st.image(final_denoised, caption="处理后图像", use_container_width=True)
                
                # 下载按钮
                buf = io.BytesIO()
                Image.fromarray(final_denoised).save(buf, format="PNG")
                st.download_button(
                    "📥 下载处理结果",
                    buf.getvalue(),
                    "denoised_image.png",
                    "image/png",
                    use_container_width=True
                )

# ==================== 底部信息 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><b>图像修复 · 音频降噪 · 图像去噪</p>
    <p>陈星 · 付小娜 · 高静怡 · 郭银银 · 刘梦园 · 马婉欣 · 田彩琴 · 王婧怡 </p>
    <p style="font-size: 0.9em;">© 西安电子科技大学 · 矩阵分析与应用</p>
</div>
""", unsafe_allow_html=True)

# ==================== 导入必要的模块 ====================
# 注意：需要确保 stain.py 和 scratch.py 在同一目录下
# 如果缺失，提供错误提示
try:
    import stain
    import scratch
except ImportError as e:
    st.error(f"缺少必要模块: {e}")
    st.info("请确保 stain.py 和 scratch.py 文件存在于当前目录")
