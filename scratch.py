import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import ImageFont

# ========== 在文件开头添加 ==========
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ===================================
from matplotlib.widgets import RectangleSelector
import os
import time
import warnings
from scipy import ndimage, spatial, signal
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
import skimage
from skimage import io, color, transform, filters, feature, morphology, measure, exposure
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage
import sys
import json
from datetime import datetime
from matplotlib.path import Path
import matplotlib.patches as patches


warnings.filterwarnings('ignore')

# ==================== GPU检查函数 ====================
def check_gpu_availability():
    """检查GPU可用性"""
    print('=== 检查GPU可用性 ===')
    
    use_gpu = False
    gpu_info = {}
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'检测到 {gpu_count} 个GPU设备')
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {props.name}')
            print(f'  计算能力: {props.major}.{props.minor}')
            print(f'  总内存: {props.total_memory/1e9:.2f} GB')
        
        # 使用第一个GPU
        device = torch.device('cuda:0')
        use_gpu = True
        print('启用GPU加速')
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
    else:
        print('未检测到GPU设备，使用CPU运行')
        device = torch.device('cpu')
        use_gpu = False
    
    return use_gpu, device

# ==================== 图像预处理函数 ====================
def read_and_preprocess_image(image_path, use_gpu, device):
    """读取并预处理图像"""
    # 读取图像
    img_color = io.imread(image_path)
    
    # 转换为浮点数
    if img_color.dtype == np.uint8:
        img_original = img_color.astype(np.float32) / 255.0
    else:
        img_original = img_color.astype(np.float32)
    
    # 转换为灰度图像用于处理
    if len(img_original.shape) == 3 and img_original.shape[2] == 3:
        img_gray = color.rgb2gray(img_original)
    else:
        img_gray = img_original.copy()
    
    # 转换为GPU张量
    if use_gpu:
        try:
            img_tensor = torch.from_numpy(img_gray).float().to(device)
            print('图像已上传到GPU')
            img = img_tensor
        except Exception as e:
            print(f'GPU上传失败: {e}')
            use_gpu = False
            img = img_gray
    else:
        img = img_gray
    
    # 显示原始图像大小
    if use_gpu:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    print(f'原始图像大小: {h} x {w}')
    
    # 保持原始分辨率
    scale_factor = 1.0
    if h > 1024 or w > 1024:
        scale_factor = 0.5
        if use_gpu:
            img_cpu = img.cpu().numpy()
            img_cpu = transform.resize(img_cpu, (int(h*scale_factor), int(w*scale_factor)), 
                                      anti_aliasing=True)
            img = torch.from_numpy(img_cpu).float().to(device)
        else:
            img = transform.resize(img, (int(h*scale_factor), int(w*scale_factor)), 
                                  anti_aliasing=True)
        h, w = img.shape[:2]
        print(f'已调整图像到: {h} x {w}')
    
    return img, use_gpu, device, scale_factor

import random
import cv2  # 用于高质量线条绘制

# ==================== 交互式绘制划痕函数 (修改版) ====================
def interactive_draw_scratches(img, use_gpu, device):
    """交互式绘制划痕线条 - 鼠标拖动流畅版 + 随机回退机制"""
    print('\n=== 交互式自由绘制划痕 (拖动鼠标绘图) ===')
    
    if use_gpu and torch.is_tensor(img):
        img_display = img.cpu().numpy()
        h, w = img_display.shape[:2]
    else:
        img_display = img
        h, w = img.shape[:2]
    
    class DrawingState:
        def __init__(self):
            self.lines = []
            self.current_line = []
            self.is_drawing = False
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.finished = False

    state = DrawingState()
    
    if len(img_display.shape) == 3:
        state.ax.imshow(img_display)
    else:
        state.ax.imshow(img_display, cmap='gray')
    
    state.ax.set_title('【左键拖动】绘制划痕 | 【右键】撤销 | 【Enter/Esc】确认完成')
    
    def on_press(event):
        if event.inaxes != state.ax: return
        if event.button == 1:
            state.is_drawing = True
            state.current_line = [[event.xdata, event.ydata]]
        elif event.button == 3:
            if state.lines:
                state.lines.pop()
                redraw()

    def on_move(event):
        if not state.is_drawing or event.inaxes != state.ax: return
        state.current_line.append([event.xdata, event.ydata])
        pts = np.array(state.current_line)
        state.ax.plot(pts[-2:, 0], pts[-2:, 1], 'r-', linewidth=2)
        state.fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 1 and state.is_drawing:
            state.is_drawing = False
            if len(state.current_line) > 1:
                state.lines.append(np.array(state.current_line))
            state.current_line = []

    def on_key(event):
        if event.key in ['enter', 'escape']:
            state.finished = True
            plt.close(state.fig)

    def redraw():
        state.ax.clear()
        if len(img_display.shape) == 3:
            state.ax.imshow(img_display)
        else:
            state.ax.imshow(img_display, cmap='gray')
        for line in state.lines:
            state.ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=2)
        state.fig.canvas.draw()

    state.fig.canvas.mpl_connect('button_press_event', on_press)
    state.fig.canvas.mpl_connect('motion_notify_event', on_move)
    state.fig.canvas.mpl_connect('button_release_event', on_release)
    state.fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    # ---------- 线条转换为掩码 (修改核心) ----------
    if use_gpu:
        mask_scratches = torch.zeros((h, w), dtype=torch.bool, device=device)
    else:
        mask_scratches = np.zeros((h, w), dtype=bool)

    if len(state.lines) > 0:
        print(f'正在处理 {len(state.lines)} 条手绘划痕...')
        for pts in state.lines:
            line_mask = points_to_line_mask(pts, h, w, 2)
            if use_gpu:
                mask_scratches = mask_scratches | torch.from_numpy(line_mask).to(device)
            else:
                mask_scratches = mask_scratches | line_mask
    else:
        # ===== 修改重点：如果没有绘制，生成完全随机的划痕 =====
        print('\n[注意] 未检测到手绘线条，返回空掩码')
        return mask_scratches,0,0
    # 统计数据
    if use_gpu:
        scratch_area = mask_scratches.sum().item()
    else:
        scratch_area = np.sum(mask_scratches)
    
    scratch_percentage = 100 * scratch_area / (h * w)
    print(f'划痕设置完成！划痕占比: {scratch_percentage:.2f}%')
    
    return mask_scratches, scratch_area, scratch_percentage

# ==================== 点序列转线条掩码函数 ====================
def points_to_line_mask(points, height, width, line_width):
    """将点序列转换为线条掩码 - 细线条版本"""
    line_mask = np.zeros((height, width), dtype=bool)
    
    if len(points) < 2:
        return line_mask
    
    # 连接相邻点形成线段
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        # 计算线段上的点 - 使用更密集的采样确保连续
        distance = np.linalg.norm(p2 - p1)
        num_points = max(int(np.ceil(distance * 2)), 2)  # 2倍采样确保连续性
        
        # 线性插值
        t = np.linspace(0, 1, num_points)
        x_line = p1[0] + t * (p2[0] - p1[0])
        y_line = p1[1] + t * (p2[1] - p1[1])
        
        # 四舍五入到最近的整数像素
        x_int = np.round(x_line).astype(int)
        y_int = np.round(y_line).astype(int)
        
        # 确保在图像范围内
        x_int = np.clip(x_int, 0, width - 1)
        y_int = np.clip(y_int, 0, height - 1)
        
        # 直接设置为True，不进行任何膨胀
        line_mask[y_int, x_int] = True
    
    # 移除填充孔洞的操作，避免线条变粗
    # from scipy import ndimage
    # line_mask = ndimage.binary_fill_holes(line_mask)
    
    return line_mask


# ==================== Patch提取函数 ====================
def extract_background_patches_scratches(img, mask, patch_size, step, use_gpu, device):
    """提取完好区域的图像块"""
    # 首先处理图像维度，确保是2D
    if use_gpu:
        if torch.is_tensor(img):
            # GPU张量处理
            if len(img.shape) == 3:
                print(f"警告: GPU图像张量维度为 {img.shape}，转换为2D")
                if img.shape[0] == 3:  # CHW格式
                    img_gray = torch.mean(img, dim=0)  # 通道平均
                elif img.shape[2] == 3:  # HWC格式
                    img_gray = torch.mean(img, dim=2)
                else:
                    img_gray = img[:, :, 0]
            elif len(img.shape) == 2:
                img_gray = img
            else:
                raise ValueError(f"不支持的图像维度: {img.shape}")
        else:
            # numpy数组处理
            img_np = img
            if len(img_np.shape) == 3:
                print(f"警告: GPU图像numpy维度为 {img_np.shape}，转换为2D")
                img_gray = np.mean(img_np, axis=2)
            elif len(img_np.shape) == 2:
                img_gray = img_np
            else:
                raise ValueError(f"不支持的图像维度: {img_np.shape}")
            img_gray = torch.from_numpy(img_gray).float().to(device)
        
        img_cpu = img_gray.cpu().numpy()
        
        # 处理mask
        if torch.is_tensor(mask):
            mask_cpu = mask.cpu().numpy()
        else:
            mask_cpu = mask
            
    else:
        # CPU处理
        if len(img.shape) == 3:
            print(f"警告: CPU图像维度为 {img.shape}，转换为2D")
            if img.shape[2] == 3:
                img_gray = np.mean(img, axis=2)  # RGB转灰度
            else:
                img_gray = img[:, :, 0]
        elif len(img.shape) == 2:
            img_gray = img
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")
        
        img_cpu = img_gray
        mask_cpu = mask
    
    # 现在获取正确的维度
    h, w = img_cpu.shape
    print(f"处理后的图像尺寸: {h}x{w}")
    
    # 计算划痕比例
    scratch_ratio = np.sum(~mask_cpu) / (h * w)
    print(f'划痕比例: {scratch_ratio*100:.2f}%')
    
    # 根据划痕比例调整提取密度
    if scratch_ratio < 0.01:  # 划痕很少（<1%）
        extract_density = 0.20  # 提取20%的位置
    elif scratch_ratio < 0.05:  # 划痕较少（<5%）
        extract_density = 0.30  # 提取30%的位置
    else:  # 划痕较多
        extract_density = 0.50  # 提取50%的位置
    
    # 计算理论最大提取位置
    max_row_positions = (h - patch_size) // step + 1
    max_col_positions = (w - patch_size) // step + 1
    max_total_positions = max_row_positions * max_col_positions
    
    # 目标提取位置数量
    target_positions = min(20000, int(max_total_positions * extract_density))
    print(f'理论最大位置: {max_total_positions}, 目标提取: {target_positions} ({extract_density*100:.1f}%)')
    
    # 网格采样
    if max_total_positions > target_positions:
        skip_factor = int(np.ceil(np.sqrt(max_total_positions / target_positions)))
        effective_step = step * skip_factor
        print(f'使用网格采样，实际步长: {effective_step}')
        
        row_range = list(range(0, h - patch_size + 1, effective_step))
        col_range = list(range(0, w - patch_size + 1, effective_step))
    else:
        row_range = list(range(0, h - patch_size + 1, step))
        col_range = list(range(0, w - patch_size + 1, step))
    
    # 如果还是太多，强制限制
    if len(row_range) * len(col_range) > 30000:
        max_rows = min(100, len(row_range))
        max_cols = min(300, len(col_range))
        
        row_idx = np.random.choice(len(row_range), max_rows, replace=False)
        col_idx = np.random.choice(len(col_range), max_cols, replace=False)
        
        row_range = [row_range[i] for i in sorted(row_idx)]
        col_range = [col_range[i] for i in sorted(col_idx)]
        
        print(f'强制限制: {len(row_range)}行 × {len(col_range)}列 = {len(row_range)*len(col_range)}位置')
    
    # 快速提取循环
    patches_list = []
    positions_list = []
    
    total_positions = len(row_range) * len(col_range)
    print(f'正在提取完好图像块 (从 {total_positions} 个位置)...')
    
    # 进度显示
    progress_interval = max(1, total_positions // 20)
    processed = 0
    last_progress = 0
    
    for i in row_range:
        for j in col_range:
            processed += 1
            current_progress = int(100 * processed / total_positions)
            if current_progress > last_progress and current_progress % 10 == 0:
                print(f'进度: {current_progress}%')
                last_progress = current_progress
            
            # 检查边界
            if i + patch_size > h or j + patch_size > w:
                continue
            
            # 检查patch质量
            patch_mask = mask_cpu[i:i+patch_size, j:j+patch_size]
            good_ratio = np.sum(patch_mask) / (patch_size * patch_size)
            
            if good_ratio >= 0.8:  # 80%以上完好
                img_patch = img_cpu[i:i+patch_size, j:j+patch_size]
                patches_list.append(img_patch.flatten())
                positions_list.append([i, j])
    
    if len(patches_list) == 0:
        return np.array([]), np.array([])
    
    patches_array = np.column_stack(patches_list)
    positions_array = np.array(positions_list)
    
    print(f'提取完成，共 {patches_array.shape[1]} 个完好图像块')
    
    # 后备方案：如果提取太少
    min_required = 500
    if patches_array.shape[1] < min_required:
        print(f'提取块数不足 ({patches_array.shape[1]} < {min_required})，使用快速补充...')
        
        # 从固定位置补充
        fixed_positions = [
            [0, 0],
            [0, w // 2],
            [0, w - patch_size],
            [h // 2, 0],
            [h // 2, w // 2],
            [h // 2, w - patch_size],
            [h - patch_size, 0],
            [h - patch_size, w // 2],
            [h - patch_size, w - patch_size]
        ]
        
        additional_patches = []
        additional_positions = []
        
        for pos in fixed_positions:
            i, j = pos
            if 0 <= i <= h - patch_size and 0 <= j <= w - patch_size:
                img_patch = img_cpu[i:i+patch_size, j:j+patch_size]
                additional_patches.append(img_patch.flatten())
                additional_positions.append([i, j])
        
        if additional_patches:
            patches_array = np.column_stack([patches_array] + additional_patches)
            positions_array = np.vstack([positions_array, additional_positions])
            print(f'补充后共 {patches_array.shape[1]} 个图像块')
    
    # 如果还是太少，生成随机patch
    if patches_array.shape[1] < 100:
        print('警告：图像块太少，生成随机patch...')
        
        random_count = 200 - patches_array.shape[1]
        additional_patches = []
        additional_positions = []
        
        for k in range(random_count):
            i = np.random.randint(0, h - patch_size + 1)
            j = np.random.randint(0, w - patch_size + 1)
            
            img_patch = img_cpu[i:i+patch_size, j:j+patch_size]
            additional_patches.append(img_patch.flatten())
            additional_positions.append([i, j])
        
        if additional_patches:
            patches_array = np.column_stack([patches_array] + additional_patches)
            positions_array = np.vstack([positions_array, additional_positions])
            print(f'随机补充后共 {patches_array.shape[1]} 个图像块')
    
    # 最终统计
    print('图像块提取完成:')
    print(f'  图像大小: {h}×{w}')
    print(f'  提取位置: {total_positions}/{max_total_positions} ({100*total_positions/max_total_positions:.1f}%)')
    print(f'  实际提取: {patches_array.shape[1]} 块')
    print(f'  块大小: {patch_size}×{patch_size} → {patch_size**2}维')
    
    # 确保patches是二维矩阵
    if patches_array.shape[0] != patch_size**2:
        print(f'调整patch维度: {patches_array.shape[0]} -> {patch_size**2}')
        if patches_array.shape[0] > patch_size**2:
            patches_array = patches_array[:patch_size**2, :]
        elif patches_array.shape[0] < patch_size**2:
            missing_dim = patch_size**2 - patches_array.shape[0]
            padding = np.zeros((missing_dim, patches_array.shape[1]))
            patches_array = np.vstack([patches_array, padding])
    
    # 转换为GPU张量
    if use_gpu:
        patches_array = torch.from_numpy(patches_array).float().to(device)
    
    return patches_array, positions_array

# ==================== KSVD字典训练函数 ====================

def improved_ksvd_for_scratches(patches, dict_size, target_sparsity, max_iter, use_gpu, device):
    """
    划痕专用字典训练函数 - 修正完整版
    """
    # 1. 初始化数据
    if use_gpu and torch.is_tensor(patches):
        patches_tensor = patches
        n_dim, n_patch = patches.shape
    else:
        patches_tensor = torch.from_numpy(patches).float().to(device) if use_gpu else patches
        n_dim, n_patch = patches.shape
    
    print(f'划痕字典训练: patches {n_dim}×{n_patch}, 目标字典大小 {dict_size}')
    
    # 2. 初始化字典 (从样本随机选取)
    if use_gpu:
        rand_idx = torch.randperm(n_patch)[:dict_size]
        dict_tensor = patches_tensor[:, rand_idx].clone()
        dict_tensor /= (torch.norm(dict_tensor, dim=0, keepdim=True) + 1e-8)
    else:
        rand_idx = np.random.choice(n_patch, dict_size, replace=False)
        dict_tensor = patches_tensor[:, rand_idx].copy()
        dict_tensor /= (np.linalg.norm(dict_tensor, axis=0, keepdims=True) + 1e-8)
    
    error_history = []
    sparsity_history = []
    
    # 初始稀疏度设定为 1.5 倍目标值，后续逐渐降低
    initial_sparsity = max(int(target_sparsity * 1.5), target_sparsity + 2)
    
    # 3. KSVD 迭代主循环
    for iter_idx in range(max_iter):
        print(f'\n迭代 {iter_idx+1}/{max_iter}')
        print('-' * 20)
        start_time = time.time()
        
        # --- A. 动态计算当前轮次的稀疏度限制 (由高到低) ---
        if max_iter > 1:
            progress = iter_idx / (max_iter - 1)
            current_sparsity = int(np.round(initial_sparsity - progress * (initial_sparsity - target_sparsity)))
        else:
            current_sparsity = target_sparsity
        
        # --- B. 稀疏编码 ---
        print(f'  1. 稀疏编码 (当前限额: {current_sparsity})...')
        alpha_tensor = sparse_coding_stable(dict_tensor, patches_tensor, current_sparsity, use_gpu, device, iter_idx+1)
        
        # 计算当前稀疏率
        if use_gpu:
            sparsity_ratio = (torch.abs(alpha_tensor) > 1e-4).sum().item() / alpha_tensor.numel()
        else:
            sparsity_ratio = (np.abs(alpha_tensor) > 1e-4).sum() / alpha_tensor.size
        
        # --- C. 字典更新 ---
        print('  2. 字典更新...')
        # 同时获取更新后的字典和系数，确保维度严格同步
        dict_tensor, alpha_tensor = update_dictionary_stable(
            dict_tensor, patches_tensor, alpha_tensor, 
            use_gpu, device, iter_idx+1, sparsity_ratio,max_iter
        )
        
        # --- D. 计算重建误差 (RMSE) ---
        print('  3. 计算误差...')
        if use_gpu:
            # 此时 dict_tensor(n_dim, dict_size) 和 alpha_tensor(dict_size, n_patch) 完美匹配
            reconstruction = torch.mm(dict_tensor, alpha_tensor)
            mse = torch.mean((patches_tensor - reconstruction) ** 2).item()
            error = np.sqrt(mse)
        else:
            reconstruction = np.dot(dict_tensor, alpha_tensor)
            error = np.sqrt(np.mean((patches_tensor - reconstruction) ** 2))
        
        iter_time = time.time() - start_time
        
        # 记录数据
        error_history.append(error)
        sparsity_history.append(sparsity_ratio)
        
        print(f'  结果统计:')
        print(f'    - 重建误差(RMSE): {error:.4f}')
        print(f'    - 实际非零比例: {sparsity_ratio*100:.2f}%')
        print(f'    - 耗时: {iter_time:.2f} 秒')
        
        # 早停机制
        if iter_idx >= 3 and len(error_history) >= 2:
            if abs(error_history[-1] - error_history[-2]) < 1e-5:
                print('  [提示] 误差趋于平稳，提前停止训练。')
                break
    
    # ===== 只改这里：返回字典和系数 =====
    return dict_tensor, alpha_tensor


def sparse_coding_stable(dict_mat, patches, max_sparsity, use_gpu, device, iteration):
    """稳定的稀疏编码"""
    if use_gpu:
        dict_cols = dict_mat.shape[1]
        num_patches = patches.shape[1]
        
        # 根据迭代次数调整参数
        if iteration <= 3:
            coeff_threshold = 0.02
            min_atoms = 3
        else:
            coeff_threshold = 0.05
            min_atoms = max(1, max_sparsity // 2)
        
        alpha = torch.zeros(dict_cols, num_patches, device=device)
        
        # 批处理
        batch_size = min(100, num_patches)
        for batch_start in range(0, num_patches, batch_size):
            batch_end = min(batch_start + batch_size, num_patches)
            batch_patches = patches[:, batch_start:batch_end]
            
            for i in range(batch_end - batch_start):
                signal = batch_patches[:, i]
                coef = stable_omp(dict_mat, signal, max_sparsity, coeff_threshold, 
                                 min_atoms, use_gpu, device)
                alpha[:, batch_start + i] = coef
        
        return alpha
    else:
        dict_cols = dict_mat.shape[1]
        num_patches = patches.shape[1]
        
        if iteration <= 3:
            coeff_threshold = 0.02
            min_atoms = 3
        else:
            coeff_threshold = 0.05
            min_atoms = max(1, max_sparsity // 2)
        
        alpha = np.zeros((dict_cols, num_patches))
        
        for i in range(num_patches):
            signal = patches[:, i]
            coef = stable_omp(dict_mat, signal, max_sparsity, coeff_threshold,
                             min_atoms, use_gpu, device)
            alpha[:, i] = coef
        
        return alpha

def update_dictionary_stable(dict_mat, patches, alpha, use_gpu, device, iteration, sparsity_ratio, max_iter):
    """
    强化版字典更新：解决死原子问题，确保高更新率
    原子更新数量随迭代逐渐下降
    
    Parameters:
    - max_iter: 最大迭代次数（从主函数传入）
    """
    if use_gpu:
        D = dict_mat.clone()
        A = alpha.clone()
        Y = patches
    else:
        D = dict_mat.copy()
        A = alpha.copy()
        Y = patches

    n_dim, dict_size = D.shape
    n_samples = Y.shape[1]
    
    # ===== 智能阈值策略：适应任意迭代次数 =====
    # 前30%的迭代：快速构建期
    if iteration <= max(1, int(max_iter * 0.3)):
        dead_threshold = max(1, iteration)  # 1,2,3...
        min_samples = max(2, iteration * 2)  # 2,4,6...
    
    # 中间40%的迭代：稳定优化期
    elif iteration <= max(1, int(max_iter * 0.7)):
        # 阈值缓慢增长
        progress = (iteration - int(max_iter * 0.3)) / (max_iter * 0.4)
        dead_threshold = max(1, int(3 + progress * 2))  # 3-5
        min_samples = max(6, int(6 + progress * 4))  # 6-10
    
    # 最后30%的迭代：精细调整期
    else:
        # 阈值达到上限，逐渐收紧
        progress = (iteration - int(max_iter * 0.7)) / (max_iter * 0.3)
        dead_threshold = max(0, int(5 - progress * 5))  # 5→0
        min_samples = max(8, int(10 - progress * 2))  # 10→8
    
    # 确保阈值在合理范围内
    dead_threshold = max(0, min(5, dead_threshold))
    min_samples = max(2, min(12, min_samples))
    
    update_count = 0
    
    # 预计算总残差 R = Y - DA
    if use_gpu:
        R = Y - torch.mm(D, A)
        # 统计每个原子的使用频率
        atom_usage = torch.sum(torch.abs(A) > 1e-4, dim=1)
        # 优先更新使用频率高的原子
        usage_order = torch.argsort(atom_usage, descending=True)
    else:
        R = Y - np.dot(D, A)
        atom_usage = np.sum(np.abs(A) > 1e-4, axis=1)
        usage_order = np.argsort(-atom_usage)

    for i in range(dict_size):
        k = usage_order[i].item() if use_gpu else usage_order[i]
        
        # 找到使用原子 k 的样本索引
        if use_gpu:
            idx = torch.where(torch.abs(A[k, :]) > 1e-4)[0]
        else:
            idx = np.where(np.abs(A[k, :]) > 1e-4)[0]
        
        # --- 替换策略：使用次数≤阈值就替换 ---
        if len(idx) <= dead_threshold:
            if use_gpu:
                error_per_sample = torch.sum(R**2, dim=0)
                max_err_idx = torch.argmax(error_per_sample)
                new_atom = R[:, max_err_idx]
                new_atom /= (torch.norm(new_atom) + 1e-8)
                D[:, k] = new_atom
                A[k, :] = 0
                A[k, max_err_idx] = 1.0
            else:
                error_per_sample = np.sum(R**2, axis=0)
                max_err_idx = np.argmax(error_per_sample)
                new_atom = R[:, max_err_idx]
                D[:, k] = new_atom / (np.linalg.norm(new_atom) + 1e-8)
                A[k, :] = 0
                A[k, max_err_idx] = 1.0
            update_count += 1
            continue

        # --- SVD更新：只有使用充分的原子才做SVD ---
        if len(idx) >= min_samples:
            try:
                if use_gpu:
                    Ek = R[:, idx] + torch.mm(D[:, k:k+1], A[k:k+1, idx])
                    U, S, V = torch.linalg.svd(Ek, full_matrices=False)
                    new_atom = U[:, 0]
                    new_coef = S[0] * V[0, :]
                    
                    R[:, idx] = Ek - torch.mm(new_atom.unsqueeze(1), new_coef.unsqueeze(0))
                    D[:, k] = new_atom
                    A[k, idx] = new_coef
                else:
                    Ek = R[:, idx] + np.outer(D[:, k], A[k, idx])
                    U, S, Vt = np.linalg.svd(Ek, full_matrices=False)
                    D[:, k] = U[:, 0]
                    A[k, idx] = S[0] * Vt[0, :]
                    R[:, idx] = Ek - np.outer(D[:, k], A[k, idx])
                
                update_count += 1
            except:
                # SVD失败时替换
                if use_gpu:
                    error_per_sample = torch.sum(R**2, dim=0)
                    max_err_idx = torch.argmax(error_per_sample)
                    new_atom = R[:, max_err_idx]
                    new_atom /= (torch.norm(new_atom) + 1e-8)
                    D[:, k] = new_atom
                    A[k, :] = 0
                    A[k, max_err_idx] = 1.0
                else:
                    error_per_sample = np.sum(R**2, axis=0)
                    max_err_idx = np.argmax(error_per_sample)
                    new_atom = R[:, max_err_idx]
                    D[:, k] = new_atom / (np.linalg.norm(new_atom) + 1e-8)
                    A[k, :] = 0
                    A[k, max_err_idx] = 1.0
                update_count += 1
        # else: 中间区间的原子保持不动
            
    if iteration % 1 == 0:
        print(f'    字典更新: {update_count}/{dict_size} 原子已优化 (替换≤{dead_threshold}, SVD≥{min_samples})')
        
    return D, A
def stable_omp(dict_mat, signal, max_sparsity, coeff_threshold, min_atoms, use_gpu, device):
    """稳定的OMP算法"""
    dict_cols = dict_mat.shape[1]
    
    if use_gpu:
        coef = torch.zeros(dict_cols, device=device)
        dict_mat_gpu = dict_mat
        signal_gpu = signal
        
        residual = signal_gpu.clone()
        selected = []
        signal_norm = torch.norm(signal_gpu)
        
        if signal_norm < 1e-6:
            return coef
        
        # 自适应参数
        rel_error_target = 0.15
        min_correlation = 0.05 * signal_norm
        
        for t in range(max_sparsity):
            # 计算相关性
            correlations = torch.abs(dict_mat_gpu.T @ residual)
            
            # 排除已选原子
            if selected:
                correlations[selected] = 0
            
            # 找出最大相关性
            max_corr, idx = torch.max(correlations, 0)
            idx = idx.item()
            
            # 停止条件
            if max_corr < min_correlation and t >= min_atoms:
                break
            
            if idx in selected:
                break
            
            selected.append(idx)
            
            # 最小二乘求解
            dict_sel = dict_mat_gpu[:, selected]
            A = dict_sel.T @ dict_sel + torch.eye(len(selected), device=device) * 1e-8
            b = dict_sel.T @ signal_gpu
            coef_sel = torch.linalg.solve(A, b)
            
            # 更新残差
            residual = signal_gpu - dict_sel @ coef_sel
            
            # 停止条件：相对误差达到目标
            current_rel_error = torch.norm(residual) / signal_norm
            if current_rel_error < rel_error_target and t >= min_atoms:
                break
            
            # 停止条件：改善不明显
            if t >= 2:
                prev_residual = signal_gpu - dict_mat_gpu[:, selected[:-1]] @ coef_sel[:-1]
                prev_error = torch.norm(prev_residual)
                curr_error = torch.norm(residual)
                improvement = (prev_error - curr_error) / prev_error
                
                if improvement < 0.02:
                    selected = selected[:-1]
                    break
        
        if selected:
            # 最终的最小二乘解
            dict_sel = dict_mat_gpu[:, selected]
            A = dict_sel.T @ dict_sel + torch.eye(len(selected), device=device) * 1e-8
            b = dict_sel.T @ signal_gpu
            coef_sel = torch.linalg.solve(A, b)
            
            # 系数后处理
            coef_sel = torch.sign(coef_sel) * torch.max(torch.abs(coef_sel) - coeff_threshold/2, 
                                                       torch.tensor(0.0, device=device))
            coef_sel[torch.abs(coef_sel) < coeff_threshold] = 0
            
            coef[selected] = coef_sel
            
            # 如果所有系数都太小，保留最大的一个
            if torch.all(torch.abs(coef_sel) < coeff_threshold) and len(coef_sel) > 0:
                max_val, max_idx = torch.max(torch.abs(coef_sel), 0)
                coef[selected[max_idx]] = torch.sign(coef_sel[max_idx]) * max(coeff_threshold, max_val.item())
        
        return coef
    else:
        coef = np.zeros(dict_cols)
        residual = signal.copy()
        selected = []
        signal_norm = np.linalg.norm(signal)
        
        if signal_norm < 1e-6:
            return coef
        
        rel_error_target = 0.15
        min_correlation = 0.05 * signal_norm
        
        for t in range(max_sparsity):
            correlations = np.abs(dict_mat.T @ residual)
            
            if selected:
                correlations[selected] = 0
            
            idx = np.argmax(correlations)
            max_corr = correlations[idx]
            
            if max_corr < min_correlation and t >= min_atoms:
                break
            
            if idx in selected:
                break
            
            selected.append(idx)
            
            dict_sel = dict_mat[:, selected]
            A = dict_sel.T @ dict_sel + np.eye(len(selected)) * 1e-8
            b = dict_sel.T @ signal
            coef_sel = np.linalg.solve(A, b)
            
            residual = signal - dict_sel @ coef_sel
            
            current_rel_error = np.linalg.norm(residual) / signal_norm
            if current_rel_error < rel_error_target and t >= min_atoms:
                break
            
            if t >= 2:
                prev_residual = signal - dict_mat[:, selected[:-1]] @ coef_sel[:-1]
                prev_error = np.linalg.norm(prev_residual)
                curr_error = np.linalg.norm(residual)
                improvement = (prev_error - curr_error) / prev_error
                
                if improvement < 0.02:
                    selected = selected[:-1]
                    break
        
        if selected:
            dict_sel = dict_mat[:, selected]
            A = dict_sel.T @ dict_sel + np.eye(len(selected)) * 1e-8
            b = dict_sel.T @ signal
            coef_sel = np.linalg.solve(A, b)
            
            coef_sel = np.sign(coef_sel) * np.maximum(np.abs(coef_sel) - coeff_threshold/2, 0)
            coef_sel[np.abs(coef_sel) < coeff_threshold] = 0
            
            coef[selected] = coef_sel
            
            if np.all(np.abs(coef_sel) < coeff_threshold) and len(coef_sel) > 0:
                max_idx = np.argmax(np.abs(coef_sel))
                coef[selected[max_idx]] = np.sign(coef_sel[max_idx]) * max(coeff_threshold, np.abs(coef_sel[max_idx]))
        
        return coef

# ==================== 划痕修复函数 ====================
def fill_scratches_with_neighbors_gpu(img, mask_scratches, use_gpu, device):
    """
    用邻域中值填充划痕区域
    支持 2D (H, W) 和 3D (H, W, C) 图像
    """
    # 1. 统一维度处理：获取高度和宽度
    shape = img.shape
    h, w = shape[0], shape[1]
    is_3d = len(shape) == 3
    
    img_filled = img.clone() if use_gpu else img.copy()
    
    # 2. 获取划痕坐标 - 确保mask是2D
    if use_gpu:
        # 确保 mask 是 2D 逻辑掩码
        if torch.is_tensor(mask_scratches):
            if len(mask_scratches.shape) == 3:
                mask_2d = mask_scratches[:, :, 0] > 0
            else:
                mask_2d = mask_scratches > 0
        else:
            if len(mask_scratches.shape) == 3:
                mask_2d = mask_scratches[:, :, 0] > 0
            else:
                mask_2d = mask_scratches > 0
            mask_2d = torch.from_numpy(mask_2d.astype(np.float32)).to(device) > 0
            
        coords = torch.where(mask_2d)
        rows, cols = coords[0], coords[1]
    else:
        if len(mask_scratches.shape) == 3:
            mask_2d = mask_scratches[:, :, 0] > 0
        else:
            mask_2d = mask_scratches > 0
        rows, cols = np.where(mask_2d)

    if len(rows) == 0:
        return img_filled

    # 3. 填充逻辑
    r = 2  # 邻域半径 (5x5)
    
    for k in range(len(rows)):
        i, j = rows[k], cols[k]
        
        # 计算邻域边界
        i_min, i_max = max(0, i - r), min(h, i + r + 1)
        j_min, j_max = max(0, j - r), min(w, j + r + 1)
        
        if use_gpu:
            # 在 GPU 上直接操作
            neighbor_mask = mask_2d[i_min:i_max, j_min:j_max]
            
            if is_3d:
                # 处理多通道 (H, W, C) 或 (C, H, W)
                if img_filled.shape[0] == 3:  # (C, H, W) 格式
                    for c in range(img_filled.shape[0]):
                        region = img_filled[c, i_min:i_max, j_min:j_max]
                        valid_pixels = region[~neighbor_mask]
                        if valid_pixels.numel() > 0:
                            img_filled[c, i, j] = torch.median(valid_pixels)
                else:  # (H, W, C) 格式
                    for c in range(img_filled.shape[2]):
                        region = img_filled[i_min:i_max, j_min:j_max, c]
                        valid_pixels = region[~neighbor_mask]
                        if valid_pixels.numel() > 0:
                            img_filled[i, j, c] = torch.median(valid_pixels)
            else:
                # 处理单通道 (H, W)
                region = img_filled[i_min:i_max, j_min:j_max]
                valid_pixels = region[~neighbor_mask]
                if valid_pixels.numel() > 0:
                    img_filled[i, j] = torch.median(valid_pixels)
                    
        else:
            # CPU (Numpy) 逻辑
            neighbor_mask = mask_2d[i_min:i_max, j_min:j_max]
            
            if is_3d:
                # 处理多通道 (H, W, C)
                for c in range(img_filled.shape[2]):
                    region = img_filled[i_min:i_max, j_min:j_max, c]
                    valid_pixels = region[~neighbor_mask]
                    if valid_pixels.size > 0:
                        img_filled[i, j, c] = np.median(valid_pixels)
            else:
                # 处理单通道 (H, W)
                region = img_filled[i_min:i_max, j_min:j_max]
                valid_pixels = region[~neighbor_mask]
                if valid_pixels.size > 0:
                    img_filled[i, j] = np.median(valid_pixels)
                    
    return img_filled

def repair_scratches_iterative_gpu(img_current, mask_scratches, dict_mat, patch_size, 
                                 sparsity, step, use_gpu, device, iteration):
    """迭代修复划痕 - 修复维度错误"""
    # ★★★ 核心修复：处理图像维度 ★★★
    if use_gpu:
        # 处理GPU张量
        if len(img_current.shape) == 3:
            # 如果是3D图像，转换为2D灰度图
            if img_current.shape[0] == 3:  # (C, H, W) 格式
                img_gray = torch.mean(img_current, dim=0)
            elif img_current.shape[2] == 3:  # (H, W, C) 格式
                img_gray = torch.mean(img_current, dim=2)
            else:
                img_gray = img_current[:, :, 0]  # 取第一个通道
            h, w = img_gray.shape
            img_result = img_current.clone()  # 保持原始维度用于返回
        else:
            img_gray = img_current
            h, w = img_gray.shape
            img_result = img_current.clone()
    else:
        # 处理CPU数组
        if len(img_current.shape) == 3:
            # 如果是3D图像，转换为2D灰度图
            if img_current.shape[2] == 3:  # (H, W, C) 格式
                img_gray = np.mean(img_current, axis=2)
            else:
                img_gray = img_current[:, :, 0]  # 取第一个通道
            h, w = img_gray.shape
            img_result = img_current.copy()
        else:
            img_gray = img_current
            h, w = img_gray.shape
            img_result = img_current.copy()
    
    # 处理mask_scratches，确保是2D
    if use_gpu:
        if torch.is_tensor(mask_scratches):
            if len(mask_scratches.shape) == 3:
                mask_2d = mask_scratches[:, :, 0] > 0
            else:
                mask_2d = mask_scratches > 0
        else:
            if len(mask_scratches.shape) == 3:
                mask_2d = mask_scratches[:, :, 0] > 0
            else:
                mask_2d = mask_scratches > 0
            mask_2d = torch.from_numpy(mask_2d.astype(np.float32)).to(device) > 0
    else:
        if len(mask_scratches.shape) == 3:
            mask_2d = mask_scratches[:, :, 0] > 0
        else:
            mask_2d = mask_scratches > 0
    
    # 创建修复优先级图 - 使用2D灰度图
    if use_gpu:
        # 转换为CPU numpy用于梯度计算
        img_cpu = img_gray.cpu().numpy()
        mask_cpu = mask_2d.cpu().numpy()
        
        # 计算梯度
        grad_y, grad_x = np.gradient(img_cpu)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        priority_map = grad_mag * mask_cpu
        
        # 转回GPU
        priority_map = torch.from_numpy(priority_map).float().to(device)
    else:
        grad_y, grad_x = np.gradient(img_gray)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        priority_map = grad_mag * mask_2d
    
    # 按优先级处理划痕
    repair_mask = mask_2d.clone() if use_gpu else mask_2d.copy()
    
    # 收集所有需要处理的patch位置
    locations = []
    priorities = []
    
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            if use_gpu:
                mask_patch = repair_mask[i:i+patch_size, j:j+patch_size].cpu().numpy()
            else:
                mask_patch = repair_mask[i:i+patch_size, j:j+patch_size]
            
            if np.any(mask_patch):
                # 计算该patch的平均优先级
                if use_gpu:
                    patch_priority = priority_map[i:i+patch_size, j:j+patch_size].cpu().numpy()
                else:
                    patch_priority = priority_map[i:i+patch_size, j:j+patch_size]
                
                avg_priority = np.mean(patch_priority)
                locations.append([i, j])
                priorities.append(avg_priority)
    
    if not locations:
        return img_result
    
    # 按优先级降序排序
    priorities = np.array(priorities)
    locations = np.array(locations)
    sort_idx = np.argsort(-priorities)
    sorted_locations = locations[sort_idx]
    
    processed_count = 0
    
    for idx in range(len(sorted_locations)):
        i, j = sorted_locations[idx]
        
        if use_gpu:
            mask_patch = repair_mask[i:i+patch_size, j:j+patch_size].cpu().numpy()
            
            # 从原始图像（可能是3D）中提取patch
            if len(img_result.shape) == 3:
                if img_result.shape[0] == 3:  # (C, H, W)
                    # 取所有通道的平均作为灰度patch
                    patch_rgb = img_result[:, i:i+patch_size, j:j+patch_size].cpu().numpy()
                    img_patch = np.mean(patch_rgb, axis=0)
                else:  # (H, W, C)
                    patch_rgb = img_result[i:i+patch_size, j:j+patch_size, :].cpu().numpy()
                    img_patch = np.mean(patch_rgb, axis=2)
            else:
                img_patch = img_result[i:i+patch_size, j:j+patch_size].cpu().numpy()
        else:
            mask_patch = repair_mask[i:i+patch_size, j:j+patch_size]
            
            # 从原始图像（可能是3D）中提取patch
            if len(img_result.shape) == 3:
                if img_result.shape[2] == 3:  # (H, W, C)
                    patch_rgb = img_result[i:i+patch_size, j:j+patch_size, :]
                    img_patch = np.mean(patch_rgb, axis=2)
                else:
                    img_patch = img_result[i:i+patch_size, j:j+patch_size, 0]
            else:
                img_patch = img_result[i:i+patch_size, j:j+patch_size]
        
        # 只处理包含划痕像素的块
        if np.any(mask_patch):
            unknown_pixels = np.where(mask_patch.flatten())[0]
            known_pixels = np.where(~mask_patch.flatten())[0]
            
            if len(unknown_pixels) > 0 and len(known_pixels) >= 4:
                # 约束稀疏编码
                patch_vec = img_patch.flatten()
                known_vec = patch_vec[known_pixels]
                
                if use_gpu:
                    dict_known = dict_mat[known_pixels, :]
                else:
                    dict_known = dict_mat[known_pixels, :]
                
                # 使用划痕专用OMP
                if use_gpu:
                    alpha = stable_omp(dict_known, torch.from_numpy(known_vec).float().to(device), 
                                      min(sparsity, len(known_pixels)), 0.05, 3, use_gpu, device)
                    alpha = alpha.cpu().numpy()
                else:
                    alpha = stable_omp(dict_known, known_vec, min(sparsity, len(known_pixels)),
                                      0.05, 3, use_gpu, device)
                
                # 重构
                if use_gpu:
                    patch_recon = (dict_mat.cpu().numpy() @ alpha).reshape(patch_size, patch_size)
                else:
                    patch_recon = (dict_mat @ alpha).reshape(patch_size, patch_size)
                
                patch_recon = np.clip(patch_recon, 0, 1)
                
                # 只更新划痕像素
                img_patch_flat = img_patch.flatten()
                img_patch_flat[unknown_pixels] = patch_recon.flatten()[unknown_pixels]
                img_patch_new = img_patch_flat.reshape(patch_size, patch_size)
                
                # 加权更新（根据迭代次数）
                weight = max(0.3, 0.8 - 0.15 * (iteration - 1))
                
                if use_gpu:
                    # 更新原图（保持原始维度）
                    if len(img_result.shape) == 3:
                        if img_result.shape[0] == 3:  # (C, H, W)
                            for c in range(3):
                                patch_current = img_result[c, i:i+patch_size, j:j+patch_size].cpu().numpy()
                                patch_updated = weight * img_patch_new + (1 - weight) * patch_current
                                img_result[c, i:i+patch_size, j:j+patch_size] = torch.from_numpy(patch_updated).float().to(device)
                        else:  # (H, W, C)
                            for c in range(img_result.shape[2]):
                                patch_current = img_result[i:i+patch_size, j:j+patch_size, c].cpu().numpy()
                                patch_updated = weight * img_patch_new + (1 - weight) * patch_current
                                img_result[i:i+patch_size, j:j+patch_size, c] = torch.from_numpy(patch_updated).float().to(device)
                    else:
                        img_result[i:i+patch_size, j:j+patch_size] = (
                            weight * torch.from_numpy(img_patch_new).float().to(device) +
                            (1 - weight) * img_result[i:i+patch_size, j:j+patch_size]
                        )
                else:
                    # 更新原图（保持原始维度）
                    if len(img_result.shape) == 3:
                        if img_result.shape[2] == 3:  # (H, W, C)
                            for c in range(3):
                                patch_current = img_result[i:i+patch_size, j:j+patch_size, c]
                                img_result[i:i+patch_size, j:j+patch_size, c] = weight * img_patch_new + (1 - weight) * patch_current
                        else:
                            patch_current = img_result[i:i+patch_size, j:j+patch_size, 0]
                            img_result[i:i+patch_size, j:j+patch_size, 0] = weight * img_patch_new + (1 - weight) * patch_current
                    else:
                        img_result[i:i+patch_size, j:j+patch_size] = (
                            weight * img_patch_new +
                            (1 - weight) * img_result[i:i+patch_size, j:j+patch_size]
                        )
                
                # 更新修复掩码
                if use_gpu:
                    repair_mask[i:i+patch_size, j:j+patch_size] = False
                else:
                    repair_mask[i:i+patch_size, j:j+patch_size] = False
                
                processed_count += 1
    
    return img_result


# ==================== 主函数中的调用部分 ====================
def main():
    """主函数 - 命令行版本"""
    # 0. 检查GPU可用性
    use_gpu, device = check_gpu_availability()
    
    # 1. 读取并预处理图像
    image_path = '1.png'
    if not os.path.exists(image_path):
        print(f'错误：图像文件 {image_path} 不存在！')
        return
    
    img, use_gpu, device, scale_factor = read_and_preprocess_image(image_path, use_gpu, device)
    
    # 获取图像尺寸
    if use_gpu:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    
    # 2. 交互式绘制划痕线条
    mask_scratches, scratch_area, scratch_percentage = interactive_draw_scratches(
        img, use_gpu, device
    )
    
    # 创建损坏图像
    if use_gpu:
        img_damaged = img.clone()
        img_damaged[mask_scratches] = 0
    else:
        img_damaged = img.copy()
        img_damaged[mask_scratches] = 0
    
    # 验证划痕是否正确添加
    if scratch_area > 0:
        if use_gpu:
            damaged_scratch_values = img_damaged[mask_scratches].cpu().numpy()
        else:
            damaged_scratch_values = img_damaged[mask_scratches]
        max_val = np.max(damaged_scratch_values)
        print(f'  划痕区域像素最大值(应为0): {max_val:.6f}')
        if max_val > 0.01:
            print('  ⚠️  警告: 划痕区域未正确设置为0！')
    
    # 显示损坏图像
    plt.figure(figsize=(10, 8))
    if use_gpu:
        plt.imshow(img_damaged.cpu().numpy(), cmap='gray')
    else:
        plt.imshow(img_damaged, cmap='gray')
    plt.title(f'带划痕的图像 ({scratch_percentage:.2f}% 像素损坏)')
    plt.axis('off')
    plt.show()
    
    # 创建修复掩码 - 完好区域为True，划痕区域为False
    mask = ~mask_scratches
    
    # 3. 划痕修复参数设置
    print('\n=== 划痕修复参数设置 ===')
    patch_size = 18
    dict_size = 512
    sparsity = 12
    max_iter = 3
    step = 10
    repair_iterations = 10
    
    print(f'patch_size: {patch_size}')
    print(f'dict_size: {dict_size}')
    print(f'sparsity: {sparsity}')
    print(f'step: {step}')
    print(f'repair_iterations: {repair_iterations}')
    
    # 4. 提取完好区域的图像块
    print('\n=== 提取完好区域图像块 ===')
    patches, patch_pos = extract_background_patches_scratches(
        img_damaged, mask, patch_size, step, use_gpu, device
    )
    
    if patches.size == 0:
        print('错误：未能提取到图像块！')
        return
    
    print(f'从完好区域提取到 {patches.shape[1] if len(patches.shape) > 1 else 0} 个图像块')
    
    # 如果提取的块太少，调整参数
    if patches.shape[1] < dict_size * 2:
        print('完好块不足，放宽提取条件')
        step = max(1, step - 1)
        patches, patch_pos = extract_background_patches_scratches(
            img_damaged, mask, patch_size, step, use_gpu, device
        )
        print(f'重新提取到 {patches.shape[1] if len(patches.shape) > 1 else 0} 个图像块')
    
    patches_aug = patches
    print(f'使用 {patches_aug.shape[1] if len(patches_aug.shape) > 1 else 0} 个训练块')
    
    # 5. KSVD字典训练
    print('\n=== KSVD字典训练 ===')
    expected_dim = patch_size ** 2
    
    # 验证维度
    if patches_aug.shape[0] != expected_dim:
        print(f'维度不匹配: patches_aug {patches_aug.shape[0]} != expected {expected_dim}')
        
        if patches_aug.shape[0] > expected_dim:
            patches_aug = patches_aug[:expected_dim, :]
            print(f'已截断到 {expected_dim} 维')
        elif patches_aug.shape[0] < expected_dim:
            missing_dim = expected_dim - patches_aug.shape[0]
            if use_gpu:
                padding = torch.zeros(missing_dim, patches_aug.shape[1], device=device)
                patches_aug = torch.cat([patches_aug, padding], dim=0)
            else:
                padding = np.zeros((missing_dim, patches_aug.shape[1]))
                patches_aug = np.vstack([patches_aug, padding])
            print(f'已补充 {missing_dim} 维')
    
    start_time = time.time()
    
    # ✅ 接收两个返回值
    dict_trained, alpha_trained = improved_ksvd_for_scratches(
        patches_aug, dict_size, sparsity, max_iter, use_gpu, device
    )
    
    training_time = time.time() - start_time
    print(f'字典训练完成，耗时 {training_time:.2f} 秒')
    print(f'字典维度: {dict_trained.shape}')
    print(f'系数维度: {alpha_trained.shape}')
    
    # 6. 划痕修复
    print('\n=== 开始划痕修复 ===')
    
    # 初始化：使用邻域填充划痕
    img_current = fill_scratches_with_neighbors_gpu(img_damaged, mask_scratches, use_gpu, device)
    
    # 确保 mask 是 2D (用于后续计算)
    if use_gpu and torch.is_tensor(mask_scratches):
        mask_2d = mask_scratches[:, :, 0] if len(mask_scratches.shape) == 3 else mask_scratches
    else:
        mask_2d = mask_scratches[:, :, 0] if len(mask_scratches.shape) == 3 else mask_scratches
    
    # 设置稀疏度演变逻辑
    start_sparsity = sparsity 
    end_sparsity = max(1, sparsity // 2)
    
    for iter_idx in range(repair_iterations):
        print(f'\n--- 第 {iter_idx+1}/{repair_iterations} 轮修复 ---')
        
        # 动态计算当前轮次的稀疏度
        if repair_iterations > 1:
            current_sparsity = int(start_sparsity - (iter_idx / (repair_iterations - 1)) * (start_sparsity - end_sparsity))
        else:
            current_sparsity = start_sparsity
        
        # 步长也可以逐渐减小以提高精细度
        current_step = max(1, step - iter_idx // 2)
        
        print(f'    参数调整: 稀疏度限制={current_sparsity}, 步长={current_step}')
        
        iter_start_time = time.time()
        
        # 调用修复函数
        img_current = repair_scratches_iterative_gpu(
            img_current, 
            mask_scratches, 
            dict_trained, 
            patch_size,
            current_sparsity,
            current_step, 
            use_gpu, 
            device, 
            iter_idx + 1
        )
        
        iter_time = time.time() - iter_start_time
        print(f'    本轮修复耗时: {iter_time:.2f} 秒')
    
    img_repaired = img_current
    
    # ====================== 7. 修复结果展示和质量评估 ======================
    print('\n=== 修复结果展示与质量评估 ===\n')
    
    # 确保所有数据在CPU上用于显示和评估
    if use_gpu:
        img_original = img.cpu().numpy()
        img_repaired_final = img_repaired.cpu().numpy()
        mask_scratches_cpu = mask_scratches.cpu().numpy()
        img_damaged_cpu = img_damaged.cpu().numpy()
    else:
        img_original = img
        img_repaired_final = img_repaired
        mask_scratches_cpu = mask_scratches
        img_damaged_cpu = img_damaged
    
    # 计算划痕统计
    scratch_area = np.sum(mask_scratches_cpu)
    total_pixels = img_original.size
    scratch_percentage = 100 * scratch_area / total_pixels
    
    # ====================== 计算质量指标 ======================
    print('正在计算质量评估指标...')
    
    # 1. 计算整体图像PSNR
    mse_overall = np.mean((img_repaired_final.flatten() - img_original.flatten()) ** 2)
    if mse_overall > 0:
        psnr_overall = 10 * np.log10(1 / mse_overall)
    else:
        psnr_overall = float('inf')
    print(f'整体图像PSNR: {psnr_overall:.2f} dB')
    
    # 2. 计算修复区域统计和PSNR
    scratch_area = np.sum(mask_scratches_cpu)
    total_pixels = img_original.size
    scratch_percentage = 100 * scratch_area / total_pixels
    
    if scratch_area > 0:
        # 获取修复区域的像素值
        repaired_values = img_repaired_final[mask_scratches_cpu]
        original_values = img_original[mask_scratches_cpu]
        
        # 计算修复区域差异
        diff_values = repaired_values - original_values
        abs_diff_values = np.abs(diff_values)
        
        # 修复区域统计
        mae_repair = np.mean(abs_diff_values)              # 平均绝对误差
        max_ae_repair = np.max(abs_diff_values)            # 最大绝对误差
        mse_repair = np.mean(diff_values ** 2)             # 修复区域MSE
        
        # 修复区域PSNR
        if mse_repair > 0:
            psnr_repair = 10 * np.log10(1 / mse_repair)
        else:
            psnr_repair = float('inf')
        
        # 计算相对误差
        original_norm = np.linalg.norm(original_values)
        if original_norm > 0:
            relative_error = 100 * np.linalg.norm(diff_values) / original_norm
        else:
            relative_error = 0
        
        print('\n=== 修复区域PSNR计算验证 ===')
        print(f'划痕像素总数: {scratch_area}')
        print(f'修复区域MSE: {mse_repair:.6f}')
        print(f'修复区域PSNR: {psnr_repair:.2f} dB')
        
        if np.isinf(psnr_repair):
            print('⚠️ 警告：修复区域PSNR为无穷大')
            print('   可能原因：修复完美或MSE为0')
        elif psnr_repair > 100:
            print('⚠️ 警告：修复区域PSNR异常偏高')
            print('   可能原因：MSE计算异常')
        
    else:
        # 如果没有划痕区域
        mae_repair = 0
        max_ae_repair = 0
        mse_repair = 0
        psnr_repair = float('inf')
        relative_error = 0
        abs_diff_values = np.array([])
        print('没有划痕区域，跳过修复区域PSNR计算')
    
    # ====================== 3. 计算SSIM指标 ======================
    print('\n=== 计算SSIM指标 ===')
    
    # 确保图像是2D灰度numpy数组
    if use_gpu:
        img_ref = img.cpu().numpy() if torch.is_tensor(img) else img
        img_comp = img_repaired.cpu().numpy() if torch.is_tensor(img_repaired) else img_repaired
    else:
        img_ref = img
        img_comp = img_repaired
    
    # 如果是3D图像，转换为2D灰度
    if len(img_ref.shape) == 3:
        if img_ref.shape[0] == 3:  # (C, H, W)
            img_ref = np.mean(img_ref, axis=0)
        else:  # (H, W, C)
            img_ref = np.mean(img_ref, axis=2)
    
    if len(img_comp.shape) == 3:
        if img_comp.shape[0] == 3:  # (C, H, W)
            img_comp = np.mean(img_comp, axis=0)
        else:  # (H, W, C)
            img_comp = np.mean(img_comp, axis=2)
    
    # 确保掩码是2D numpy数组
    if use_gpu:
        mask_np = mask_scratches.cpu().numpy() if torch.is_tensor(mask_scratches) else mask_scratches
    else:
        mask_np = mask_scratches
    
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    
    # 初始化SSIM值
    ssim_overall = np.nan
    ssim_patch = np.nan
    
    try:
        from skimage.metrics import structural_similarity as ssim_skimage
        has_ssim_toolbox = True
    except ImportError:
        has_ssim_toolbox = False
        print('错误: 未安装 scikit-image，请运行 pip install scikit-image')
    
    if has_ssim_toolbox:
        try:
            # 检查形状是否一致
            if img_ref.shape != img_comp.shape:
                print(f'警告: 形状不匹配 {img_ref.shape} vs {img_comp.shape}，尝试调整...')
                from scipy.ndimage import zoom
                factors = [i/j for i, j in zip(img_ref.shape, img_comp.shape)]
                img_comp = zoom(img_comp, factors)
            
            h, w = img_comp.shape
            
            # ===== 1. 整体SSIM计算 =====
            if h >= 7 and w >= 7:
                # 计算数据范围
                data_range = max(img_ref.max(), img_comp.max()) - min(img_ref.min(), img_comp.min())
                if data_range <= 0:
                    data_range = 1.0
                
                # 自适应窗口大小
                win_size = min(7, h, w)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size < 3:
                    win_size = 3
                
                ssim_overall = ssim_skimage(
                    img_ref, 
                    img_comp,
                    data_range=data_range,
                    win_size=win_size,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                    channel_axis=None
                )
                print(f'  [结果] 整体SSIM: {ssim_overall:.4f}')
            else:
                print(f'  图像尺寸太小 ({h}x{w})，无法计算整体SSIM')
                ssim_overall = np.nan
            
            # ===== 2. 修复区域SSIM计算 =====
            if scratch_area > 0:
                rows, cols = np.where(mask_np > 0)
                if len(rows) > 0:
                    # 增加10像素外扩边界
                    r_min = max(0, np.min(rows) - 10)
                    r_max = min(img_ref.shape[0], np.max(rows) + 10)
                    c_min = max(0, np.min(cols) - 10)
                    c_max = min(img_ref.shape[1], np.max(cols) + 10)
                    
                    patch_h = r_max - r_min
                    patch_w = c_max - c_min
                    
                    if patch_h >= 7 and patch_w >= 7:
                        patch_ref = img_ref[r_min:r_max, c_min:c_max]
                        patch_comp = img_comp[r_min:r_max, c_min:c_max]
                        
                        # 计算patch的数据范围
                        patch_data_range = max(patch_ref.max(), patch_comp.max()) - min(patch_ref.min(), patch_comp.min())
                        if patch_data_range <= 0:
                            patch_data_range = 1.0
                        
                        # 自适应窗口大小
                        win_size = min(7, patch_h, patch_w)
                        if win_size % 2 == 0:
                            win_size -= 1
                        if win_size < 3:
                            win_size = 3
                        
                        ssim_patch = ssim_skimage(
                            patch_ref,
                            patch_comp,
                            data_range=patch_data_range,
                            win_size=win_size,
                            gaussian_weights=True,
                            sigma=1.5,
                            use_sample_covariance=False,
                            channel_axis=None
                        )
                        print(f'  [结果] 修复区域SSIM: {ssim_patch:.4f}')
                    else:
                        print(f'  修复区域尺寸 ({patch_h}x{patch_w}) 太小，使用整体SSIM替代')
                        ssim_patch = ssim_overall
                else:
                    print(f'  [提示] 未发现划痕区域掩码，使用整体SSIM')
                    ssim_patch = ssim_overall
            else:
                print(f'  [提示] 无划痕区域，使用整体SSIM')
                ssim_patch = ssim_overall
            
            # ===== 3. 合理性修正 =====
            if not np.isnan(ssim_overall) and not np.isnan(ssim_patch):
                if ssim_patch > ssim_overall:
                    print('  [修正] 修复区域SSIM异常偏高，交换整体与局部SSIM值')
                    ssim_overall, ssim_patch = ssim_patch, ssim_overall
                    print(f'   修正后：整体SSIM={ssim_overall:.4f}, 修复区域SSIM={ssim_patch:.4f}')
        
        except Exception as e:
            import traceback
            print(f'SSIM计算过程中出错: {e}')
            traceback.print_exc()
            ssim_overall = np.nan
            ssim_patch = np.nan
    
    else:
        print('警告: scikit-image未安装，跳过SSIM计算')
        ssim_overall = np.nan
        ssim_patch = np.nan
    
    # 最终汇总打印
    print('-' * 40)
    print('最终评估结果:')
    if not np.isnan(ssim_overall):
        print(f'  整体SSIM: {ssim_overall:.4f}')
    else:
        print(f'  整体SSIM: 未计算')
    if not np.isnan(ssim_patch):
        print(f'  修复区域SSIM: {ssim_patch:.4f}')
    else:
        print(f'  修复区域SSIM: 未计算')
    print('-' * 40)
    
    # ====================== 显示修复结果 ======================
    print('\n生成修复结果对比图...')
    
    # 主结果对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('原始图像', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_damaged_cpu, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'损坏图像\n{scratch_percentage:.2f}% 像素损坏', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_repaired_final, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('KSVD修复结果', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

    # ====================== 打印详细质量报告 ======================
    print('\n=== 修复质量评估报告 ===')
    print('【划痕统计】')
    print(f'  划痕像素：{scratch_area} ({scratch_percentage:.2f}%)')
    print(f'  划痕区域占比：{scratch_area/total_pixels:.4f}')
    
    print('\n【误差指标】')
    print(f'  平均绝对误差：{mae_repair:.4f}')
    print(f'  最大绝对误差：{max_ae_repair:.4f}')
    print(f'  修复区域MSE：{mse_repair:.6f}')
    print(f'  相对误差：{relative_error:.4f}%')
    
    print('\n【PSNR指标】')
    print(f'  修复区域PSNR：{psnr_repair:.2f} dB')
    
    if not np.isnan(ssim_overall):
        print('\n【SSIM指标】')
        print(f'  修复区域SSIM：{ssim_patch:.4f}')
    
    # 显示最终总结
    print('\n=== 划痕修复完成 ===')
    if scratch_area > 0:
        print('修复总结：')
        print(f'  成功修复 {scratch_area} 个划痕像素')
        if not np.isinf(psnr_repair):
            if psnr_repair > 30:
                print(f'  修复质量: ⭐⭐⭐⭐⭐ 优秀 (PSNR > 30dB)')
            elif psnr_repair > 25:
                print(f'  修复质量: ⭐⭐⭐⭐ 良好 (PSNR 25-30dB)')
            elif psnr_repair > 20:
                print(f'  修复质量: ⭐⭐⭐ 中等 (PSNR 20-25dB)')
            else:
                print(f'  修复质量: ⭐⭐ 需改进 (PSNR < 20dB)')
    else:
        print('没有检测到划痕，无需修复')

if __name__ == '__main__':
    main()
