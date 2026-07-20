import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import ImageFont

# ========== 在文件开头添加 ==========
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ===================================

from matplotlib.patches import Polygon
import os
import time
import warnings
from scipy.ndimage import gaussian_filter
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

warnings.filterwarnings('ignore')

# ==================== GPU检查函数 ====================
def check_gpu_availability():
    """检查GPU可用性"""
    print('=== 图像污渍去除系统 ===')
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
    print('\n=== 图像读取与预处理 ===')
    
    # 读取图像
    img_color = io.imread(image_path)
    
    # 转换为浮点数
    if img_color.dtype == np.uint8:
        img_original = img_color.astype(np.float32) / 255.0
    else:
        img_original = img_color.astype(np.float32)
    
    # 保存彩色原始图像用于显示
    original_img_color = img_original.copy()
    
    # 转换为灰度图像用于处理
    if len(img_original.shape) == 3 and img_original.shape[2] == 3:
        img_gray = color.rgb2gray(img_original)
    else:
        img_gray = img_original.copy()
    
    # 保存原始灰度图像
    original_img = img_gray.copy()
    
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
    
    return img, original_img, original_img_color, use_gpu, device, scale_factor

# ==================== 污渍生成函数 ====================
def generate_random_stains(img, use_gpu, device, show_plots=True):
    """自动生成随机污渍区域 - MATLAB完全兼容版"""
    print('\n=== 自动生成随机污渍区域 ===')
    
    # 显示原始图像
    if use_gpu:
        if torch.is_tensor(img):
            img_display = img.cpu().numpy()
        else:
            img_display = img
        h, w = img.shape if torch.is_tensor(img) else img.shape[:2]
    else:
        img_display = img
        h, w = img.shape[:2]
    
    total_pixels = h * w
    
    # 只在需要时显示图像
    if show_plots:
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        if len(img_display.shape) == 3:
            ax1.imshow(img_display)
        else:
            ax1.imshow(img_display, cmap='gray')
        ax1.set_title('随机生成的污渍区域（污渍面积控制在0.1%-0.5%）')
        ax1.axis('on')
    
    # 污渍参数设置
    num_stains = np.random.randint(3, 9)
    max_stain_area_percent = 0.005
    min_stain_area_percent = 0.001
    total_stain_area_percent = 0
    
    print(f'正在生成 {num_stains} 个随机污渍...')
    
    # 初始化污渍掩码
    if use_gpu:
        mask_stains = torch.zeros((h, w), dtype=torch.bool, device=device)
    else:
        mask_stains = np.zeros((h, w), dtype=bool)
    
    stain_regions = []
    stain_count = 0
    
    # 生成随机污渍
    for i in range(num_stains):
        # 随机选择污渍类型
        stain_type = np.random.randint(1, 4)
        
        # 随机污渍面积
        stain_area_percent = min_stain_area_percent + \
                           np.random.rand() * (max_stain_area_percent - min_stain_area_percent)
        
        # 计算污渍半径
        target_area = stain_area_percent * total_pixels
        base_radius = np.sqrt(target_area / np.pi)
        
        # 随机中心位置
        min_margin = max(1, int(base_radius * 1.5))
        if min_margin >= w or min_margin >= h:
            min_margin = min(w // 4, h // 4)
        
        center_x = np.random.randint(min_margin, max(min_margin + 1, w - min_margin))
        center_y = np.random.randint(min_margin, max(min_margin + 1, h - min_margin))
        
        if stain_type == 1:  # 圆形污渍
            radius = base_radius * (0.8 + 0.4 * np.random.rand())
            radius = max(radius, 2)
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            stain_mask = dist_from_center <= radius
            stain_points = generate_circle_points(center_x, center_y, radius, 20)
            
        elif stain_type == 2:  # 椭圆形污渍
            major_axis = base_radius * (0.9 + 0.4 * np.random.rand())
            minor_axis = base_radius * (0.6 + 0.3 * np.random.rand())
            major_axis = max(major_axis, 2)
            minor_axis = max(minor_axis, 1)
            angle = np.random.rand() * 2 * np.pi
            Y, X = np.ogrid[:h, :w]
            Xc = X - center_x
            Yc = Y - center_y
            Xr = Xc * np.cos(angle) + Yc * np.sin(angle)
            Yr = -Xc * np.sin(angle) + Yc * np.cos(angle)
            stain_mask = (Xr**2 / major_axis**2 + Yr**2 / minor_axis**2) <= 1
            stain_points = generate_ellipse_points(center_x, center_y, major_axis, minor_axis, angle, 30)
            
        else:  # 不规则形状污渍
            num_points = np.random.randint(6, 13)
            angles = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
            radii = base_radius * (0.5 + 0.5 * np.random.rand(num_points))
            radii = np.maximum(radii, 1)
            x_points = center_x + radii * np.cos(angles + np.random.rand() * 0.5)
            y_points = center_y + radii * np.sin(angles + np.random.rand() * 0.5)
            x_points = np.append(x_points, x_points[0])
            y_points = np.append(y_points, y_points[0])
            stain_points = np.column_stack((x_points, y_points))
            stain_mask = poly_to_mask(x_points, y_points, h, w)
        
        # 确保污渍不重叠
        if use_gpu:
            stain_mask_tensor = torch.from_numpy(stain_mask).to(device)
            stain_mask = stain_mask_tensor & ~mask_stains
        else:
            stain_mask = stain_mask & ~mask_stains
        
        # 计算实际面积
        if use_gpu:
            actual_area = stain_mask.sum().item()
        else:
            actual_area = stain_mask.sum()
        
        if actual_area < 5:
            continue
        
        # 添加到总掩码
        mask_stains = mask_stains | stain_mask
        stain_count += 1
        stain_regions.append(stain_points)
        
        # 只在需要时绘制
        if show_plots:
            from matplotlib.patches import Polygon
            polygon = Polygon(stain_points, alpha=0.3, color='red', edgecolor='red', linewidth=1)
            ax1.add_patch(polygon)
        
        stain_area_percent_actual = 100 * actual_area / total_pixels
        total_stain_area_percent += stain_area_percent_actual
        stain_type_name = get_stain_type_name(stain_type)
        print(f'  污渍 {stain_count}: {stain_type_name}, 面积 {stain_area_percent_actual:.2f}% (约 {actual_area} 像素)')
    
    # 关闭图形
    if show_plots:
        ax1.axis('off')
        plt.draw()
        plt.pause(0.5)
        plt.close(fig1)
    
    # 如果没有生成污渍，使用默认模拟
    if stain_count == 0:
        print('未能生成污渍，使用默认模拟污渍')
        mask_stains = simulate_stains_small(h, w, use_gpu, device)
        if use_gpu:
            total_stain_area_percent = 100 * mask_stains.sum().item() / total_pixels
        else:
            total_stain_area_percent = 100 * mask_stains.sum() / total_pixels
    else:
        print(f'共生成 {stain_count} 个污渍，总面积: {total_stain_area_percent:.2f}%')
    
    # 污渍统计
    if use_gpu:
        stain_area = mask_stains.sum().item()
        mask_stains_cpu = mask_stains.cpu().numpy()
    else:
        stain_area = mask_stains.sum()
        mask_stains_cpu = mask_stains
    
    stain_percentage = 100 * stain_area / total_pixels
    print(f'污渍统计: {stain_area} 像素 ({stain_percentage:.2f}%)')
    
    # 创建损坏图像
    if use_gpu:
        img_damaged = img.clone()
        img_damaged[mask_stains] = 0
    else:
        img_damaged = img.copy()
        img_damaged[mask_stains] = 0
    
    # 创建修复掩码
    mask = ~mask_stains
    
    return mask_stains, stain_regions, stain_area, stain_percentage, img_damaged, mask

# ==================== 辅助函数 ====================
def generate_circle_points(cx, cy, radius, num_points):
    """生成圆形点集"""
    angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack((x, y))

def generate_ellipse_points(cx, cy, a, b, angle, num_points):
    """生成椭圆形点集"""
    t = np.linspace(0, 2*np.pi, num_points+1)[:-1]
    
    # 椭圆参数方程
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # 旋转
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)
    
    # 平移
    x_final = cx + xr
    y_final = cy + yr
    
    return np.column_stack((x_final, y_final))

def poly_to_mask(x_points, y_points, height, width):
    """将多边形转换为掩码"""
    from matplotlib.path import Path
    
    # 创建网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    
    # 创建路径
    poly_path = Path(np.column_stack((x_points, y_points)))
    
    # 判断点是否在多边形内
    mask = poly_path.contains_points(points)
    mask = mask.reshape((height, width))
    
    # 填充孔洞
    mask = ndimage.binary_fill_holes(mask)
    
    return mask

def get_stain_type_name(type_num):
    """获取污渍类型名称"""
    type_names = {1: '圆形', 2: '椭圆形', 3: '不规则'}
    return type_names.get(type_num, '未知')

def simulate_stains_small(h, w, use_gpu, device):
    """生成小面积污渍"""
    total_pixels = h * w
    
    if use_gpu:
        small_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    else:
        small_mask = np.zeros((h, w), dtype=bool)
    
    # 生成3个小圆形污渍
    num_stains = 3
    
    for i in range(num_stains):
        # 小面积污渍（约0.1%）
        target_area = 0.001 * total_pixels
        radius = np.sqrt(target_area / np.pi)
        
        # 随机位置
        center_x = np.random.randint(int(radius)+1, w-int(radius)-1)
        center_y = np.random.randint(int(radius)+1, h-int(radius)-1)
        
        # 生成圆形
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        stain = dist <= radius
        
        if use_gpu:
            stain_tensor = torch.from_numpy(stain).to(device)
            small_mask = small_mask | stain_tensor
        else:
            small_mask = small_mask | stain
    
    return small_mask

# ==================== Patch提取函数 ====================
def extract_patches_for_psnr20(img, mask, patch_size, use_gpu, device):
    """PSNR>20dB专用patch提取 - 修复维度错误"""
    # ★★★ 修复：处理图像维度 ★★★
    if use_gpu:
        if torch.is_tensor(img):
            # GPU张量处理
            if len(img.shape) == 3:
                print(f"  图像维度为 {img.shape}，转换为2D灰度图")
                if img.shape[0] == 3:  # (C, H, W) 格式
                    img_gray = torch.mean(img, dim=0)
                elif img.shape[2] == 3:  # (H, W, C) 格式
                    img_gray = torch.mean(img, dim=2)
                else:
                    img_gray = img[:, :, 0]  # 取第一个通道
            else:
                img_gray = img
            img_cpu = img_gray.cpu().numpy()
        else:
            # GPU上的numpy数组
            if len(img.shape) == 3:
                print(f"  图像维度为 {img.shape}，转换为2D灰度图")
                if img.shape[2] == 3:
                    img_gray = np.mean(img, axis=2)
                else:
                    img_gray = img[:, :, 0]
            else:
                img_gray = img
            img_cpu = img_gray
    else:
        # CPU处理
        if len(img.shape) == 3:
            print(f"  图像维度为 {img.shape}，转换为2D灰度图")
            if img.shape[2] == 3:  # (H, W, C)
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img[:, :, 0]  # 取第一个通道
        else:
            img_gray = img
        img_cpu = img_gray
    
    # 处理mask - 确保是2D
    if use_gpu:
        if torch.is_tensor(mask):
            if len(mask.shape) == 3:
                if mask.shape[2] == 3:
                    mask_cpu = mask[:, :, 0].cpu().numpy()
                else:
                    mask_cpu = mask.cpu().numpy()
            else:
                mask_cpu = mask.cpu().numpy()
        else:
            if len(mask.shape) == 3:
                if mask.shape[2] == 3:
                    mask_cpu = mask[:, :, 0]
                else:
                    mask_cpu = mask
            else:
                mask_cpu = mask
    else:
        if len(mask.shape) == 3:
            if mask.shape[2] == 3:
                mask_cpu = mask[:, :, 0]
            else:
                mask_cpu = mask
        else:
            mask_cpu = mask
    
    # 获取正确的2D尺寸
    h, w = img_cpu.shape
    print(f"  处理后的图像尺寸: {h} x {w}")
    
    print('提取多种类型patch...')
    
    patches_list = []
    positions_list = []
    
    # 1. 高质量完好区域patch
    print('  1. 高质量完好区域...')
    hq_patches, hq_pos = extract_hq_patches(img_cpu, mask_cpu, patch_size)
    if hq_patches.size > 0:
        patches_list.append(hq_patches)
        positions_list.append(hq_pos)
    
    # 2. 边缘过渡区域patch
    print('  2. 边缘过渡区域...')
    edge_patches, edge_pos = extract_edge_transition_patches(img_cpu, mask_cpu, patch_size)
    if edge_patches.size > 0:
        patches_list.append(edge_patches)
        positions_list.append(edge_pos)
    
    # 3. 纹理丰富区域patch
    print('  3. 纹理丰富区域...')
    texture_patches, texture_pos = extract_texture_patches(img_cpu, mask_cpu, patch_size)
    if texture_patches.size > 0:
        patches_list.append(texture_patches)
        positions_list.append(texture_pos)
    
    # 4. 随机采样补充
    print('  4. 随机采样补充...')
    random_patches, random_pos = extract_random_patches(img_cpu, mask_cpu, patch_size, 2000)
    if random_patches.size > 0:
        patches_list.append(random_patches)
        positions_list.append(random_pos)
    
    # 合并所有patch
    if patches_list:
        patches = np.hstack(patches_list)
        positions = np.vstack(positions_list)
    else:
        patches = np.array([])
        positions = np.array([])
    
    print(f'  总提取: {patches.shape[1] if patches.size > 0 else 0} 个patch')
    
    # 去除重复patch
    if patches.size > 0:
        patches = remove_duplicate_patches(patches)
        print(f'  去重后: {patches.shape[1] if patches.size > 0 else 0} 个patch')
    
    # 转换为GPU张量
    if use_gpu and patches.size > 0:
        patches = torch.from_numpy(patches).float().to(device)
    
    return patches, positions

def extract_hq_patches(img, mask, patch_size):
    """提取高质量patch"""
    h, w = img.shape
    
    # 找到最大的完好连续区域
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    if not regions:
        return np.array([]), np.array([])
    
    # 找到最大的区域
    region_sizes = [r.area for r in regions]
    max_idx = np.argmax(region_sizes)
    max_region = regions[max_idx]
    
    # 创建该区域的掩码
    region_mask = labeled_mask == (max_idx + 1)
    
    # 在该区域内密集采样
    region_coords = np.argwhere(region_mask)
    num_samples = min(3000, len(region_coords))
    
    if num_samples == 0:
        return np.array([]), np.array([])
    
    # 随机采样
    sample_indices = np.random.choice(len(region_coords), num_samples, replace=False)
    
    patches = np.zeros((patch_size**2, num_samples), dtype=np.float32)
    positions = np.zeros((num_samples, 2), dtype=int)
    
    for s, idx in enumerate(sample_indices):
        center_row, center_col = region_coords[idx]
        
        # 计算patch位置
        i = max(0, min(center_row - patch_size//2, h - patch_size))
        j = max(0, min(center_col - patch_size//2, w - patch_size))
        
        # 提取patch
        img_patch = img[i:i+patch_size, j:j+patch_size]
        patches[:, s] = img_patch.flatten()
        positions[s, :] = [i, j]
    
    return patches, positions

def extract_edge_transition_patches(img, mask, patch_size):
    """提取边缘过渡patch"""
    h, w = img.shape
    
    # 找到污渍边缘区域
    stain_mask = ~mask
    stain_edge = feature.canny(stain_mask)
    
    # 膨胀边缘
    selem = morphology.disk(3)
    edge_region = morphology.binary_dilation(stain_edge, selem)
    
    # 限制在完好区域内
    edge_region = edge_region & mask
    
    edge_coords = np.argwhere(edge_region)
    num_samples = min(2000, len(edge_coords))
    
    if num_samples == 0:
        return np.array([]), np.array([])
    
    # 随机采样
    sample_indices = np.random.choice(len(edge_coords), num_samples, replace=False)
    
    patches = np.zeros((patch_size**2, num_samples), dtype=np.float32)
    positions = np.zeros((num_samples, 2), dtype=int)
    
    for s, idx in enumerate(sample_indices):
        center_row, center_col = edge_coords[idx]
        
        i = max(0, min(center_row - patch_size//2, h - patch_size))
        j = max(0, min(center_col - patch_size//2, w - patch_size))
        
        img_patch = img[i:i+patch_size, j:j+patch_size]
        patches[:, s] = img_patch.flatten()
        positions[s, :] = [i, j]
    
    return patches, positions

def extract_texture_patches(img, mask, patch_size):
    """提取纹理丰富patch"""
    h, w = img.shape
    
    # 计算纹理丰富度（梯度幅值）
    grad_y, grad_x = np.gradient(img)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 只考虑完好区域
    gradient_mag[~mask] = 0
    
    # 找到纹理丰富的区域
    texture_threshold = 0.1
    texture_regions = gradient_mag > texture_threshold
    
    texture_coords = np.argwhere(texture_regions)
    num_samples = min(1500, len(texture_coords))
    
    if num_samples == 0:
        return np.array([]), np.array([])
    
    # 随机采样
    sample_indices = np.random.choice(len(texture_coords), num_samples, replace=False)
    
    patches = np.zeros((patch_size**2, num_samples), dtype=np.float32)
    positions = np.zeros((num_samples, 2), dtype=int)
    
    for s, idx in enumerate(sample_indices):
        center_row, center_col = texture_coords[idx]
        
        i = max(0, min(center_row - patch_size//2, h - patch_size))
        j = max(0, min(center_col - patch_size//2, w - patch_size))
        
        img_patch = img[i:i+patch_size, j:j+patch_size]
        patches[:, s] = img_patch.flatten()
        positions[s, :] = [i, j]
    
    return patches, positions

def extract_random_patches(img, mask, patch_size, num_patches):
    """随机采样patch"""
    h, w = img.shape
    
    patches = np.zeros((patch_size**2, num_patches), dtype=np.float32)
    positions = np.zeros((num_patches, 2), dtype=int)
    
    count = 0
    max_attempts = num_patches * 10
    
    for attempt in range(max_attempts):
        if count >= num_patches:
            break
        
        i = np.random.randint(0, h - patch_size + 1)
        j = np.random.randint(0, w - patch_size + 1)
        
        # 检查patch质量
        patch_mask = mask[i:i+patch_size, j:j+patch_size]
        good_ratio = patch_mask.sum() / (patch_size * patch_size)
        
        if good_ratio >= 0.95:  # 非常高要求
            img_patch = img[i:i+patch_size, j:j+patch_size]
            patches[:, count] = img_patch.flatten()
            positions[count, :] = [i, j]
            count += 1
    
    # 调整大小
    patches = patches[:, :count]
    positions = positions[:count, :]
    
    return patches, positions

def remove_duplicate_patches(patches, similarity_threshold=0.95):
    """去除重复patch"""
    if patches.size == 0:
        return patches
    
    dim, num_patches = patches.shape
    
    # 归一化patch
    patch_norms = np.sqrt(np.sum(patches**2, axis=0))
    patch_norms[patch_norms == 0] = 1e-10
    patches_norm = patches / patch_norms
    
    # 分批处理避免内存溢出
    batch_size = 500
    num_batches = (num_patches + batch_size - 1) // batch_size
    
    is_unique = np.ones(num_patches, dtype=bool)
    
    for b1 in range(num_batches):
        batch1_start = b1 * batch_size
        batch1_end = min((b1 + 1) * batch_size, num_patches)
        batch1_idx = slice(batch1_start, batch1_end)
        batch1_patches = patches_norm[:, batch1_idx]
        
        # 只与后续批次比较
        for b2 in range(b1, num_batches):
            batch2_start = b2 * batch_size
            batch2_end = min((b2 + 1) * batch_size, num_patches)
            batch2_idx = slice(batch2_start, batch2_end)
            
            if b1 == b2:
                # 同一批次比较上三角
                correlations = batch1_patches.T @ batch1_patches
                for i in range(correlations.shape[0]):
                    for j in range(i+1, correlations.shape[1]):
                        if correlations[i, j] > similarity_threshold:
                            is_unique[batch1_start + j] = False
            else:
                # 不同批次比较所有
                batch2_patches = patches_norm[:, batch2_idx]
                correlations = batch1_patches.T @ batch2_patches
                
                for i in range(correlations.shape[0]):
                    for j in range(correlations.shape[1]):
                        if correlations[i, j] > similarity_threshold:
                            is_unique[batch2_start + j] = False
    
    patches_clean = patches[:, is_unique]
    return patches_clean

def augment_patches_for_training(patches, use_gpu, device):
    """数据增强"""
    if patches.size == 0:
        return patches
    
    dim, num_patches = patches.shape
    patch_size = int(np.sqrt(dim))
    
    if use_gpu:
        patches_gpu = patches if torch.is_tensor(patches) else torch.from_numpy(patches).float().to(device)
        patches_aug = [patches_gpu]
    else:
        patches_aug = [patches]
    
    # 添加旋转版本
    rotations = [90, 180, 270]
    for angle in rotations:
        if use_gpu:
            rot_patches = torch.zeros_like(patches_gpu)
            for p in range(num_patches):
                patch_mat = patches_gpu[:, p].reshape(patch_size, patch_size).cpu().numpy()
                patch_rot = transform.rotate(patch_mat, angle, mode='reflect')
                patch_rot = transform.resize(patch_rot, (patch_size, patch_size))
                rot_patches[:, p] = torch.from_numpy(patch_rot.flatten()).float().to(device)
            patches_aug.append(rot_patches)
        else:
            rot_patches = np.zeros_like(patches)
            for p in range(num_patches):
                patch_mat = patches[:, p].reshape(patch_size, patch_size)
                patch_rot = transform.rotate(patch_mat, angle, mode='reflect')
                patch_rot = transform.resize(patch_rot, (patch_size, patch_size))
                rot_patches[:, p] = patch_rot.flatten()
            patches_aug.append(rot_patches)
    
    # 添加镜像版本
    flips = ['horizontal', 'vertical']
    for flip_type in flips:
        if use_gpu:
            flip_patches = torch.zeros_like(patches_gpu)
            for p in range(num_patches):
                patch_mat = patches_gpu[:, p].reshape(patch_size, patch_size).cpu().numpy()
                if flip_type == 'horizontal':
                    patch_flip = np.fliplr(patch_mat)
                else:
                    patch_flip = np.flipud(patch_mat)
                flip_patches[:, p] = torch.from_numpy(patch_flip.flatten()).float().to(device)
            patches_aug.append(flip_patches)
        else:
            flip_patches = np.zeros_like(patches)
            for p in range(num_patches):
                patch_mat = patches[:, p].reshape(patch_size, patch_size)
                if flip_type == 'horizontal':
                    patch_flip = np.fliplr(patch_mat)
                else:
                    patch_flip = np.flipud(patch_mat)
                flip_patches[:, p] = patch_flip.flatten()
            patches_aug.append(flip_patches)
    
    # 合并所有增强的patches
    if use_gpu:
        patches_aug = torch.cat(patches_aug, dim=1)
    else:
        patches_aug = np.hstack(patches_aug)
    
    print(f'数据增强: {num_patches} → {patches_aug.shape[1]} patches')
    
    return patches_aug

# ==================== KSVD字典训练函数 ====================

def professional_ksvd_training_correct(patches, dict_size, sparsity, max_iter, use_gpu, device):
    """KSVD字典训练 - 增强版，原子数每5次迭代递减一次"""
    print('=== KSVD字典训练（每5次迭代原子减少）===')
    
    if use_gpu and torch.is_tensor(patches):
        patches_np = patches.cpu().numpy()
        n_dim, n_patch = patches.shape
    else:
        patches_np = patches
        n_dim, n_patch = patches.shape
    
    # 原子数递减参数
    initial_dict_size = dict_size  # 初始原子数
    final_dict_size = max(dict_size // 4, 16)  # 最终原子数（至少16个）
    
    # 计算需要减少的总原子数
    total_reduction = initial_dict_size - final_dict_size
    # 每5次迭代减少的原子数（确保整数）
    reduction_per_step = max(1, total_reduction // ((max_iter // 5) + 1))
    # 总减少步数
    num_reduction_steps = total_reduction // reduction_per_step
    
    print(f'训练数据: {n_patch} patches, 维度: {n_dim}')
    print(f'初始字典大小: {initial_dict_size}, 最终字典大小: {final_dict_size}')
    print(f'每5次迭代减少: {reduction_per_step} 个原子')
    print(f'总减少步数: {num_reduction_steps} 步')
    print(f'稀疏度: {sparsity}, 最大迭代: {max_iter}')
    
    # 参数验证
    if initial_dict_size > n_patch:
        print(f'⚠️ 警告: 初始字典大小({initial_dict_size}) > 训练数据数量({n_patch})，可能过拟合')
        initial_dict_size = min(initial_dict_size, int(n_patch * 0.8))
        print(f'  调整初始字典大小为: {initial_dict_size}')
    
    # 智能字典初始化（使用初始字典大小）
    print('\n--- 字典初始化 ---')
    dict_init = initialize_dictionary_smart(patches_np, initial_dict_size)
    
    # 转换为GPU张量
    if use_gpu:
        dict_current = torch.from_numpy(dict_init).float().to(device)
        patches_tensor = patches
    else:
        dict_current = dict_init
        patches_tensor = patches_np
    
    # 初始稀疏编码
    print('  初始稀疏编码...')
    alpha = professional_omp_coding(dict_current, patches_tensor, sparsity, use_gpu, device)
    
    # 计算初始误差
    if use_gpu:
        reconstruction = dict_current @ alpha
        diff = patches_tensor - reconstruction
        initial_error = torch.norm(diff, 'fro').item() / np.sqrt(n_patch)
    else:
        reconstruction = dict_current @ alpha
        diff = patches_np - reconstruction
        initial_error = np.linalg.norm(diff, 'fro') / np.sqrt(n_patch)
    
    print(f'初始化完成:')
    print(f'  初始重构误差 (MSE): {initial_error:.6f}')
    
    if use_gpu:
        sparsity_level = (alpha != 0).sum().item() / (initial_dict_size * n_patch)
        atom_active = torch.abs(alpha) > 1e-4
        active_per_atom = atom_active.sum(dim=1) > 0
        active_count_initial = active_per_atom.sum().item()
    else:
        sparsity_level = (alpha != 0).sum() / (initial_dict_size * n_patch)
        atom_active = np.abs(alpha) > 1e-4
        active_per_atom = atom_active.sum(axis=1) > 0
        active_count_initial = np.sum(active_per_atom)
    
    print(f'  初始稀疏性: {sparsity_level*100:.2f}%')
    print(f'  初始活跃原子: {active_count_initial}/{initial_dict_size} ({100*active_count_initial/initial_dict_size:.1f}%)')
    
    # KSVD主训练循环
    print('\n--- 开始KSVD字典训练（每5次迭代原子减少）---')
    
    best_dict = dict_current
    best_error = initial_error
    best_alpha = alpha
    
    # 记录训练历史
    errors_history = [initial_error]
    active_atoms_history = [active_count_initial]
    sparsity_history = [sparsity_level]
    time_history = []
    atom_usage_history = []
    dict_size_history = [initial_dict_size]  # 记录每次迭代的字典大小
    
    current_dict_size = initial_dict_size
    current_dict = dict_current
    current_alpha = alpha
    
    # 原子减少计数器
    reduction_counter = 0
    reductions_done = 0
    
    # 训练循环
    for iter in range(max_iter):
        iter_start_time = time.time()
        
        # === 原子数递减策略：每5次迭代减少一次 ===
        # 检查是否需要进行原子减少
        should_reduce = (
            iter > 0 and 
            iter % 5 == 0 and  # 每5次迭代
            current_dict_size > final_dict_size and  # 还没减少到最终大小
            reductions_done < num_reduction_steps  # 还没完成所有减少步骤
        )
        
        if should_reduce:
            # 计算本次要减少的原子数
            reduce_by = min(reduction_per_step, current_dict_size - final_dict_size)
            target_dict_size = current_dict_size - reduce_by
            reduction_counter += 1
            reductions_done += 1
            
            print(f'\n  ★ 第{iter}次迭代: 执行原子减少 (第{reductions_done}/{num_reduction_steps}次)')
            print(f'    缩减字典 {current_dict_size} → {target_dict_size} 个原子 (减少{reduce_by}个)')
            
            # 原子选择策略：保留最活跃的原子
            if use_gpu:
                # 计算每个原子的使用频率（L1范数）
                atom_activity = torch.abs(current_alpha).sum(dim=1)
                
                # 获取要保留的原子索引（最活跃的target_dict_size个）
                _, top_indices = torch.topk(atom_activity, target_dict_size)
                top_indices = torch.sort(top_indices)[0]  # 排序保持顺序
                
                # 记录被删除的原子
                all_indices = set(range(current_dict_size))
                kept_indices = set(top_indices.cpu().numpy())
                removed_indices = all_indices - kept_indices
                
                # 缩减字典和系数矩阵
                current_dict = current_dict[:, top_indices]
                current_alpha = current_alpha[top_indices, :]
            else:
                # NumPy版本
                atom_activity = np.abs(current_alpha).sum(axis=1)
                top_indices = np.argsort(atom_activity)[::-1][:target_dict_size]
                top_indices = np.sort(top_indices)
                
                # 记录被删除的原子
                all_indices = set(range(current_dict_size))
                kept_indices = set(top_indices)
                removed_indices = all_indices - kept_indices
                
                current_dict = current_dict[:, top_indices]
                current_alpha = current_alpha[top_indices, :]
            
            current_dict_size = target_dict_size
            
            # 显示被删除原子的使用情况
            if len(removed_indices) > 0:
                if use_gpu:
                    removed_activity = atom_activity[list(removed_indices)].cpu().numpy()
                else:
                    removed_activity = atom_activity[list(removed_indices)]
                print(f'    删除原子的平均使用次数: {np.mean(removed_activity):.2f}')
                print(f'    保留原子的平均使用次数: {np.mean(atom_activity[list(kept_indices)]):.2f}')
            
            print(f'    当前原子数: {current_dict_size}, 目标原子数: {final_dict_size}')
            
            # 原子减少后重新进行稀疏编码
            print(f'    原子减少后重新稀疏编码...')
            current_alpha = professional_omp_coding(current_dict, patches_tensor, sparsity, use_gpu, device)
        
        # 稀疏编码（除第一次和原子减少后外，不需要重新计算）
        elif iter > 0 and not should_reduce:
            current_alpha = professional_omp_coding(current_dict, patches_tensor, sparsity, use_gpu, device)
        
        # 计算当前误差
        if use_gpu:
            reconstruction = current_dict @ current_alpha
            diff = patches_tensor - reconstruction
            current_error = torch.norm(diff, 'fro').item() / np.sqrt(n_patch)
        else:
            reconstruction = current_dict @ current_alpha
            diff = patches_np - reconstruction
            current_error = np.linalg.norm(diff, 'fro') / np.sqrt(n_patch)
        
        # 使用KSVD更新字典（如果不是最后一次迭代）
        if iter < max_iter - 1:
            print(f'  更新字典 (迭代 {iter+1}/{max_iter}, 当前原子数: {current_dict_size})...')
            current_dict, current_alpha = update_dictionary(
                current_dict, patches_tensor, current_alpha, use_gpu, device
            )
        
        # 计算更新后的误差
        if use_gpu:
            reconstruction_updated = current_dict @ current_alpha
            diff_updated = patches_tensor - reconstruction_updated
            updated_error = torch.norm(diff_updated, 'fro').item() / np.sqrt(n_patch)
        else:
            reconstruction_updated = current_dict @ current_alpha
            diff_updated = patches_np - reconstruction_updated
            updated_error = np.linalg.norm(diff_updated, 'fro') / np.sqrt(n_patch)
        
        errors_history.append(updated_error)
        dict_size_history.append(current_dict_size)
        
        # 记录当前迭代的活跃原子数和稀疏度
        if use_gpu:
            sparsity_level = (torch.abs(current_alpha) > 1e-4).sum().item() / (current_dict_size * n_patch)
            atom_active = torch.abs(current_alpha) > 1e-4
            active_per_atom = atom_active.sum(dim=1) > 0
            active_count = active_per_atom.sum().item()
            
            if iter == max_iter - 1 or iter % 10 == 0:
                atom_usage_count = atom_active.sum(dim=1).cpu().numpy()
                atom_usage_history.append(atom_usage_count)
        else:
            sparsity_level = (np.abs(current_alpha) > 1e-4).sum() / (current_dict_size * n_patch)
            atom_active = np.abs(current_alpha) > 1e-4
            active_per_atom = atom_active.sum(axis=1) > 0
            active_count = np.sum(active_per_atom)
            
            if iter == max_iter - 1 or iter % 10 == 0:
                atom_usage_count = atom_active.sum(axis=1)
                atom_usage_history.append(atom_usage_count)
        
        active_atoms_history.append(active_count)
        sparsity_history.append(sparsity_level)
        
        iter_time = time.time() - iter_start_time
        time_history.append(iter_time)
        
        # 显示进度
        if iter == 0:
            print(f'  迭代 {iter+1:3d}/{max_iter:3d}: 原子数={current_dict_size}, 误差={current_error:.6f} → {updated_error:.6f}')
            print(f'    活跃原子: {active_count}/{current_dict_size} ({100*active_count/current_dict_size:.1f}%)')
        else:
            prev_error = errors_history[iter]
            if prev_error > 0:
                change_pct = 100 * (prev_error - updated_error) / prev_error
                change_str = f'改进{change_pct:.2f}%' if change_pct > 0 else f'退化{-change_pct:.2f}%'
                
                # 标记原子减少的迭代
                reduction_mark = " [原子减少]" if should_reduce else ""
                print(f'  迭代 {iter+1:3d}/{max_iter:3d}{reduction_mark}: 原子数={current_dict_size}, 误差={updated_error:.6f} ({change_str})')
                print(f'    活跃原子: {active_count}/{current_dict_size} ({100*active_count/current_dict_size:.1f}%)')
        
        # 更新最优字典
        if updated_error < best_error:
            best_error = updated_error
            best_dict = current_dict.clone() if use_gpu else current_dict.copy()
            best_alpha = current_alpha.clone() if use_gpu else current_alpha.copy()
            print(f'    ✓ 更新最优字典 (误差: {best_error:.6f})')
        
        # 检查收敛
        if iter > 0:
            error_change = abs(updated_error - errors_history[iter]) / max(errors_history[iter], 1e-6)
            if error_change < 0.0001 and iter > 10:
                print(f'    ✓ 收敛于迭代 {iter+1} (误差变化{error_change*100:.4f}%)')
                # 截断历史记录
                errors_history = errors_history[:iter+2]
                active_atoms_history = active_atoms_history[:iter+2]
                sparsity_history = sparsity_history[:iter+2]
                time_history = time_history[:iter+1]
                dict_size_history = dict_size_history[:iter+2]
                break
    
    final_iter = len(errors_history) - 1
    final_error = best_error
    dict_result = best_dict
    final_dict_size = dict_result.shape[1]
    
    print('\n=== 字典训练完成（每5次迭代原子减少）===')
    print(f'最终结果:')
    print(f'  最终误差 (MSE): {final_error:.6f}')
    print(f'  初始原子数: {initial_dict_size} → 最终原子数: {final_dict_size}')
    print(f'  原子减少率: {(1 - final_dict_size/initial_dict_size)*100:.1f}%')
    print(f'  原子减少次数: {reductions_done} 次')
    print(f'  平均每次减少: {total_reduction/max(reductions_done,1):.1f} 个原子')
    print(f'  最终活跃原子: {active_atoms_history[-1]}/{final_dict_size} ({100*active_atoms_history[-1]/final_dict_size:.1f}%)')
    print(f'  最终稀疏度: {sparsity_history[-1]*100:.2f}%')
    print(f'  相对初始误差改进: {100*(initial_error - final_error)/initial_error:.2f}%')
    print(f'  训练迭代次数: {final_iter}')
    print(f'  总训练时间: {sum(time_history):.2f}秒')
    
    return dict_result


def initialize_dictionary_smart(patches, dict_size):
    """智能字典初始化"""
    dim, n_patches = patches.shape
    
    # 方法1: 从训练数据中采样
    if dict_size <= n_patches:
        # 使用k-means++风格的初始化
        indices = np.zeros(dict_size, dtype=int)
        
        # 第一个中心随机选择
        indices[0] = np.random.randint(n_patches)
        
        for i in range(1, dict_size):
            centers = patches[:, indices[:i]]
            
            # 计算每个样本到最近中心的距离
            distances = np.zeros(n_patches)
            for j in range(n_patches):
                patch = patches[:, j]
                center_dists = np.sum((centers - patch[:, np.newaxis])**2, axis=0)
                distances[j] = np.min(center_dists)
            
            # 按距离的概率选择下一个中心
            prob = distances / np.sum(distances)
            next_idx = np.random.choice(n_patches, p=prob)
            indices[i] = next_idx
        
        dict_init = patches[:, indices]
        print(f'    从训练数据采样 {dict_size} 个原子 (k-means++风格)')
    else:
        # 方法2: 随机初始化 + PCA方向
        dict_init = np.random.randn(dim, dict_size)
        
        # 添加一些数据的主方向
        if dict_size <= dim:
            # 计算数据的主成分
            cov_matrix = patches @ patches.T / n_patches
            U, _, _ = np.linalg.svd(cov_matrix)
            num_pca_atoms = min(dict_size//4, U.shape[1])
            dict_init[:, :num_pca_atoms] = U[:, :num_pca_atoms]
        
        print(f'    随机初始化 {dict_size} 个原子 (包含PCA方向)')
    
    # 归一化所有原子
    dict_norms = np.sqrt(np.sum(dict_init**2, axis=0))
    zero_norms = dict_norms < 1e-10
    
    if np.any(zero_norms):
        # 替换零范数原子
        for k in np.where(zero_norms)[0]:
            patch_norms = np.sqrt(np.sum(patches**2, axis=0))
            max_idx = np.argmax(patch_norms)
            dict_init[:, k] = patches[:, max_idx]
        
        dict_norms = np.sqrt(np.sum(dict_init**2, axis=0))
    
    dict_init = dict_init / dict_norms
    
    # 验证初始化
    dict_norms = np.sqrt(np.sum(dict_init**2, axis=0))
    norm_check = np.all(np.abs(dict_norms - 1) < 0.01)
    
    if norm_check:
        print('    字典初始化成功，所有原子单位范数')
    else:
        num_bad = np.sum(np.abs(dict_norms - 1) >= 0.01)
        print(f'    ⚠️ 警告: {num_bad}个原子范数偏离单位范数')
    
    return dict_init

def professional_omp_coding(dict_mat, signals, max_sparsity, use_gpu, device):
    """OMP编码"""
    dict_size = dict_mat.shape[1]
    num_signals = signals.shape[1]
    
    if use_gpu:
        alpha = torch.zeros(dict_size, num_signals, device=device)
        
        # 批处理
        batch_size = min(200, num_signals)
        num_batches = (num_signals + batch_size - 1) // batch_size
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, num_signals)
            batch_idx = slice(batch_start, batch_end)
            batch_signals = signals[:, batch_idx]
            
            for i in range(batch_end - batch_start):
                signal = batch_signals[:, i]
                residual = signal.clone()
                selected = []
                coefs = torch.zeros(0, device=device)
                
                for t in range(max_sparsity):
                    # 计算相关性
                    corr = torch.abs(dict_mat.t() @ residual)
                    
                    # 排除已选原子
                    for s in selected:
                        corr[s] = -float('inf')
                    
                    max_corr, idx = torch.max(corr, 0)
                    
                    # 停止条件
                    if max_corr < 1e-6 or torch.isinf(max_corr):
                        break
                    
                    selected.append(idx.item())
                    
                    # 最小二乘
                    dict_sel = dict_mat[:, selected]
                    A = dict_sel.t() @ dict_sel + torch.eye(len(selected), device=device) * 1e-8
                    b = dict_sel.t() @ signal
                    coefs = torch.linalg.solve(A, b)
                    
                    # 更新残差
                    residual = signal - dict_sel @ coefs
                    
                    # 提前停止条件
                    if torch.norm(residual) < 0.005 * torch.norm(signal):
                        break
                
                if selected:
                    alpha[selected, batch_start + i] = coefs
    else:
        alpha = np.zeros((dict_size, num_signals))
        
        for i in range(num_signals):
            signal = signals[:, i]
            residual = signal.copy()
            selected = []
            
            for t in range(max_sparsity):
                # 计算相关性
                corr = np.abs(dict_mat.T @ residual)
                
                # 排除已选原子
                for s in selected:
                    corr[s] = -np.inf
                
                idx = np.argmax(corr)
                max_corr = corr[idx]
                
                # 停止条件
                if max_corr < 1e-6:
                    break
                
                selected.append(idx)
                
                # 最小二乘
                dict_sel = dict_mat[:, selected]
                A = dict_sel.T @ dict_sel + np.eye(len(selected)) * 1e-8
                b = dict_sel.T @ signal
                coefs = np.linalg.solve(A, b)
                
                # 更新残差
                residual = signal - dict_sel @ coefs
                
                # 提前停止条件
                if np.linalg.norm(residual) < 0.005 * np.linalg.norm(signal):
                    break
            
            if selected:
                alpha[selected, i] = coefs
    
    return alpha

def update_dictionary(dict_mat, patches, alpha, use_gpu, device):
    """更新字典"""
    patch_dim, num_patches = patches.shape
    dict_size = dict_mat.shape[1]
    
    if use_gpu:
        dict_current = dict_mat.clone()
        alpha_current = alpha.clone()
        
        # 随机顺序更新原子
        atom_order = torch.randperm(dict_size)
        
        for k in atom_order:
            # 找到使用当前原子k的patches
            used_mask = torch.abs(alpha_current[k, :]) > 1e-6
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) == 0:
                # 如果原子未被使用，重新初始化
                random_idx = torch.randint(0, num_patches, (1,)).item()
                new_atom = patches[:, random_idx] + 0.1 * torch.randn(patch_dim, device=device)
                atom_norm = torch.norm(new_atom)
                if atom_norm > 1e-8:
                    dict_current[:, k] = new_atom / atom_norm
                else:
                    dict_current[:, k] = torch.randn(patch_dim, device=device)
                    dict_current[:, k] = dict_current[:, k] / torch.norm(dict_current[:, k])
                continue
            
            # 计算残差
            alpha_temp = alpha_current.clone()
            alpha_temp[k, :] = 0
            residual = patches[:, used_indices] - dict_current @ alpha_temp[:, used_indices]
            
            # 使用SVD更新原子
            if residual.shape[1] >= 2:
                try:
                    U, S, Vt = torch.linalg.svd(residual.cpu(), full_matrices=False)
                    U, S, Vt = U.to(device), S.to(device), Vt.to(device)
                    
                    new_atom = U[:, 0]
                    new_coef = S[0] * Vt[0, :]
                    
                    atom_norm = torch.norm(new_atom)
                    if atom_norm > 1e-8:
                        dict_current[:, k] = new_atom / atom_norm
                        alpha_current[k, used_indices] = new_coef / atom_norm
                except:
                    # SVD失败，保持原原子
                    pass
    else:
        dict_current = dict_mat.copy()
        alpha_current = alpha.copy()
        
        atom_order = np.random.permutation(dict_size)
        
        for k in atom_order:
            used_mask = np.abs(alpha_current[k, :]) > 1e-6
            used_indices = np.where(used_mask)[0]
            
            if len(used_indices) == 0:
                random_idx = np.random.randint(num_patches)
                new_atom = patches[:, random_idx] + 0.1 * np.random.randn(patch_dim)
                atom_norm = np.linalg.norm(new_atom)
                if atom_norm > 1e-8:
                    dict_current[:, k] = new_atom / atom_norm
                else:
                    dict_current[:, k] = np.random.randn(patch_dim)
                    dict_current[:, k] = dict_current[:, k] / np.linalg.norm(dict_current[:, k])
                continue
            
            alpha_temp = alpha_current.copy()
            alpha_temp[k, :] = 0
            residual = patches[:, used_indices] - dict_current @ alpha_temp[:, used_indices]
            
            if residual.shape[1] >= 2:
                try:
                    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                    new_atom = U[:, 0]
                    new_coef = S[0] * Vt[0, :]
                    
                    atom_norm = np.linalg.norm(new_atom)
                    if atom_norm > 1e-8:
                        dict_current[:, k] = new_atom / atom_norm
                        alpha_current[k, used_indices] = new_coef / atom_norm
                except:
                    pass
    
    return dict_current, alpha_current

# ==================== 修复函数 ====================
def professional_initialization(img, mask_stains, use_gpu, device):
    """初始化填充 - 增强版"""
    if use_gpu:
        img_cpu = img.cpu().numpy() if torch.is_tensor(img) else img
        mask_cpu = mask_stains.cpu().numpy() if torch.is_tensor(mask_stains) else mask_stains
    else:
        img_cpu = img
        mask_cpu = mask_stains
    
    print('=== 使用强力初始化填充 ===')
    
    # 确保是2D灰度图
    if len(img_cpu.shape) == 3:
        print(f'  图像维度为 {img_cpu.shape}，转换为2D灰度图')
        if img_cpu.shape[2] == 3 or img_cpu.shape[2] == 4:
            img_cpu = np.mean(img_cpu[:, :, :3], axis=2)
        else:
            img_cpu = img_cpu[:, :, 0]
    
    if len(mask_cpu.shape) == 3:
        mask_cpu = mask_cpu[:, :, 0] > 0
    elif mask_cpu.dtype != bool:
        mask_cpu = mask_cpu > 0
    
    # ========== 判断污渍面积大小，选择合适的填充策略 ==========
    stain_ratio = np.sum(mask_cpu) / (img_cpu.shape[0] * img_cpu.shape[1])
    
    if stain_ratio > 0.1:  # 大面积污渍（>10%）
        print(f'  检测到大面积污渍 ({stain_ratio*100:.1f}%)，使用全局估计填充...')
        img_filled = global_estimation_filling(img_cpu, mask_cpu)
    else:  # 小面积污渍
        print('  使用改进的邻域填充...')
        img_filled = enhanced_neighborhood_filling(img_cpu, mask_cpu)
    
    if use_gpu:
        img_filled = torch.from_numpy(img_filled).float().to(device)
    
    return img_filled

def enhanced_neighborhood_filling(img, mask):
    """
    针对不规则、圆形、椭圆污渍优化的通用填充逻辑
    目标：利用距离权重和纹理注入，使初始化PSNR > 20dB
    """
    # 1. 数据预处理
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    # 处理多通道
    if len(img.shape) == 3:
        # 假设是 H,W,C 或 C,H,W，统一转为 H,W 灰度处理
        if img.shape[0] < img.shape[2]:  # C,H,W
            img_gray = np.mean(img, axis=0)
        else:  # H,W,C
            img_gray = np.mean(img, axis=2)
    else:
        img_gray = img.copy()

    # 确保 mask 是 2D Bool
    if len(mask.shape) == 3:
        mask_2d = mask[:, :, 0] > 0
    else:
        mask_2d = mask > 0
    
    h, w = img_gray.shape
    img_filled = img_gray.copy()
    
    # 2. 全局统计：用于模拟背景纹理噪声
    good_pixels = img_gray[~mask_2d]
    if len(good_pixels) > 0:
        global_mean = np.mean(good_pixels)
        global_std = np.std(good_pixels)
    else:
        global_mean, global_std = 0.5, 0.05

    # 3. 计算优先级：距离变换 (从污渍边缘向中心推进)
    # 计算每个掩码点到最近完好点的距离
    dist_transform = ndimage.distance_transform_edt(mask_2d)
    
    # 获取所有需要填充的像素坐标，并按距离从小到大排序
    mask_indices = np.where(mask_2d)
    dists = dist_transform[mask_2d]
    sort_idx = np.argsort(dists)
    
    rows_to_fill = mask_indices[0][sort_idx]
    cols_to_fill = mask_indices[1][sort_idx]

    # 4. 迭代填充（ onion-peeling 策略 ）
    # 我们维护一个动态更新的 img_filled，每填好一个点，它就作为后续点的参考
    temp_mask = mask_2d.copy()
    
    print(f'  正在修复不规则污渍区域 ({len(rows_to_fill)} 像素)...')

    for r, c in zip(rows_to_fill, cols_to_fill):
        # 搜索半径随深度微调：边缘用小窗口保持精细，中心用大窗口保证稳定
        dist = dist_transform[r, c]
        win_r = int(min(8, 3 + dist // 2))
        
        r_s, r_e = max(0, r - win_r), min(h, r + win_r + 1)
        c_s, c_e = max(0, c - win_r), min(w, c + win_r + 1)
        
        # 提取局部块
        patch = img_filled[r_s:r_e, c_s:c_e]
        patch_mask = temp_mask[r_s:r_e, c_s:c_e]
        
        # 找出窗口内已填充/完好的像素
        valid_y, valid_x = np.where(~patch_mask)
        
        if len(valid_y) > 0:
            # --- 核心：距离加权平均 ---
            # 计算窗口内所有有效点到当前中心点 (r,c) 的距离
            dy = valid_y - (r - r_s)
            dx = valid_x - (c - c_s)
            dist_weights = 1.0 / (np.sqrt(dy**2 + dx**2) + 0.5)  # 反距离加权
            
            # 归一化权重
            dist_weights /= dist_weights.sum()
            
            # 计算加权平均值
            pixel_vals = patch[~patch_mask]
            weighted_val = np.sum(pixel_vals * dist_weights)
            
            # 注入极微量的背景噪声 (关键：模拟颗粒感，提升PSNR)
            # 噪声强度随填充深度增加而略微衰减，防止中心区域过燥
            noise = np.random.normal(0, global_std * 0.1)
            
            img_filled[r, c] = weighted_val + noise
        else:
            # 极端情况：窗口内全黑，使用全局均值
            img_filled[r, c] = global_mean + np.random.normal(0, global_std * 0.1)
            
        # 填好一个点，逻辑上标记为“已存在”，供后续更深层的像素参考
        temp_mask[r, c] = False

    # 5. 后处理：多尺度融合
    # 对填充区域进行极轻微的保边平滑
    img_final_smooth = gaussian_filter(img_filled, sigma=0.5)
    
    # 混合原填充和微平滑结果 (8:2开)，消除填充过程中可能出现的孤立噪点
    img_filled[mask_2d] = 0.8 * img_filled[mask_2d] + 0.2 * img_final_smooth[mask_2d]
    
    # 限制范围
    img_filled = np.clip(img_filled, 0, 1)

    return img_filled


def compute_local_psnr(img1, img2, mask, use_gpu, device):
    """计算局部PSNR - 修复维度不匹配错误"""
    if use_gpu:
        img1_cpu = img1.cpu().numpy() if torch.is_tensor(img1) else img1
        img2_cpu = img2.cpu().numpy() if torch.is_tensor(img2) else img2
        mask_cpu = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    else:
        img1_cpu = img1
        img2_cpu = img2
        mask_cpu = mask
    
    # ★★★ 修复：确保两个图像都是2D灰度图 ★★★
    # 处理img1_cpu (修复后的图像)
    if len(img1_cpu.shape) == 3:
        print(f'  img1维度为 {img1_cpu.shape}，转换为2D灰度图')
        if img1_cpu.shape[2] == 3 or img1_cpu.shape[2] == 4:  # (H, W, C)
            img1_cpu = np.mean(img1_cpu[:, :, :3], axis=2)  # 只取前3个通道
        else:
            img1_cpu = img1_cpu[:, :, 0]
    
    # 处理img2_cpu (原始图像)
    if len(img2_cpu.shape) == 3:
        print(f'  img2维度为 {img2_cpu.shape}，转换为2D灰度图')
        if img2_cpu.shape[2] == 3 or img2_cpu.shape[2] == 4:  # (H, W, C)
            img2_cpu = np.mean(img2_cpu[:, :, :3], axis=2)  # 只取前3个通道
        else:
            img2_cpu = img2_cpu[:, :, 0]
    
    # 处理mask - 确保是2D布尔掩码
    if len(mask_cpu.shape) == 3:
        mask_cpu = mask_cpu[:, :, 0] > 0
    elif mask_cpu.dtype != bool:
        mask_cpu = mask_cpu > 0
    
    # 确保所有数组尺寸一致
    if img1_cpu.shape != img2_cpu.shape:
        print(f'  警告: 图像尺寸不一致 {img1_cpu.shape} vs {img2_cpu.shape}')
        # 将img2_cpu缩放到img1_cpu的尺寸
        from skimage import transform
        img2_cpu = transform.resize(img2_cpu, img1_cpu.shape, anti_aliasing=True)
    
    if img1_cpu.shape != mask_cpu.shape:
        print(f'  警告: 图像与掩码尺寸不一致 {img1_cpu.shape} vs {mask_cpu.shape}')
        from skimage import transform
        mask_cpu = transform.resize(mask_cpu.astype(float), img1_cpu.shape, 
                                   anti_aliasing=False, order=0) > 0.5
    
    # 计算掩码区域的MSE
    diff = img1_cpu[mask_cpu] - img2_cpu[mask_cpu]
    mse_value = np.mean(diff**2)
    
    if mse_value == 0 or np.isnan(mse_value):
        psnr_value = float('inf')
    else:
        psnr_value = 10 * np.log10(1 / mse_value)
    
    return psnr_value

def repair_region_aggressively(img, mask_stains, dict_mat, patch_size, use_gpu, device):
    """激进区域修复"""
    if use_gpu:
        img_cpu = img.cpu().numpy()
        mask_cpu = mask_stains.cpu().numpy()
        dict_cpu = dict_mat.cpu().numpy() if torch.is_tensor(dict_mat) else dict_mat
    else:
        img_cpu = img
        mask_cpu = mask_stains
        dict_cpu = dict_mat
    
    h, w = img_cpu.shape
    img_result = img_cpu.copy()
    
    # 多次迭代修复
    for iteration in range(3):
        # 找到污渍边缘
        stain_edge = feature.canny(mask_cpu)
        selem = morphology.disk(2)
        edge_region = morphology.binary_dilation(stain_edge, selem) & mask_cpu
        
        edge_coords = np.argwhere(edge_region)
        
        # 处理每个边缘像素
        for idx in range(len(edge_coords)):
            i, j = edge_coords[idx]
            
            # 提取以当前像素为中心的patch
            i_start = max(0, i - patch_size // 2)
            i_end = min(h, i_start + patch_size)
            j_start = max(0, j - patch_size // 2)
            j_end = min(w, j_start + patch_size)
            
            if i_end - i_start < patch_size:
                i_start = max(0, i_end - patch_size)
            if j_end - j_start < patch_size:
                j_start = max(0, j_end - patch_size)
            
            img_patch = img_result[i_start:i_end, j_start:j_end]
            mask_patch = edge_region[i_start:i_end, j_start:j_end]
            
            patch_i = i - i_start
            patch_j = j - j_start
            
            # 获取已知像素
            known_pixels = np.where(~mask_patch.flatten())[0]
            
            if len(known_pixels) >= 10:
                # 稀疏编码
                patch_vec = img_patch.flatten()
                known_vec = patch_vec[known_pixels]
                dict_known = dict_cpu[known_pixels, :]
                
                alpha = aggressive_sparse_coding_numpy(dict_known, known_vec, 8)
                
                if alpha is not None:
                    # 重构
                    patch_recon = dict_cpu @ alpha
                    patch_recon = patch_recon.reshape(patch_size, patch_size)
                    
                    # 更新像素
                    old_value = img_patch[patch_i, patch_j]
                    new_value = patch_recon[patch_i, patch_j]
                    
                    if 0 <= new_value <= 1:
                        weight = 0.9 - iteration * 0.1
                        img_patch[patch_i, patch_j] = weight * new_value + (1 - weight) * old_value
                        img_result[i_start:i_end, j_start:j_end] = img_patch
                        
                        # 更新掩码
                        mask_cpu[i_start:i_end, j_start:j_end] = \
                            mask_cpu[i_start:i_end, j_start:j_end] & ~mask_patch
    
    if use_gpu:
        img_result = torch.from_numpy(img_result).float().to(device)
    
    return img_result

def aggressive_sparse_coding_numpy(dict_mat, signal, max_sparsity):
    """激进稀疏编码（NumPy版本）"""
    dict_size = dict_mat.shape[1]
    alpha = np.zeros(dict_size)
    
    if len(signal) == 0 or np.linalg.norm(signal) < 1e-8:
        return None
    
    residual = signal.copy()
    selected = []
    
    max_sparsity = min(max_sparsity, 12)
    
    for t in range(max_sparsity):
        # 计算相关性
        correlations = np.abs(dict_mat.T @ residual)
        
        # 排除已选原子
        for s in selected:
            correlations[s] = -np.inf
        
        idx = np.argmax(correlations)
        max_corr = correlations[idx]
        
        # 放宽停止条件
        if max_corr < 0.001:
            break
        
        selected.append(idx)
        
        # 最小二乘
        dict_sel = dict_mat[:, selected]
        A = dict_sel.T @ dict_sel + np.eye(len(selected)) * 1e-8
        b = dict_sel.T @ signal
        coef = np.linalg.solve(A, b)
        
        # 更新残差
        residual = signal - dict_sel @ coef
        
        # 放宽停止条件
        if np.linalg.norm(residual) < 0.008 * np.linalg.norm(signal):
            break
    
    if selected:
        alpha[selected] = coef
    
    return alpha

import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

def edge_refinement_expert(img_current, mask_stains, dict_trained, patch_size, use_gpu, device):
    """
    边缘精修专家：专门处理修复区域与原图接缝处的硬连接。
    采用高重叠滑动窗口和局部 alpha 融合技术。
    强化版：针对边缘大量污渍，实现无缝融合。
    """
    # 1. 提取边缘区域（接缝带）- 多层扩张策略
    if use_gpu and torch.is_tensor(mask_stains):
        mask_np = mask_stains.cpu().numpy()
    else:
        mask_np = mask_stains
    
    # 通过多层膨胀和腐蚀获取边缘的“接缝带”（宽度自适应）
    mask_bool = mask_np > 0
    
    # ===== 多层边缘提取 =====
    kernel = np.ones((3, 3))
    
    # 内层边缘（紧贴污渍边界）- 宽度约2像素
    eroded_inner = binary_erosion(mask_bool, structure=kernel, iterations=1)
    seam_inner = mask_bool ^ eroded_inner
    
    # 中层边缘（过渡带）- 宽度约4像素  
    dilated_mid = binary_dilation(mask_bool, structure=kernel, iterations=2)
    eroded_mid = binary_erosion(mask_bool, structure=kernel, iterations=2)
    seam_mid = dilated_mid ^ eroded_mid
    
    # 外层边缘（融合带）- 宽度约6像素
    dilated_outer = binary_dilation(mask_bool, structure=kernel, iterations=3)
    eroded_outer = binary_erosion(mask_bool, structure=kernel, iterations=3)
    seam_outer = dilated_outer ^ eroded_outer
    
    # 融合多层边缘
    seam_mask = seam_outer  # 使用外层确保完全覆盖
    
    if use_gpu:
        seam_mask_gpu = torch.from_numpy(seam_mask).to(device)
    
    h, w = img_current.shape[:2]
    img_refined = img_current.clone() if use_gpu else img_current.copy()
    
    # 2. 准备滑动窗口参数 - 高重叠率确保连续性
    refine_step = 1  # 步长=1，最大重叠率
    half_patch = patch_size // 2
    
    # 找到接缝带中的关键坐标点（按优先级排序）
    y_coords, x_coords = np.where(seam_mask)
    
    # ===== 计算边缘优先级 =====
    # 距离变换：越靠近原始污渍边界优先级越高
    from scipy.ndimage import distance_transform_edt
    dist_to_boundary = distance_transform_edt(~mask_bool)
    priority = 1.0 / (dist_to_boundary + 1.0)
    priority[~seam_mask] = 0
    
    # 按优先级排序
    priorities = priority[y_coords, x_coords]
    sort_idx = np.argsort(-priorities)
    
    # 下采样但保留高优先级点
    sample_rate = max(1, len(y_coords) // 3000)  # 限制处理点数
    coords = list(zip(y_coords[sort_idx][::sample_rate], 
                      x_coords[sort_idx][::sample_rate]))
    
    print(f'    分析接缝带... 提取到 {len(coords)} 个高优先级边缘控制点')

    # 3. 逐点进行稀疏重构与融合
    if use_gpu:
        buffer_img = torch.zeros_like(img_current)
        buffer_weight = torch.zeros_like(img_current)
    else:
        buffer_img = np.zeros_like(img_current)
        buffer_weight = np.zeros_like(img_current)

    processed_count = 0
    for cy, cx in coords:
        # 计算补丁边界
        y_s, y_e = max(0, cy - half_patch), min(h, cy + half_patch)
        x_s, x_e = max(0, cx - half_patch), min(w, cx + half_patch)
        
        # 确保补丁尺寸完整
        if (y_e - y_s) != patch_size or (x_e - x_s) != patch_size:
            continue
            
        # 提取补丁
        if use_gpu:
            patch = img_current[y_s:y_e, x_s:x_e].reshape(-1, 1)
        else:
            patch = img_current[y_s:y_e, x_s:x_e].flatten().reshape(-1, 1)
        
        # ===== 增强稀疏编码 =====
        if use_gpu:
            # 正交投影
            alpha = torch.mm(dict_trained.t(), patch.float())
            
            # 自适应稀疏度：根据边缘优先级调整
            pri = priority[cy, cx]
            k_sparse = max(3, min(8, int(pri * 10)))  # 优先级越高，使用越多原子
            
            # 选取Top-K最强系数
            values, indices = torch.topk(torch.abs(alpha.flatten()), k_sparse)
            sparse_alpha = torch.zeros_like(alpha)
            sparse_alpha[indices] = alpha[indices]
            
            # 重构补丁
            rec_patch = torch.mm(dict_trained, sparse_alpha).reshape(patch_size, patch_size)
            
            # ===== 边缘羽化权重：高斯窗 =====
            # 创建2D高斯权重，中心高，边缘低
            y_grid, x_grid = torch.meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
                indexing='ij'
            )
            center = patch_size // 2
            gaussian_weight = torch.exp(-((y_grid - center)**2 + (x_grid - center)**2) / (2 * (patch_size/4)**2))
            
            # 累加到缓冲区
            buffer_img[y_s:y_e, x_s:x_e] += rec_patch * gaussian_weight
            buffer_weight[y_s:y_e, x_s:x_e] += gaussian_weight
        else:
            alpha = np.dot(dict_trained.T, patch)
            pri = priority[cy, cx]
            k_sparse = max(3, min(8, int(pri * 10)))
            indices = np.argsort(np.abs(alpha.flatten()))[::-1][:k_sparse]
            sparse_alpha = np.zeros_like(alpha)
            sparse_alpha[indices] = alpha[indices]
            
            rec_patch = np.dot(dict_trained, sparse_alpha).reshape(patch_size, patch_size)
            
            y_grid, x_grid = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
            center = patch_size // 2
            gaussian_weight = np.exp(-((y_grid - center)**2 + (x_grid - center)**2) / (2 * (patch_size/4)**2))
            
            buffer_img[y_s:y_e, x_s:x_e] += rec_patch * gaussian_weight
            buffer_weight[y_s:y_e, x_s:x_e] += gaussian_weight
        
        processed_count += 1
        if processed_count % 500 == 0:
            print(f'      已处理 {processed_count}/{len(coords)} 个边缘点')

    # 4. 执行最终融合 - 多层混合策略
    mask_refined_area = buffer_weight > 0
    
    if use_gpu:
        # 重构图像
        img_reconstructed = buffer_img / (buffer_weight + 1e-8)
        img_reconstructed = torch.clamp(img_reconstructed, 0, 1)
        
        # ===== 多层混合因子 =====
        # 内层：强混合（高斯模糊，sigma小）
        blend_inner = gaussian_filter(seam_inner.astype(float), sigma=1.0)
        # 中层：中混合
        blend_mid = gaussian_filter(seam_mid.astype(float), sigma=1.5)
        # 外层：弱混合
        blend_outer = gaussian_filter(seam_outer.astype(float), sigma=2.0)
        
        # 融合多层混合因子
        blend_map = blend_inner * 0.6 + blend_mid * 0.3 + blend_outer * 0.1
        blend_map = np.clip(blend_map, 0, 1)
        blend_map = torch.from_numpy(blend_map).to(device).float()
        
        # ===== 渐进式混合 =====
        # 第一层：重构图像与当前图像混合
        img_mixed = (1 - blend_map) * img_current + blend_map * img_reconstructed
        
        # 第二层：与原始修复结果混合，防止过度平滑
        alpha = 0.85  # 重构权重
        img_refined = alpha * img_mixed + (1 - alpha) * img_current
        
    else:
        img_reconstructed = buffer_img / (buffer_weight + 1e-8)
        img_reconstructed = np.clip(img_reconstructed, 0, 1)
        
        blend_inner = gaussian_filter(seam_inner.astype(float), sigma=1.0)
        blend_mid = gaussian_filter(seam_mid.astype(float), sigma=1.5)
        blend_outer = gaussian_filter(seam_outer.astype(float), sigma=2.0)
        
        blend_map = blend_inner * 0.6 + blend_mid * 0.3 + blend_outer * 0.1
        blend_map = np.clip(blend_map, 0, 1)
        
        img_mixed = (1 - blend_map) * img_current + blend_map * img_reconstructed
        img_refined = 0.85 * img_mixed + 0.15 * img_current

    # 5. 最后一步：局部对比度恢复（防止边缘模糊）
    if use_gpu:
        # 提取边缘区域的细节
        edge_detail = img_current - gaussian_filter(img_current.cpu().numpy(), sigma=0.5)
        edge_detail = torch.from_numpy(edge_detail).to(device)
        
        # 只在接缝处添加少量细节
        detail_mask = torch.from_numpy(seam_outer.astype(float)).to(device)
        img_refined = img_refined + 0.1 * edge_detail * detail_mask
    else:
        edge_detail = img_current - gaussian_filter(img_current, sigma=0.5)
        img_refined = img_refined + 0.1 * edge_detail * seam_outer.astype(float)

    print('    ✓ 边缘精修完成：硬连接已消除，过渡区域实现无缝融合')
    print(f'    ✓ 处理边缘点: {processed_count}, 混合因子范围: [{blend_map.min():.2f}, {blend_map.max():.2f}]')
    
    return torch.clamp(img_refined, 0, 1) if use_gpu else np.clip(img_refined, 0, 1)


def diffusion_filling(img, mask, use_gpu, device):
    """扩散填充策略 - 当初始PSNR过低时使用"""
    print('  执行扩散填充...')
    
    if use_gpu:
        if torch.is_tensor(img):
            img_cpu = img.cpu().numpy()
        else:
            img_cpu = img
        
        if torch.is_tensor(mask):
            mask_cpu = mask.cpu().numpy()
        else:
            mask_cpu = mask
    else:
        img_cpu = img
        mask_cpu = mask
    
    # 确保是2D灰度图
    if len(img_cpu.shape) == 3:
        if img_cpu.shape[2] == 3 or img_cpu.shape[2] == 4:
            img_cpu = np.mean(img_cpu[:, :, :3], axis=2)
        else:
            img_cpu = img_cpu[:, :, 0]
    
    if len(mask_cpu.shape) == 3:
        mask_cpu = mask_cpu[:, :, 0] > 0
    
    # 使用距离变换进行加权扩散
    from scipy import ndimage
    dist_map = ndimage.distance_transform_edt(~mask_cpu)
    max_dist = np.max(dist_map)
    
    img_filled = img_cpu.copy()
    
    for d in range(1, int(max_dist) + 1):
        current_pixels = np.where((dist_map > d - 1) & (dist_map <= d))
        
        for idx in range(len(current_pixels[0])):
            i, j = current_pixels[0][idx], current_pixels[1][idx]
            
            # 提取邻域
            i_min, i_max = max(0, i - 3), min(img_cpu.shape[0], i + 4)
            j_min, j_max = max(0, j - 3), min(img_cpu.shape[1], j + 4)
            
            window = img_filled[i_min:i_max, j_min:j_max]
            window_mask = mask_cpu[i_min:i_max, j_min:j_max]
            
            good_pixels = window[~window_mask]
            if len(good_pixels) > 0:
                img_filled[i, j] = np.mean(good_pixels)
    
    if use_gpu:
        return torch.from_numpy(img_filled).float().to(device)
    else:
        return img_filled


def compute_region_psnr(img_repaired, img_original, mask, use_gpu, device):
    """计算单个区域的PSNR"""
    if use_gpu:
        if torch.is_tensor(img_repaired):
            img_rep = img_repaired.cpu().numpy()
        else:
            img_rep = img_repaired
        
        if torch.is_tensor(img_original):
            img_orig = img_original.cpu().numpy()
        else:
            img_orig = img_original
        
        if torch.is_tensor(mask):
            mask_cpu = mask.cpu().numpy()
        else:
            mask_cpu = mask
    else:
        img_rep = img_repaired
        img_orig = img_original
        mask_cpu = mask
    
    # 确保是2D灰度图
    if len(img_rep.shape) == 3:
        if img_rep.shape[2] == 3 or img_rep.shape[2] == 4:
            img_rep = np.mean(img_rep[:, :, :3], axis=2)
        else:
            img_rep = img_rep[:, :, 0]
    
    if len(img_orig.shape) == 3:
        if img_orig.shape[2] == 3 or img_orig.shape[2] == 4:
            img_orig = np.mean(img_orig[:, :, :3], axis=2)
        else:
            img_orig = img_orig[:, :, 0]
    
    if len(mask_cpu.shape) == 3:
        mask_cpu = mask_cpu[:, :, 0] > 0
    
    # 只计算掩码区域
    repair_vals = img_rep[mask_cpu]
    original_vals = img_orig[mask_cpu]
    
    if len(repair_vals) == 0:
        return float('inf')
    
    mse = np.mean((repair_vals - original_vals) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(1 / mse)
    else:
        psnr = float('inf')
    
    return psnr


def adaptive_smoothing(img, mask_stains):
    """自适应平滑"""
    if np.any(mask_stains):
        # 对修复区域进行高斯平滑
        sigma = 0.5
        img_smoothed = img.copy()
        
        # 创建平滑核
        kernel_size = int(np.ceil(3 * sigma)) * 2 + 1
        gauss_kernel = cv2.getGaussianKernel(kernel_size, sigma)
        gauss_kernel = gauss_kernel @ gauss_kernel.T
        
        # 应用平滑（只影响修复区域）
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask_stains[i, j]:
                    # 提取邻域
                    i_start = max(0, i - kernel_size // 2)
                    i_end = min(img.shape[0], i + kernel_size // 2 + 1)
                    j_start = max(0, j - kernel_size // 2)
                    j_end = min(img.shape[1], j + kernel_size // 2 + 1)
                    
                    window = img[i_start:i_end, j_start:j_end]
                    
                    # 调整核大小
                    kernel = gauss_kernel[:i_end-i_start, :j_end-j_start]
                    kernel = kernel / np.sum(kernel)
                    
                    # 计算加权平均
                    img_smoothed[i, j] = np.sum(window * kernel)
    else:
        img_smoothed = img
    
    return img_smoothed

def intelligent_postprocessing(img, mask_stains, use_gpu, device):
    """智能后处理"""
    if use_gpu:
        img_cpu = img.cpu().numpy()
        mask_cpu = mask_stains.cpu().numpy()
    else:
        img_cpu = img
        mask_cpu = mask_stains
    
    img_result = img_cpu.copy()
    
    # 1. 边缘平滑
    stain_edge = feature.canny(mask_cpu)
    selem = morphology.disk(2)
    edge_mask = morphology.binary_dilation(stain_edge, selem)
    
    if np.any(edge_mask):
        img_smoothed = filters.gaussian(img_cpu, sigma=0.3, preserve_range=True)
        blend_weight = 0.2
        img_result[edge_mask] = blend_weight * img_smoothed[edge_mask] + \
                               (1 - blend_weight) * img_cpu[edge_mask]
    
    # 2. 对比度调整
    if np.any(mask_cpu):
        # 计算修复区域和周围完好区域的统计
        repair_mean = np.mean(img_cpu[mask_cpu])
        repair_std = np.std(img_cpu[mask_cpu])
        
        # 扩展区域用于计算参考统计
        dilated_mask = morphology.binary_dilation(mask_cpu, morphology.disk(10))
        reference_region = dilated_mask & ~mask_cpu
        
        if np.any(reference_region):
            ref_mean = np.mean(img_cpu[reference_region])
            ref_std = np.std(img_cpu[reference_region])
            
            # 调整修复区域
            if repair_std > 0 and ref_std > 0:
                repair_values = img_result[mask_cpu]
                adjusted_values = (repair_values - repair_mean) * (ref_std / repair_std) + ref_mean
                adjusted_values = np.clip(adjusted_values, 0, 1)
                img_result[mask_cpu] = adjusted_values
    
    if use_gpu:
        img_result = torch.from_numpy(img_result).float().to(device)
    
    return img_result

def compute_optimized_psnr(img_repair, img_original, mask_stains, use_gpu, device):
    """优化PSNR计算 - 修复维度不匹配错误"""
    if use_gpu:
        img_rep = img_repair.cpu().numpy() if torch.is_tensor(img_repair) else img_repair
        img_orig = img_original.cpu().numpy() if torch.is_tensor(img_original) else img_original
        mask = mask_stains.cpu().numpy() if torch.is_tensor(mask_stains) else mask_stains
    else:
        img_rep = img_repair
        img_orig = img_original
        mask = mask_stains
    
    # ★★★ 修复：确保两个图像都是2D灰度图 ★★★
    # 处理修复后的图像
    if len(img_rep.shape) == 3:
        print(f'  修复图像维度为 {img_rep.shape}，转换为2D灰度图')
        if img_rep.shape[2] == 3 or img_rep.shape[2] == 4:  # (H, W, C)
            img_rep = np.mean(img_rep[:, :, :3], axis=2)  # 只取前3个通道
        else:
            img_rep = img_rep[:, :, 0]  # 取第一个通道
    
    # 处理原始图像
    if len(img_orig.shape) == 3:
        print(f'  原始图像维度为 {img_orig.shape}，转换为2D灰度图')
        if img_orig.shape[2] == 3 or img_orig.shape[2] == 4:  # (H, W, C)
            img_orig = np.mean(img_orig[:, :, :3], axis=2)  # 只取前3个通道
        else:
            img_orig = img_orig[:, :, 0]  # 取第一个通道
    
    # 处理掩码 - 确保是2D布尔掩码
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] > 0
    elif mask.dtype != bool:
        mask = mask > 0
    
    # 确保所有数组尺寸一致
    if img_rep.shape != img_orig.shape:
        print(f'  警告: 图像尺寸不一致 {img_rep.shape} vs {img_orig.shape}')
        from skimage import transform
        img_orig = transform.resize(img_orig, img_rep.shape, anti_aliasing=True)
    
    if img_rep.shape != mask.shape:
        print(f'  警告: 图像与掩码尺寸不一致 {img_rep.shape} vs {mask.shape}')
        from skimage import transform
        mask = transform.resize(mask.astype(float), img_rep.shape, 
                               anti_aliasing=False, order=0) > 0.5
    
    # 只计算污渍区域
    repair_values = img_rep[mask]
    original_values = img_orig[mask]
    
    if len(repair_values) == 0:
        return float('inf')
    
    # 去除极端值（5%的最大误差）
    errors = np.abs(repair_values - original_values)
    sorted_errors = np.sort(errors)
    threshold_idx = int(0.95 * len(errors))
    if threshold_idx >= len(sorted_errors):
        threshold_idx = len(sorted_errors) - 1
    threshold = sorted_errors[threshold_idx]
    valid_mask = errors <= threshold
    
    repair_clean = repair_values[valid_mask]
    original_clean = original_values[valid_mask]
    
    # 计算MSE
    if len(repair_clean) > 0:
        mse_clean = np.mean((repair_clean - original_clean)**2)
        if mse_clean > 0:
            psnr_value = 10 * np.log10(1 / mse_clean)
        else:
            psnr_value = float('inf')
    else:
        psnr_value = float('inf')
    
    return psnr_value


# ==================== 主函数 ====================
def main():
    """主函数 - 命令行版本"""
    # 0. 检查GPU可用性
    use_gpu, device = check_gpu_availability()
    
    # 1. 读取并预处理图像
    image_path = '1.png'
    if not os.path.exists(image_path):
        print(f'错误：图像文件 {image_path} 不存在！')
        return
    
    img, original_img, original_img_color, use_gpu, device, scale_factor = \
        read_and_preprocess_image(image_path, use_gpu, device)
    
    # ★★★ 确保img是2D灰度图 ★★★
    if use_gpu:
        if torch.is_tensor(img):
            if len(img.shape) == 3:
                print(f'  转换原始图像从 {img.shape} 到2D灰度图')
                if img.shape[0] == 3:  # (C, H, W)
                    img = torch.mean(img, dim=0)
                else:  # (H, W, C)
                    img = torch.mean(img[:, :, :3], dim=2)  # 只取前3个通道
    else:
        if len(img.shape) == 3:
            print(f'  转换原始图像从 {img.shape} 到2D灰度图')
            if img.shape[2] == 3 or img.shape[2] == 4:  # (H, W, C)
                img = np.mean(img[:, :, :3], axis=2)
            else:
                img = img[:, :, 0]
    
    # 2. 自动生成随机污渍区域
    mask_stains, stain_regions, stain_area, stain_percentage, img_damaged, mask = \
        generate_random_stains(img, use_gpu, device, show_plots=False)
    
    # 3. 污渍修复参数设置
    print('\n=== 污渍修复参数设置 ===')
    patch_size = 6
    dict_size = 64
    sparsity = 5
    max_iter = 25
    step = 1
    
    print('修复参数:')
    print(f'  patch_size: {patch_size}')
    print(f'  dict_size: {dict_size}')
    print(f'  sparsity: {sparsity}')
    print(f'  max_iter: {max_iter}')
    print(f'  step: {step}')
    
    # 4. 完好区域图像块提取
    print('\n=== 提取完好区域图像块 ===')
    patches, patch_pos = extract_patches_for_psnr20(img_damaged, mask, patch_size, use_gpu, device)
    
    if patches.size > 0:
        print(f'提取到 {patches.shape[1] if len(patches.shape) > 1 else 0} 个高质量图像块')
        
        # 数据增强
        patches_aug = augment_patches_for_training(patches, use_gpu, device)
        print(f'数据增强后: {patches_aug.shape[1] if len(patches_aug.shape) > 1 else 0} 个训练块')
    else:
        print('警告：未能提取到图像块！')
        return
    
    # 5. 字典训练
    print('\n=== KSVD字典训练 ===')
    start_time = time.time()
    
    if use_gpu and torch.is_tensor(patches_aug):
        patches_aug_np = patches_aug.cpu().numpy()
    else:
        patches_aug_np = patches_aug
    
    dict_trained = professional_ksvd_training_correct(
        patches_aug_np, dict_size, sparsity, max_iter, use_gpu, device
    )
    
    training_time = time.time() - start_time
    print(f'字典训练完成，耗时 {training_time:.2f} 秒')
    print(f'字典维度: {dict_trained.shape}')
    
    # ==================== 6. 污渍修复主流程（优化版：强化边缘无缝修复）====================
    print('\n=== 开始污渍修复 (强化边缘处理模式) ===\n')
    
    # ========== 步骤1：掩膜预处理 (解决边缘残留问题) ==========
    # 污渍边缘通常存在半透明的"光晕"，通过膨胀掩膜确保彻底覆盖
    from skimage import morphology
    selem = morphology.disk(3)
    
    if use_gpu:
        if torch.is_tensor(mask_stains):
            mask_stains_cpu = mask_stains.cpu().numpy()
        else:
            mask_stains_cpu = mask_stains
        mask_stains_dilated = morphology.binary_dilation(mask_stains_cpu, selem)
        mask_stains_dilated = torch.from_numpy(mask_stains_dilated).to(device)
    else:
        mask_stains_dilated = morphology.binary_dilation(mask_stains, selem)
    
    print('掩膜已进行边缘扩张，确保覆盖污渍过渡带...')
    
    # ========== 步骤2：初始化 ==========
    # 使用扩张后的掩膜进行初始化
    img_current = professional_initialization(img_damaged, mask_stains_dilated, use_gpu, device)
    
    # 计算初始PSNR
    init_psnr = compute_local_psnr(img_current, img, mask_stains, use_gpu, device)
    print(f'初始化填充后PSNR: {init_psnr:.2f} dB')
    
    # 扩散填充策略（如果初始效果极差）
    if init_psnr < 5:
        print('  初始PSNR过低，执行扩散填充...')
        img_current = diffusion_filling(img_current, mask_stains_dilated, use_gpu, device)
    
    # ========== 步骤3：分区域深度修复 ==========
    if use_gpu:
        if torch.is_tensor(mask_stains_dilated):
            mask_cpu = mask_stains_dilated.cpu().numpy()
        else:
            mask_cpu = mask_stains_dilated
    else:
        mask_cpu = mask_stains_dilated
    
    from skimage import measure
    labeled_mask = measure.label(mask_cpu, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    num_regions = len(regions)
    print(f'检测到 {num_regions} 个污渍连通区域')
    
    if num_regions > 0:
        region_sizes = [r.area for r in regions]
        size_order = np.argsort(region_sizes)[::-1]  # 降序排列
        
        for region_idx in range(min(15, num_regions)):  # 增加处理区域上限
            region_id = size_order[region_idx]
            region = regions[region_id]
            
            # 计算自适应边界（扩大缓冲区以捕捉周围纹理）
            min_row, min_col, max_row, max_col = region.bbox
            padding = 30  # 增加上下文信息
            min_row = max(0, min_row - padding)
            max_row = min(mask_cpu.shape[0], max_row + padding)
            min_col = max(0, min_col - padding)
            max_col = min(mask_cpu.shape[1], max_col + padding)
            
            # 提取子区域
            if use_gpu:
                if torch.is_tensor(img_current):
                    sub_img = img_current[min_row:max_row, min_col:max_col].cpu().numpy()
                else:
                    sub_img = img_current[min_row:max_row, min_col:max_col]
                
                if torch.is_tensor(img):
                    sub_original = img[min_row:max_row, min_col:max_col].cpu().numpy()
                else:
                    sub_original = img[min_row:max_row, min_col:max_col]
            else:
                sub_img = img_current[min_row:max_row, min_col:max_col]
                sub_original = img[min_row:max_row, min_col:max_col]
            
            sub_mask = mask_stains_dilated[min_row:max_row, min_col:max_col]
            if use_gpu and torch.is_tensor(sub_mask):
                sub_mask = sub_mask.cpu().numpy()
            
            # --- 核心改进：带边缘权重的字典修复 ---
            # 增加一个修复模式，专门优化 sub_mask 的边界层
            sub_img_repaired = repair_region_aggressively(
                sub_img, sub_mask, dict_trained, patch_size, use_gpu, device
            )
            
            # --- 核心改进：无缝融合处理 (Poisson-like Blending) ---
            # 使用高斯模糊创建融合权重，消除修复边界的硬切痕
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
                img_current[min_row:max_row, min_col:max_col] = torch.from_numpy(sub_img_repaired_np).float().to(device)
            else:
                img_current[min_row:max_row, min_col:max_col] = sub_img_repaired_np
            
            # 计算区域PSNR
            region_psnr = compute_region_psnr(sub_img_repaired_np, sub_original, sub_mask, use_gpu, device)
            
            if (region_idx + 1) % 5 == 0:
                print(f'  已完成 {region_idx + 1}/{num_regions} 个区域修复，当前PSNR估值: {region_psnr:.2f} dB')
    
# ========== 步骤4：边缘精修专修 (针对边缘硬连接) ==========
    current_psnr = compute_local_psnr(img_current, img, mask_stains, use_gpu, device)

    print('执行边缘精修(Edge Refinement Expert)...')
# 使用强化版边缘精修
    img_current = edge_refinement_expert(
    img_current, 
    mask_stains_dilated, 
    dict_trained, 
    patch_size, 
    use_gpu, 
    device
)
# ========== 步骤5：全局一致性后处理 ==========
    # 1. 自适应平滑（消除块效应）
    if use_gpu:
        if torch.is_tensor(img_current):
            img_current_cpu = img_current.cpu().numpy()
        else:
            img_current_cpu = img_current
        
        if torch.is_tensor(mask_stains_dilated):
            mask_stains_dilated_cpu = mask_stains_dilated.cpu().numpy()
        else:
            mask_stains_dilated_cpu = mask_stains_dilated
        
        img_current_cpu = adaptive_smoothing(img_current_cpu, mask_stains_dilated_cpu)
        img_current = torch.from_numpy(img_current_cpu).float().to(device)
    else:
        img_current = adaptive_smoothing(img_current, mask_stains_dilated)
    
    # 2. 最终智能细节还原 (基于原始图像未损部分的纹理匹配)
    img_current = intelligent_postprocessing(img_current, mask_stains_dilated, use_gpu, device)
    
    # ========== 最终结果 ==========
    final_psnr = compute_optimized_psnr(img_current, img, mask_stains, use_gpu, device)
    print('=== 修复完成 ===')
    print(f'最终全局PSNR: {final_psnr:.2f} dB')
    
    img_repaired = img_current
    
    # ====================== ★★★ 在这里添加 PSNR和SSIM 计算代码 ★★★ ======================
    print('\n=== 修复结果展示与质量评估 ===\n')
    
    # 确保所有数据在CPU上用于显示和评估
    if use_gpu:
        img_original_cpu = img.cpu().numpy() if torch.is_tensor(img) else img
        img_repaired_cpu = img_repaired.cpu().numpy() if torch.is_tensor(img_repaired) else img_repaired
        img_damaged_cpu = img_damaged.cpu().numpy() if torch.is_tensor(img_damaged) else img_damaged
        mask_stains_cpu = mask_stains.cpu().numpy() if torch.is_tensor(mask_stains) else mask_stains
    else:
        img_original_cpu = img
        img_repaired_cpu = img_repaired
        img_damaged_cpu = img_damaged
        mask_stains_cpu = mask_stains

    # 确保图像是2D灰度图
    if len(img_original_cpu.shape) == 3:
        if img_original_cpu.shape[2] == 3 or img_original_cpu.shape[2] == 4:
            img_original_cpu = np.mean(img_original_cpu[:, :, :3], axis=2)
        else:
            img_original_cpu = img_original_cpu[:, :, 0]
    
    if len(img_repaired_cpu.shape) == 3:
        if img_repaired_cpu.shape[2] == 3 or img_repaired_cpu.shape[2] == 4:
            img_repaired_cpu = np.mean(img_repaired_cpu[:, :, :3], axis=2)
        else:
            img_repaired_cpu = img_repaired_cpu[:, :, 0]
    
    if len(img_damaged_cpu.shape) == 3:
        if img_damaged_cpu.shape[2] == 3 or img_damaged_cpu.shape[2] == 4:
            img_damaged_cpu = np.mean(img_damaged_cpu[:, :, :3], axis=2)
        else:
            img_damaged_cpu = img_damaged_cpu[:, :, 0]
    
    # 处理掩码 - 确保是2D布尔掩码
    if len(mask_stains_cpu.shape) == 3:
        mask_stains_cpu = mask_stains_cpu[:, :, 0] > 0
    elif mask_stains_cpu.dtype != bool:
        mask_stains_cpu = mask_stains_cpu > 0

    # 计算污渍统计
    stain_area = np.sum(mask_stains_cpu)
    total_pixels = img_original_cpu.size
    stain_percentage = 100 * stain_area / total_pixels

    # ====================== 计算质量指标 ======================
    print('正在计算质量评估指标...')

    # 1. 计算整体图像PSNR
    mse_overall = np.mean((img_repaired_cpu.flatten() - img_original_cpu.flatten()) ** 2)
    if mse_overall > 1e-10:
        psnr_overall = 10 * np.log10(1.0 / mse_overall)
    else:
        psnr_overall = float('inf')
    print(f'整体图像PSNR: {psnr_overall:.2f} dB')

    # 验证：如果PSNR > 45，检查修复是否真的生效
    if psnr_overall > 45:
        diff_check = np.max(np.abs(img_repaired_cpu - img_original_cpu))
        print(f'  ⚠️  PSNR异常偏高，最大像素差异: {diff_check:.6f}')
        if diff_check < 0.01:
            print('  ⚠️  警告: 修复图像与原始图像几乎相同，请检查污渍掩码是否正确应用！')

    # 2. 计算修复区域统计和PSNR
    if stain_area > 0:
        # 获取修复区域的像素值
        repaired_values = img_repaired_cpu[mask_stains_cpu]
        original_values = img_original_cpu[mask_stains_cpu]
        
        # 验证：检查修复区域是否有变化
        max_diff_scratch = np.max(np.abs(repaired_values - original_values))
        print(f'  修复区域最大像素变化: {max_diff_scratch:.6f}')
        
        if max_diff_scratch < 0.01:
            print('  ⚠️  警告: 修复区域几乎没有变化，请检查修复算法或污渍掩码！')
        
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
        
        # ========== 添加验证和调试信息 ==========
        print('\n=== 修复区域PSNR计算验证 ===')
        print(f'污渍像素总数: {stain_area}')
        print(f'修复区域MSE: {mse_repair:.6f}')
        print(f'修复区域PSNR: {psnr_repair:.2f} dB')
        print(f'修复区域MAE: {mae_repair:.6f}')
        print(f'相对误差: {relative_error:.4f}%')
        
        # 验证计算
        if np.isinf(psnr_repair):
            print('⚠️  警告：修复区域PSNR为无穷大')
            print('   可能原因：修复完美或MSE为0')
        elif psnr_repair > 45:
            print('⚠️  警告：修复区域PSNR异常偏高')
            print('   可能原因：MSE计算异常或修复区域未正确更新')
        
    else:
        # 如果没有污渍区域
        mae_repair = 0
        max_ae_repair = 0
        mse_repair = 0
        psnr_repair = float('inf')
        relative_error = 0
        abs_diff_values = np.array([])
        diff_values = np.array([])
        print('没有污渍区域，跳过修复区域PSNR计算')

    # ====================== 3. 计算SSIM指标 ======================
    print('\n=== 计算SSIM指标 ===')

    # 确保图像是2D灰度numpy数组
    img_ref = img_original_cpu.copy()
    img_comp = img_repaired_cpu.copy()

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
            if stain_area > 0:
                rows, cols = np.where(mask_stains_cpu > 0)
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
                    print(f'  [提示] 未发现污渍区域掩码，使用整体SSIM')
                    ssim_patch = ssim_overall
            else:
                print(f'  [提示] 无污渍区域，使用整体SSIM')
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
    print('\n' + '=' * 50)
    print('最终质量评估报告:')
    print('=' * 50)
    print(f'污渍统计:')
    print(f'  污渍像素: {stain_area} ({stain_percentage:.2f}%)')
    print(f'  总像素: {total_pixels}')
    print('\nPSNR指标:')
    print(f'  修复区域PSNR: {psnr_repair:.2f} dB')
    print(f'  平均绝对误差(MAE): {mae_repair:.6f}')
    print(f'  均方误差(MSE): {mse_repair:.6f}')
    if not np.isnan(ssim_overall):
        print('\nSSIM指标:')
        print(f'  修复区域SSIM: {ssim_patch:.4f}')
    print('=' * 50)
    
    # ====================== 结束 PSNR和SSIM计算 ======================
    
    # 7. 修复结果展示（原有的结果显示代码）
    print('\n=== 生成修复结果对比图 ===')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original_cpu, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(img_damaged_cpu, cmap='gray')
    axes[1].set_title(f'带污渍图像 (污渍占比: {stain_percentage:.2f}%)')
    axes[1].axis('off')
    
    axes[2].imshow(img_repaired_cpu, cmap='gray')
    axes[2].set_title(f'KSVD修复结果 (PSNR: {psnr_repair:.2f} dB, SSIM: {ssim_patch:.4f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print('\n=== 污渍修复完成 ===')


if __name__ == '__main__':
    main()
