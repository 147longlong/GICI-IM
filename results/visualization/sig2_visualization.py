#!/usr/bin/env python3
"""
Visualize Matrix matrix from visual integrity monitoring
Shows correlation structure for same landmark observations
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def parse_matrix_file(filename):
    """
    Parse the Matrix_output.txt file and extract matrices
    Returns a list of dictionaries with timestamp and matrix data
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by timestamp blocks
    blocks = content.split('----------------------------------------')
    
    data_list = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract timestamp
        # timestamp_match = re.search(r'Timestamp:\s*([\d\.]+)', block)
        # if not timestamp_match:
        #     continue
            
        # timestamp = float(timestamp_match.group(1))
        timestamp = 0.0 # Default timestamp
        
        # Extract shape
        shape_match = re.search(r'Matrix shape:\s*(\d+)\s*x\s*(\d+)', block)
        if not shape_match:
            continue
            
        rows = int(shape_match.group(1))
        cols = int(shape_match.group(2))
        
        # Extract matrix data
        matrix_start = block.find('Matrix:')
        if matrix_start == -1:
            continue
            
        matrix_text = block[matrix_start:].split('\n', 1)[1]
        
        # Parse matrix
        matrix_lines = matrix_text.strip().split('\n')
        matrix_data = []
        
        for line in matrix_lines:
            if not line.strip() or line.startswith('---'):
                break
            # Parse numbers from line
            numbers = [float(x) for x in line.split()]
            matrix_data.append(numbers)
        
        if len(matrix_data) == rows and all(len(row) == cols for row in matrix_data):
            matrix = np.array(matrix_data)
            data_list.append({
                'timestamp': timestamp,
                'matrix': matrix,
                'rows': rows,
                'cols': cols
            })
    
    return data_list

from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

def visualize_matrix(matrix, timestamp, output_dir=None, separators=None):
    """
    使用 SymLogNorm + 平滑白-黄-橙-红 colormap 可视化协方差矩阵
    """
    # ========== 1. 创建平滑 colormap（白→黄→橙→红） ==========
    colors_list = ['white', '#ADD8E6', '#1E90FF', '#00008B']  # 白、浅蓝、道奇蓝、深蓝
    positions = [0.0, 0.3, 0.7, 1.0]
    cmap_academic = LinearSegmentedColormap.from_list('custom_cmap', 
                                                       list(zip(positions, colors_list)),
                                                       N=256)

    # ========== 2. 归一化：SymLogNorm 处理跨数量级 ==========
    vmin = 0.0
    vmax = np.max(matrix)

    # 线性阈值设为 1e-8（介于 1e-10 和 0.0029 之间）
    # 使得 1e-10 ~ 1e-8 区间线性映射，颜色从白色向浅黄过渡
    # 0.0029 ~ 20 也基本在线性区（因为 20 < 1e8? 不对，20 > 1e-8 但仍在 log 区起点附近）
    # 实际上 20 相对于 1e-8 已进入 log 区，但压缩轻微；300+ 被明显压缩
    linthresh = 1e-8
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax)

    # ========== 3. 绘图 ==========
    fig1, ax1 = plt.subplots(figsize=(14, 12))
    im = ax1.imshow(matrix, cmap=cmap_academic, aspect='auto', norm=norm)
    ax1.set_title('Covariance Matrix Structure', 
                  fontsize=18, fontweight='bold')
    ax1.set_xlabel('Column Index (Parameters)', fontsize=16)
    ax1.set_ylabel('Row Index (Residuals)', fontsize=16)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax1, label='Variance Value')
    cbar.ax.tick_params(labelsize=14)

    # 可选边界: 不同时间块实线; 同一时间不同残差类型虚线
    if separators:
        shown_labels = set()
        for sep in separators:
            boundary = sep.get('index')
            style = sep.get('style', 'dashed')
            if boundary is None:
                continue
            if not (0 < boundary < matrix.shape[0]):
                continue

            if style == 'solid':
                line_style = '-'
                color = '#101010'
                width = 2.4
                alpha = 0.95
                label = 'Time Boundary'
            else:
                line_style = '--'
                color = '#606060'
                width = 0.9
                alpha = 0.60
                label = 'Residual Boundary'

            plot_label = label if label not in shown_labels else None
            ax1.axvline(x=boundary - 0.5, color=color, linestyle=line_style,
                        linewidth=width, alpha=alpha, label=plot_label,
                        zorder=5)
            ax1.axhline(y=boundary - 0.5, color=color, linestyle=line_style,
                        linewidth=width, alpha=alpha,
                        zorder=5)
            shown_labels.add(label)

        if shown_labels:
            ax1.legend(loc='upper right', fontsize=10)

    ax1.grid(True, alpha=0.2, color='gray', linestyle=':', linewidth=0.3)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, 'sig2_correlation.png')
        fig1.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved correlation structure to: {out_file}")
    plt.close(fig1)

    # ========== 4. 图2：对角线 + 直方图（保持不变） ==========
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Matrix Analysis (raw values)\n', fontsize=16)

    diag = np.diag(matrix)
    axes[0].plot(diag, 'o-', linewidth=2, markersize=6, color='#1f77b4')
    axes[0].set_title('Diagonal Elements', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Index', fontsize=11)
    axes[0].set_ylabel('Variance', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    off_diag = matrix - np.diag(diag)
    non_zero_mask = np.abs(off_diag) > 1e-10
    if np.any(non_zero_mask):
        off_vals = off_diag[non_zero_mask]
        axes[1].hist(off_vals, bins=20, alpha=0.7, edgecolor='black', color='#2ca02c')
        axes[1].set_title('Distribution of Off-Diagonal Elements', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No off-diagonal correlations', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=14, color='gray')
        axes[1].set_title('Off-Diagonal Elements', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if output_dir:
        out_file2 = os.path.join(output_dir, 'sig2_diag_and_hist.png')
        fig2.savefig(out_file2, dpi=150, bbox_inches='tight')
        print(f"Saved diagonal and histogram to: {out_file2}")
    plt.close(fig2)

def analyze_correlation_structure(matrix, timestamp):
    """
    Analyze the correlation structure to identify blocks corresponding to same landmarks
    """
    print(f"\n=== Analysis for Timestamp: {timestamp} ===")
    print(f"Matrix shape: {matrix.shape}")
    
    # Check if matrix is diagonal
    off_diag = matrix - np.diag(np.diag(matrix))
    max_off_diag = np.max(np.abs(off_diag))
    print(f"Maximum off-diagonal value: {max_off_diag:.6e}")
    
    if max_off_diag < 1e-10:
        print("Matrix is diagonal (no correlations between observations)")
        return
    
    # Find non-zero off-diagonal blocks
    # Group rows/columns that have correlations
    corr_matrix = np.abs(off_diag) > 1e-10
    
    # Find connected components (blocks of correlated observations)
    visited = set()
    blocks = []
    
    for i in range(matrix.shape[0]):
        if i in visited:
            continue
        
        # Find all rows/columns correlated with i
        block = set([i])
        to_check = [i]
        
        while to_check:
            current = to_check.pop(0)
            # Find rows correlated with current
            for j in range(matrix.shape[0]):
                if j not in block and (corr_matrix[current, j] or corr_matrix[j, current]):
                    block.add(j)
                    to_check.append(j)
        
        if len(block) > 1:
            blocks.append(sorted(list(block)))
            visited.update(block)
    
    print(f"Found {len(blocks)} correlated blocks:")
    for idx, block in enumerate(blocks):
        print(f"  Block {idx+1}: rows {block}")
        # Extract submatrix for this block
        submatrix = matrix[np.ix_(block, block)]
        print(f"    Submatrix shape: {submatrix.shape}")
        print(f"    Diagonal values: {np.diag(submatrix)}")
        print(f"    Off-diagonal values: {submatrix - np.diag(np.diag(submatrix))}")
    
    return blocks

def extract_separators_from_jacobian(jacobian_file):
    """
    从 Jacobian Analysis 行块中提取矩阵分隔线。
    规则:
    - 不同时间块之间: 实线 (solid)
    - 同一时间内残差类型变化: 虚线 (dashed)
    """
    if not os.path.exists(jacobian_file):
        print(f"Warning: Jacobian file {jacobian_file} not found!")
        return None
    
    try:
        with open(jacobian_file, 'r') as f:
            content = f.read()
        
        # Find the Jacobian Analysis section
        analysis_start = content.find("Jacobian Analysis:")
        if analysis_start == -1:
            print("Warning: Jacobian Analysis section not found!")
            return None
        
        analysis_section = content[analysis_start:]
        
        # Parse Jacobian Analysis row mapping
        lines = analysis_section.split('\n')

        # 行信息映射: row_index -> {'res_type': str, 'pose_ids': set()}
        row_info = {}

        row_range_re = re.compile(r'\[(\d+),(\d+)\]')
        residual_re = re.compile(r'\(([^)]+)\)')
        param_re = re.compile(r'\s(\d+)\s\(([^)]+)\)\s*$')

        for line in lines:
            if not line.strip() or line.startswith('===') or line.startswith('---'):
                continue

            row_match = row_range_re.search(line)
            if not row_match:
                continue

            row_start = int(row_match.group(1))
            row_end = int(row_match.group(2))

            residual_tokens = residual_re.findall(line)
            if not residual_tokens:
                continue
            # 第一个括号通常是残差类型
            res_type = residual_tokens[0]

            param_match = param_re.search(line)
            pose_id = None
            pose_type = None
            if param_match:
                pose_id = int(param_match.group(1))
                pose_type = param_match.group(2)

            for row in range(row_start, row_end + 1):
                if row not in row_info:
                    row_info[row] = {'res_type': res_type, 'pose_ids': set()}
                if row_info[row]['res_type'] == 'Unknown' and res_type:
                    row_info[row]['res_type'] = res_type
                if pose_id is not None and pose_type in ('gPose', 'cPose'):
                    row_info[row]['pose_ids'].add(pose_id)

        if not row_info:
            print("Warning: No row mapping found in Jacobian analysis!")
            return None

        sorted_rows = sorted(row_info.keys())
        row_meta = []
        last_time_key = None
        for row in sorted_rows:
            info = row_info[row]
            if info['pose_ids']:
                time_key = min(info['pose_ids'])
                last_time_key = time_key
            else:
                # 对无 Pose 绑定项(如频率/模糊度), 继承最近时间块
                time_key = last_time_key

            row_meta.append({
                'row': row,
                'res_type': info['res_type'],
                'time_key': time_key,
            })

        def is_ambiguity_family(res_type):
            return res_type in ('AmbiguityError', 'RelativeAmbiguityError', 'FrequencyError', 'RelativeFrequencyError')

        separators = []
        for i in range(1, len(row_meta)):
            prev_meta = row_meta[i - 1]
            curr_meta = row_meta[i]
            if curr_meta['row'] != prev_meta['row'] + 1:
                continue

            is_time_change = curr_meta['time_key'] != prev_meta['time_key']
            is_res_change = curr_meta['res_type'] != prev_meta['res_type']

            if is_time_change:
                separators.append({'index': curr_meta['row'], 'style': 'solid'})
            elif is_res_change:
                # Ambiguity 家族与其他残差块之间不画虚线
                if is_ambiguity_family(prev_meta['res_type']) or is_ambiguity_family(curr_meta['res_type']):
                    continue
                separators.append({'index': curr_meta['row'], 'style': 'dashed'})

        # 去重(同一个 index 可能重复), 优先保留实线
        dedup = {}
        for sep in separators:
            idx = sep['index']
            sty = sep['style']
            if idx not in dedup or sty == 'solid':
                dedup[idx] = sty
        separators = [{'index': idx, 'style': dedup[idx]} for idx in sorted(dedup.keys())]

        solid_cnt = sum(1 for s in separators if s['style'] == 'solid')
        dashed_cnt = sum(1 for s in separators if s['style'] == 'dashed')
        print(f"Extracted separators: total={len(separators)}, solid={solid_cnt}, dashed={dashed_cnt}")
        return separators
        
    except Exception as e:
        print(f"Error parsing Jacobian file: {e}")
        return None


def apply_manual_time_boundaries(separators, manual_time_boundaries, matrix_rows):
    """
    将手工给定的时间分块边界强制合并进 separators。
    手工时间边界总是实线，并覆盖同 index 的虚线。
    """
    merged = {}

    if separators:
        for sep in separators:
            idx = sep.get('index')
            sty = sep.get('style', 'dashed')
            if idx is None:
                continue
            if 0 < idx < matrix_rows:
                merged[idx] = sty

    if manual_time_boundaries:
        for idx in manual_time_boundaries:
            if 0 < idx < matrix_rows:
                merged[idx] = 'solid'

    out = [{'index': idx, 'style': merged[idx]} for idx in sorted(merged.keys())]
    solid_cnt = sum(1 for s in out if s['style'] == 'solid')
    dashed_cnt = sum(1 for s in out if s['style'] == 'dashed')
    print(f"After manual time boundaries: total={len(out)}, solid={solid_cnt}, dashed={dashed_cnt}")
    return out

def main():
    # File paths
    matrix_file = "/home/syl/GICI-IM/results/visualization/sig2_int_gnss_vis_wo_corr.txt"
    jacobian_file = "/home/syl/GICI-IM/results/visualization/jacobian_vis_rrr_wo_corr.txt"
    output_dir = "/home/syl/GICI-IM/results/"
    # 手工指定时间块分界(行索引): 不同时间块之间画实线
    manual_time_boundaries = [129, 247, 491, 664]# [123, 230, 484, 667]
    
    # Parse data
    print("Parsing Matrix output file...")
    data_list = parse_matrix_file(matrix_file)
    
    if not data_list:
        print("No data found in file!")
        return
    
    print(f"Found {len(data_list)} matrices to visualize")
    
    # Extract separators from Jacobian analysis
    print("\nExtracting separators from Jacobian analysis...")
    separators = extract_separators_from_jacobian(jacobian_file)

    if separators is None:
        print("Warning: Could not extract separators. Proceeding without grouping lines.")
    
    # Process each matrix
    for i, data in enumerate(data_list):
        print(f"\nProcessing matrix {i+1}/{len(data_list)}")

        # 基于当前矩阵尺寸应用手工时间边界，确保不同时间块可见
        separators_for_plot = apply_manual_time_boundaries(
            separators,
            manual_time_boundaries,
            data['matrix'].shape[0]
        )
        
        # Analyze structure
        blocks = analyze_correlation_structure(data['matrix'], data['timestamp'])
        
        # Visualize with separators
        visualize_matrix(data['matrix'], data['timestamp'], output_dir, separators_for_plot)
        
        # Ask if user wants to continue
        if i < len(data_list) - 1:
            response = input(f"\nPress Enter to continue to next matrix, or 'q' to quit: ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    main()