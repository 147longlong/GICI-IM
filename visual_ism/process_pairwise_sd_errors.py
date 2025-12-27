import numpy as np
import matplotlib.pyplot as plt
import os
from paired_overbounding import paired_overbounding
from fcdf_overbounding import fcdf_overbounding
import glob
import pathlib
import json




def solve_qk_for_phi(phi, max_j, file_paths, threshold=2e-3):
    """针对特定phi值求解q_i"""
    # 读取数据
    data_map = {}
    all_errors = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                pair_type = parts[0]
                try:
                    error = float(parts[1])
                except ValueError:
                    continue
                i, j = parse_pair(pair_type)
                
                # 判断是否满足条件
                if i is not None and j is not None and j <= max_j:
                    # 如果phi是nan，不限制j-i；否则要求j-i <= phi
                    if np.isnan(phi) or (j - i <= phi):
                        if pair_type not in data_map:
                            data_map[pair_type] = []
                        data_map[pair_type].append(error)
                        all_errors.append(error)

    if not all_errors:
        return None, None, None

    # 计算overbounding sigmas
    sorted_keys = sorted(data_map.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    results = []
    for pair_name in sorted_keys:
        errors = np.array(data_map[pair_name])
        if len(errors) < max_j:
            continue
        
        try:
            errors_dist = np.sqrt(errors)
            errors_sym = np.concatenate((-errors_dist, errors_dist))
            sigma_ob, _ = fcdf_overbounding(errors_sym, pair_name, force_b_zero=True, plot_flag=False, flag_remove=True, Threshold=threshold)
            results.append((pair_name, sigma_ob))
        except Exception as e:
            print(f"Error calculating for {pair_name}: {e}")
            continue

    if not results:
        return None, None, None

    # 构建方程组
    A = []
    b = []
    num_vars = max_j
    valid_pairs = 0
    for pair_name, sigma_val in results:
        i, j = parse_pair(pair_name)
        if i is None or j is None:
            continue
        if j > max_j:
            continue
        LHS = sigma_val**2
        row = np.zeros(num_vars)
        for k in range(i+1, j+1):
            if k-1 < num_vars:
                row[k-1] += 2
        A.append(row)
        b.append(LHS)
        valid_pairs += 1
    
    if valid_pairs < num_vars:
        print(f"Phi={phi}: Not enough data points. Need at least {num_vars}, got {valid_pairs}.")
        return None, None, None
    
    A = np.array(A)
    b = np.array(b)
    
    # 普通最小二乘法求解 q_i^2
    q_sq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 确保q_sq为非负值（处理数值误差）
    q_sq = np.maximum(q_sq, 0)
    q_i = np.sqrt(q_sq)
    
    # 计算有效数据点数量
    num_samples = len(all_errors)
    num_categories = len(data_map)
    
    return q_i, num_samples, num_categories


def main(max_j=200, phi_values=None, threshold=2e-3, 
         data_dir='/media/syl/longlong/GICI-Dataset',
         results_file='/home/syl/GICI-IM/visual_ism/qk_results.json',
         plot_file='/home/syl/GICI-IM/visual_ism/phi_analysis.png'):
    """主函数
    
    参数:
        max_j: 最大j值
        phi_values: phi值列表，如 [1,2,3,np.nan]
        threshold: fcdf_overbounding的阈值
        data_dir: 数据目录路径
        results_file: 结果保存文件路径
        plot_file: 分析图保存路径
    """
    if phi_values is None:
        phi_values = [3]  # 默认phi值
    
    # 检查是否已有保存的结果
    if os.path.exists(results_file):
        print(f"Found saved results file: {results_file}")
        qk_results_all, summary_data_all = load_qk_results(results_file)
        
        if qk_results_all is not None:
            # 根据phi_values筛选对应的结果
            qk_results = {}
            summary_data = []
            
            for phi in phi_values:
                if phi in qk_results_all:
                    qk_results[phi] = qk_results_all[phi]
                    # 在summary_data_all中找到对应的条目
                    for data in summary_data_all:
                        if (np.isnan(phi) and data['phi'] == 'nan') or (not np.isnan(phi) and data['phi'] == phi):
                            summary_data.append(data)
                            break
            
            if qk_results:
                # 绘制分析图表
                plot_phi_analysis(qk_results, summary_data, plot_file)
                print(f"Plotted {len(qk_results)} phi values: {list(qk_results.keys())}")
                return
            else:
                print("No matching phi values found in saved results.")
    
    # 如果没有保存的结果或没有匹配的phi值，重新计算
    print("No saved results found or no matching phi values, computing from scratch...")
    
    # 自动查找所有pairwise_sd_errors.txt
    file_paths = glob.glob(f'{data_dir}/*/pairwise_sd_errors.txt')
    if not file_paths:
        print(f"No pairwise_sd_errors.txt files found in {data_dir}")
        return

    # 存储不同phi的结果
    qk_results = {}
    summary_data = []

    for phi in phi_values:
        q_i, num_samples, num_categories = solve_qk_for_phi(phi, max_j, file_paths, threshold)
        
        if q_i is not None:
            qk_results[phi] = q_i
            summary_data.append({
                'phi': phi if not np.isnan(phi) else 'nan',
                'num_samples': num_samples,
                'num_categories': num_categories
            })

    if not qk_results:
        print("No valid results for any phi value.")
        return

    # 保存结果到文件
    save_qk_results(qk_results, summary_data, results_file)
    
    # 绘制分析图表
    plot_phi_analysis(qk_results, summary_data, plot_file)


def save_qk_results(qk_results, summary_data, output_file):
    """保存q_k结果到JSON文件"""
    # 将numpy数组转换为列表，处理nan值
    data_to_save = {
        'qk_results': {('nan' if np.isnan(phi) else str(phi)): q_k.tolist() for phi, q_k in qk_results.items()},
        'summary_data': summary_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved q_k results to {output_file}")


def load_qk_results(input_file):
    """从JSON文件加载q_k结果"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # 处理phi值，包括nan
        qk_results = {}
        for phi_str, q_k in data['qk_results'].items():
            if phi_str == 'nan':
                phi = np.nan
            else:
                phi = int(phi_str)
            qk_results[phi] = np.array(q_k)
        
        summary_data = data['summary_data']
        
        print(f"Loaded q_k results from {input_file}")
        return qk_results, summary_data
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None


def plot_phi_analysis(qk_results, summary_data, output_file):
    """绘制不同phi值对q_i影响的分析图"""
    if not qk_results:
        print("No q_i results to plot.")
        return
    
    # 对phi值排序，nan放在最后
    phi_values = sorted(qk_results.keys(), key=lambda x: (np.isnan(x), x))
    
    # 创建2x3子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Effect of Different Phi Values on q_i Estimation', fontsize=14, fontweight='bold')
    
    # 1. 不同phi的q_i曲线对比（全部）
    ax1 = axes[0, 0]
    for phi in phi_values:
        q_i = qk_results[phi]
        indices = range(1, len(q_i) + 1)
        phi_label = 'nan' if np.isnan(phi) else f'phi={phi}'
        ax1.plot(indices, q_i, marker='o', linestyle='-', linewidth=1.5, markersize=2, label=phi_label)
    ax1.set_xlabel('Index i', fontsize=10)
    ax1.set_ylabel('q_i', fontsize=10)
    ax1.set_title('q_i vs Index (All)', fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(fontsize=8, loc='upper right')
    
    # 2. 数据量统计
    ax2 = axes[0, 1]
    samples = [data['num_samples'] for data in summary_data]
    categories = [data['num_categories'] for data in summary_data]
    
    # 创建x轴位置，处理nan
    x_positions = []
    x_labels = []
    for i, data in enumerate(summary_data):
        if data['phi'] == 'nan':
            x_positions.append(i)
            x_labels.append('nan')
        else:
            x_positions.append(data['phi'])
            x_labels.append(str(data['phi']))
    
    ax2.bar([x-0.2 for x in x_positions], samples, width=0.4, label='Samples', color='skyblue', alpha=0.8)
    ax2.bar([x+0.2 for x in x_positions], categories, width=0.4, label='Categories', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Phi Value', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Data Volume vs Phi', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    ax2.legend(fontsize=8)
    
    # 3. 前10个q_i的热图
    ax3 = axes[0, 2]
    heatmap_data = []
    for phi in phi_values:
        q_i = qk_results[phi]
        heatmap_data.append(q_i[:10])  # 前10个
    
    heatmap_data = np.array(heatmap_data)
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    ax3.set_xticks(range(10))
    ax3.set_xticklabels([f'{i+1}' for i in range(10)])
    ax3.set_yticks(range(len(phi_values)))
    ax3.set_yticklabels(['nan' if np.isnan(phi) else phi for phi in phi_values])
    ax3.set_xlabel('Index k', fontsize=10)
    ax3.set_ylabel('Phi Value', fontsize=10)
    ax3.set_title('Heatmap: q_i (k=1-10)', fontsize=11, fontweight='bold')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. q_i的平滑度分析（二阶差分）
    ax4 = axes[1, 0]
    for phi in phi_values:
        q_i = qk_results[phi]
        if len(q_i) >= 3:
            # 计算一阶差分（变化率）
            first_diff = np.diff(q_i)
            # 计算二阶差分（加速度/平滑度）
            second_diff = np.diff(first_diff)
            
            # indices应该与second_diff长度一致
            indices = range(2, 2 + len(second_diff))
            phi_label = 'nan' if np.isnan(phi) else f'phi={phi}'
            ax4.plot(indices, second_diff, marker='o', linestyle='-', linewidth=1.5, markersize=2, label=phi_label)
    
    ax4.set_xlabel('Index k', fontsize=10)
    ax4.set_ylabel('Second Difference', fontsize=10)
    ax4.set_title('Smoothness (2nd Diff)', fontsize=11, fontweight='bold')
    ax4.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend(fontsize=8, loc='upper right')
    
    # 5. q_i的归一化变化率分析
    ax5 = axes[1, 1]
    for phi in phi_values:
        q_i = qk_results[phi]
        if len(q_i) >= 2:
            # 计算归一化变化率 (q_i - q_{i+1}) / q_i
            change_rates = []
            for i in range(len(q_i)-1):
                if q_i[i] > 0:
                    rate = (q_i[i] - q_i[i+1]) / q_i[i]
                    change_rates.append(rate)
            
            indices = range(1, len(change_rates) + 1)
            phi_label = 'nan' if np.isnan(phi) else f'phi={phi}'
            ax5.plot(indices, change_rates, marker='o', linestyle='-', linewidth=1.5, markersize=2, label=phi_label)
    
    ax5.set_xlabel('Index k', fontsize=10)
    ax5.set_ylabel('Normalized Change Rate', fontsize=10)
    ax5.set_title('Decrease Rate', fontsize=11, fontweight='bold')
    ax5.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.legend(fontsize=8, loc='upper right')
    
    # 6. q_i的对数坐标图（观察衰减趋势）
    ax6 = axes[1, 2]
    for phi in phi_values:
        q_i = qk_results[phi]
        indices = range(1, len(q_i) + 1)
        phi_label = 'nan' if np.isnan(phi) else f'phi={phi}'
        ax6.plot(indices, q_i, marker='o', linestyle='-', linewidth=1.5, markersize=2, label=phi_label)
    
    ax6.set_xlabel('Index k', fontsize=10)
    ax6.set_ylabel('q_i', fontsize=10)
    ax6.set_title('q_i vs Index (Log Scale)', fontsize=11, fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax6.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Phi analysis plot saved to {output_file}")
    plt.close(fig)



def plot_relative_frequency(data_map, bins, bin_centers, output_file):
    # Plotting Relative Frequency
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort keys to process in order
    sorted_keys = sorted(data_map.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

    # Color map for 0-x series
    # We want to highlight 0-1, 0-2, ..., 0-9
    # There are at most 9 such pairs (0-1 to 0-9)
    highlight_colors = plt.cm.jet(np.linspace(0, 1, 10))
    
    # To create an envelope, we can collect all frequency curves
    all_freqs = []

    for pair_name in sorted_keys:
        errors = np.array(data_map[pair_name])
        hist, _ = np.histogram(errors, bins=bins)
        # Normalize
        if len(errors) > 0:
            rel_freq = hist / len(errors)
        else:
            rel_freq = np.zeros_like(hist, dtype=float)
        
        all_freqs.append(rel_freq)
        
        i, j = parse_pair(pair_name)
        
        # Plot
        mask_high = rel_freq > 1e-3
        mask_low = rel_freq <= 1e-3

        if i == 0:
            # Highlight 0-x
            color_idx = j % 10
            ax.plot(bin_centers[mask_low], rel_freq[mask_low], 
                    color=highlight_colors[color_idx], marker='.', linestyle='None', zorder=10)
            ax.plot(bin_centers[mask_high], rel_freq[mask_high], label=pair_name, 
                    color=highlight_colors[color_idx], linestyle='-', zorder=10)
        else:
            # Background envelope
            ax.plot(bin_centers[mask_low], rel_freq[mask_low], 
                    color='grey', alpha=0.2, marker='.', linestyle='None', zorder=1)
            ax.plot(bin_centers[mask_high], rel_freq[mask_high], 
                    color='grey', alpha=0.2, linestyle='-', zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sampson Error', fontsize=20)
    ax.set_ylabel('Relative Frequency', fontsize=20)
    ax.set_title('Pairwise Sampson Error Distribution (Start i = 0)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.16, 1), fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    # plt.show() # Cannot show in this environment

def plot_relative_frequency_subplots(data_map, bins, bin_centers, min_val, max_val, output_file):
    # Plotting Relative Frequency
    # Group data by i to create subplots
    data_by_i = {}
    for pair_name in data_map.keys():
        i, j = parse_pair(pair_name)
        if i not in data_by_i:
            data_by_i[i] = []
        data_by_i[i].append(pair_name)
    
    sorted_i = sorted(data_by_i.keys())
    n_plots = len(sorted_i)
    
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=False, sharey=True)
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    highlight_colors = plt.cm.jet(np.linspace(0, 1, 10))

    for idx, i_val in enumerate(sorted_i):
        ax = axes[idx]
        pairs = sorted(data_by_i[i_val], key=lambda x: int(x.split('-')[1]))
        
        for pair_name in pairs:
            i, j = parse_pair(pair_name)
            errors = np.array(data_map[pair_name])
            hist, _ = np.histogram(errors, bins=bins)
            
            if len(errors) > 0:
                rel_freq = hist / len(errors)
            else:
                rel_freq = np.zeros_like(hist, dtype=float)
            
            # Filter out data with frequency < 1e-5
            mask_visible = rel_freq >= 1e-5
            
            if not np.any(mask_visible):
                continue

            visible_centers = bin_centers[mask_visible]
            visible_freq = rel_freq[mask_visible]

            mask_high = visible_freq > 1e-3
            mask_low = visible_freq <= 1e-3
            
            color_idx = j % 10
            c = highlight_colors[color_idx]
            
            # Plot points for <= 1e-3
            if np.any(mask_low):
                ax.plot(visible_centers[mask_low], visible_freq[mask_low], 
                        color=c, marker='.', linestyle='None', markersize=3, alpha=0.5)
            
            # Plot lines for > 1e-3
            if np.any(mask_high):
                ax.plot(visible_centers[mask_high], visible_freq[mask_high], label=pair_name, 
                        color=c, linestyle='-', linewidth=1.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(min_val, max_val)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(f'Pairwise Sampson Error Distribution (Start i={i_val})', fontsize=16)
        ax.legend(fontsize=12, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        if idx % cols == 0:
            ax.set_ylabel('Relative Frequency', fontsize=14)
        ax.set_xlabel('Sampson Error', fontsize=14)

    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    # plt.show() # Cannot show in this environment


def plot_qk_comparison(q_i_nnls, q_i_ols, output_file):
    """绘制NNLS和OLS的q_i对比图"""
    if q_i_nnls is None or q_i_ols is None:
        print("No q_i data to plot.")
        return
    
    indices = range(1, len(q_i_nnls) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. q_i值对比
    ax1.plot(indices, q_i_nnls, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue', label='NNLS')
    ax1.plot(indices, q_i_ols, marker='s', linestyle='--', linewidth=2, markersize=4, color='red', label='OLS')
    ax1.set_xlabel('Index k', fontsize=12)
    ax1.set_ylabel('q_i', fontsize=12)
    ax1.set_title('q_i Values: NNLS vs OLS', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend()
    ax1.set_xlim(0, len(q_i_nnls) + 1)
    
    # 2. 差异对比
    diffs = q_i_nnls - q_i_ols
    colors = ['green' if d >= 0 else 'red' for d in diffs]
    ax2.bar(indices, diffs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Index k', fontsize=12)
    ax2.set_ylabel('Difference (NNLS - OLS)', fontsize=12)
    ax2.set_title('Difference between NNLS and OLS', fontsize=14, fontweight='bold')
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xlim(0, len(q_i_nnls) + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved q_i comparison plot to {output_file}")
    plt.close(fig)

def plot_sigma_trend(results, output_file):
    # results: list of (pair_name, sigma_ob)
    if not results:
        print("No sigma results to plot.")
        return

    # sort by pair indices (i,j)
    def key_fn(t):
        try:
            i, j = t[0].split('-')
            return (int(i), int(j))
        except:
            return (999, 999)

    results_sorted = sorted(results, key=key_fn)
    pair_names = [r[0] for r in results_sorted]
    sigmas = np.array([r[1] for r in results_sorted])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(sigmas)), sigmas, marker='o', linestyle='-')
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(pair_names, rotation=90, fontsize=8)
    ax.set_xlabel('Pair', fontsize=12)
    ax.set_ylabel('sigma_ob', fontsize=12)
    ax.set_title('Overbounding sigma trend per pair', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved sigma trend plot to {output_file}")
    plt.close(fig)

def parse_pair(pair_str):
    try:
        parts = pair_str.split('-')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except:
        pass
    return None, None
def plot_from_saved_file(results_file=None, phi_values=None):
    """从保存的文件直接绘制图表"""
    if results_file is None:
        results_file = '/home/syl/GICI-IM/visual_ism/qk_results.json'
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    qk_results_all, summary_data_all = load_qk_results(results_file)
    if qk_results_all is None:
        return
    
    # 如果没有指定phi_values，默认使用所有值
    if phi_values is None:
        phi_values = list(qk_results_all.keys())
    
    # 筛选对应的phi值
    qk_results = {}
    summary_data = []
    
    for phi in phi_values:
        if phi in qk_results_all:
            qk_results[phi] = qk_results_all[phi]
            # 在summary_data_all中找到对应的条目
            for data in summary_data_all:
                if (np.isnan(phi) and data['phi'] == 'nan') or (not np.isnan(phi) and data['phi'] == phi):
                    summary_data.append(data)
                    break
    
    if qk_results:
        plot_phi_analysis(qk_results, summary_data, '/home/syl/GICI-IM/visual_ism/phi_analysis.png')
        print(f"Plot generated from saved data for phi values: {[('nan' if np.isnan(phi) else phi) for phi in qk_results.keys()]}")
    else:
        print("No matching phi values found in saved results.")


if __name__ == "__main__":
    # 检查命令行参数
    import sys
    
    # ==================== 参数配置 ====================
    # 计算参数
    max_j = 500
    phi_values = [3]
    threshold = 2e-3  # fcdf_overbounding的阈值
    
    # 路径配置
    data_dir = '/media/syl/longlong/GICI-Dataset'  # 数据目录
    results_file = '/home/syl/GICI-IM/visual_ism/qk_results1e-2.json'  # 结果文件
    plot_file = '/home/syl/GICI-IM/visual_ism/phi_analysis.png'  # 分析图文件
    # ==================================================
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--plot-only':
            # 只绘图，不重新计算
            if len(sys.argv) > 2:
                # 从命令行读取phi值
                phi_str_list = sys.argv[2].split(',')
                phi_values = []
                for phi_str in phi_str_list:
                    phi_str = phi_str.strip()
                    if phi_str.lower() == 'nan':
                        phi_values.append(np.nan)
                    else:
                        phi_values.append(int(phi_str))
                print(f"Plotting only for phi values: {phi_values}")
            
            plot_from_saved_file(results_file=results_file, phi_values=phi_values)
        else:
            # 从命令行读取phi值进行计算
            phi_str_list = sys.argv[1].split(',')
            phi_values = []
            for phi_str in phi_str_list:
                phi_str = phi_str.strip()
                if phi_str.lower() == 'nan':
                    phi_values.append(np.nan)
                else:
                    phi_values.append(int(phi_str))
            
            print(f"Computing for phi values: {phi_values}")
            
            # 调用main函数，使用配置的路径
            main(max_j=max_j, phi_values=phi_values, threshold=threshold,
                 data_dir=data_dir, results_file=results_file, plot_file=plot_file)
    else:
        # 默认流程：使用配置的参数
        main(max_j=max_j, phi_values=phi_values, threshold=threshold,
             data_dir=data_dir, results_file=results_file, plot_file=plot_file)
