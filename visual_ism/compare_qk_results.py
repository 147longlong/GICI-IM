"""
比较不同故障概率下求出的各步跟踪误差
读取qk_results1e-2.json、qk_results1e-3.json、qk_results1e-4.json文件
绘制对比图，展示不同故障概率下phi=3的q_i值（i=1-200）
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


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
        print(f"Error loading results from {input_file}: {e}")
        return None, None


def compare_qk_different_probabilities():
    """比较不同故障概率下的q_i结果"""
    
    # 配置路径
    base_dir = '/home/dell/sunyulong/GICI-IM/visual_ism'
    result_files = {
        '1e-2': os.path.join(base_dir, 'qk_results1e-2.json'),
        '1e-3': os.path.join(base_dir, 'qk_results1e-3.json'),
        '1e-4': os.path.join(base_dir, 'qk_results1e-4.json')
    }
    
    # 目标phi值
    target_phi = 3
    
    # 存储不同概率下的q_i数据
    qk_data = {}
    
    # 读取所有文件
    for prob_name, file_path in result_files.items():
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        qk_results_all, _ = load_qk_results(file_path)
        if qk_results_all is None:
            print(f"无法加载文件: {file_path}")
            continue
        
        # 检查是否有phi=3的数据
        if target_phi in qk_results_all:
            q_i = qk_results_all[target_phi]
            # 取前200个值（i=1-200）
            qk_data[prob_name] = q_i[:200]
            print(f"已加载 {prob_name} 的phi={target_phi}数据，共 {len(q_i[:200])} 个点")
        else:
            print(f"文件 {file_path} 中没有phi={target_phi}的数据")
    
    if not qk_data:
        print("没有找到任何有效数据")
        return
    
    # 绘制对比图
    plot_comparison(qk_data, target_phi, base_dir)


def plot_comparison(qk_data, target_phi, output_dir):
    """绘制不同故障概率下的q_i对比图"""
    
    # --- 绘图参数设置 ---
    FONT_SIZE_GLOBAL = 16
    FONT_SIZE_TITLE = 36
    FONT_SIZE_LABEL = 32
    FONT_SIZE_TICK = 28
    FONT_SIZE_LEGEND = 24
    FONT_SIZE_LEGEND_TITLE = 28

    # --- 绘图设置 (白色背景，大字体) ---
    plt.style.use('default') # 使用默认样式（通常是白色背景）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 确保兼容性
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': FONT_SIZE_GLOBAL,              # 全局字体大小
        'axes.titlesize': FONT_SIZE_TITLE,          # 标题字体大小
        'axes.labelsize': FONT_SIZE_LABEL,          # 轴标签字体大小
        'xtick.labelsize': FONT_SIZE_TICK,          # x轴刻度字体大小
        'ytick.labelsize': FONT_SIZE_TICK,          # y轴刻度字体大小
        'legend.fontsize': FONT_SIZE_LEGEND,        # 图例字体大小
        'figure.facecolor': 'white',  # 图片背景色
        'axes.facecolor': 'white',    # 坐标轴背景色
        'axes.grid': True,            # 开启网格
        'grid.alpha': 0.4,            # 网格透明度
        'grid.linestyle': '--',       # 网格线型
        'lines.linewidth': 2.5        # 线宽
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(30, 12))
    fig.suptitle(f'Comparison of Tracking Error $q_i$ under Different Prior Fault Probabilities (φ={target_phi})', 
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    # 颜色映射
    colors = {
        '1e-2': 'red',
        '1e-3': 'blue',
        '1e-4': 'green'
    }
    labels = {
        '1e-2': r'$P_f = 10^{-2}$',
        '1e-3': r'$P_f = 10^{-3}$',
        '1e-4': r'$P_f = 10^{-4}$'
    }
    
    # 1. 线性坐标对比图
    ax1 = axes[0]
    for prob_name, q_i in qk_data.items():
        indices = range(1, len(q_i) + 1)
        ax1.plot(indices, q_i, marker='o', linestyle='-', linewidth=2.5, 
                markersize=6, color=colors[prob_name], label=labels[prob_name])
    
    ax1.set_xlabel(r'$i$', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel(r'$q_i$', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('Linear Scale', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.legend(fontsize=FONT_SIZE_LEGEND)
    ax1.set_xlim(0, 201)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # 2. 对数坐标对比图
    ax2 = axes[1]
    for prob_name, q_i in qk_data.items():
        indices = range(1, len(q_i) + 1)
        ax2.plot(indices, q_i, marker='o', linestyle='-', linewidth=2.5, 
                markersize=6, color=colors[prob_name], label=labels[prob_name])
    
    ax2.set_xlabel(r'$i$', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel(r'$q_i$', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Log Scale', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.legend(fontsize=FONT_SIZE_LEGEND)
    ax2.set_xlim(0, 201)
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'qk_comparison_1e-2_1e-3_1e-4.png')
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to: {output_file}")
    plt.close(fig)



if __name__ == "__main__":
    compare_qk_different_probabilities()