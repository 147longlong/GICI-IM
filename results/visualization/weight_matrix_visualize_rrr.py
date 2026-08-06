import sys
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_jacobian_file_for_weight(filepath):
    # 解析 Jacobian 文本文件，提取残差项的维度，用于绘制权重矩阵对角块
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if "Residual Type" in line:
            header_idx = i
            break
            
    header_line = lines[header_idx].rstrip('\n')
    cols_part = header_line[35:]
    num_cols = len(cols_part) // 21

    residual_groups = {}
    residual_patterns = []
    
    for line in lines[header_idx+4:]:
        line = line.rstrip('\n')
        if not line or line.startswith('---') or line.startswith('==='): continue
        row_name_raw = line[0:35].strip()
        if not row_name_raw: continue
        
        if 'Error' not in row_name_raw:
            continue
            
        cells_str = line[35:]
        active_orig_cols = []
        for i in range(num_cols):
            if i*21 >= len(cells_str): break
            cell = cells_str[i*21:(i+1)*21].strip()
            if set(cell) - set('- ') != set(): active_orig_cols.append(i)
                
        row_name_full = line.split("  ")[0] if "  " in line else line.split()[0]
        
        if 'IMU' in row_name_raw: base = 'IMU'
        elif 'Reproj' in row_name_raw or 'Reprojection' in row_name_raw: base = 'Reproj'
        elif 'Marg' in row_name_raw: base = 'Marg'
        elif 'Pos' in row_name_raw: base = 'Pos'
        elif 'Vel' in row_name_raw: base = 'Vel'
        elif 'HMC' in row_name_raw: base = 'HMC'
        elif 'NHC' in row_name_raw: base = 'NHC'
        elif 'Pseudorange' in row_name_raw: base = 'Pseudorange'
        elif 'Phaserange' in row_name_raw: base = 'Phaserange'
        elif 'Doppler' in row_name_raw: base = 'Doppler'
        elif 'Frequency' in row_name_raw: base = 'Frequency'
        elif 'Ambiguity' in row_name_raw: base = 'Ambiguity'
        else: base = row_name_raw.split('(')[0].strip()
            
        dim_match = re.search(r'\((.*?)\)', row_name_full)
        row_dim = 1
        if dim_match:
            dim_str = dim_match.group(1)
            parts = dim_str.split('x')
            dim_val = 1
            for p in parts:
                try: dim_val *= int(p)
                except: pass
            row_dim = dim_val

        # 根据影响的列来简单分组残差，这样在矩阵对角线可以显示为连续的维度块
        # 对 Ambiguity 和 Frequency进行强制合并，避免产生无数1x1矩阵块
        if base in ['Ambiguity', 'Frequency']:
            key = base
        else:
            key = (base, tuple(active_orig_cols))
        
        if key not in residual_groups:
            residual_groups[key] = {
                'dim': row_dim,
                'base': base
            }
            residual_patterns.append(key)
        else:
            residual_groups[key]['dim'] += row_dim
            
    return residual_patterns, residual_groups

def draw_weight_matrix_structure():
    filepath = '/home/syl/GICI-IM/results/visualization/jacobian_visualization-rrr.txt'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    residual_patterns, residual_groups = parse_jacobian_file_for_weight(filepath)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # 画布设置为正方形或者比例近似正方形，因为协方差矩阵是基于残差维度构成的方阵
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 颜色配置，与刚刚的jacobian保持绝对一致
    colors = {
        'Marg': "#A34FE7",
        'IMU': "#f16d6d6c", 
        'NHC': "#CBE8FFBA",
        'Pos': "#e3ab77",
        'HMC': '#2CA02C',  
        'Vel': '#FFFFCC',
        'Reproj': "#cfe4c3",
        'Pseudorange': "#0299FE",      
        'Phaserange': "#807DBA",      
        'Doppler': "#E6D7B5", 
        'Frequency': "#AAAAAA",
        'Ambiguity': "#DDDDDD",
        'Other': "#cccccc"
    }

    current_pos = 0
    ticks = []
    labels = []
    
    ax.axhline(0, color='#e0e0e0', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='#e0e0e0', linestyle='-', linewidth=0.5)

    scale_factor = 0.15 
    standard_visual_dim = 15  # 为了图表能看清，最小块强制按15倍距缩放绘制，其它大块按比例

    used_keys = set()

    for key in residual_patterns:
        group_data = residual_groups[key]
        dim = group_data['dim']
        type_name = group_data['base']
        
        # 定制化各类矩阵块的可视化大小
        if type_name == 'Marg':
            visual_dim = 22  # 将Marg稍稍放大，防止坐标轴由于挤压而黏连在一起
        elif type_name in ['Ambiguity', 'Frequency']:
            visual_dim = 50 # 两者大矩阵块保底尺寸完全保持一致
        elif type_name in ['Reproj', 'Phaserange', 'Pseudorange', 'Doppler']:
            visual_dim = max(45, dim * 2) # 重点突出这四类，放大展示尺寸
        else:
            visual_dim = max(22, dim) # 普通模块保底尺寸增加（15->22），拉开坐标距离防重叠

        plot_size = visual_dim * scale_factor
        
        color = colors.get(type_name, colors['Other'])
        used_keys.add(type_name)
        
        # 判断是否需要应用“突出”的框线省略样式
        is_highlighted = type_name in ['Reproj', 'Phaserange', 'Pseudorange', 'Doppler', 'Ambiguity', 'Frequency']
        
        if is_highlighted:
            # 外部大包围框，半透明底色，强化边界粗细
            rect = patches.Rectangle((current_pos, current_pos), plot_size, plot_size, 
                                     linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.25)
            ax.add_patch(rect)
            
            # 画内部稀疏子方块
            if type_name in ['Ambiguity', 'Frequency']:
                # 强制 Ambiguity 和 Frequency 的内部小块(小权重块)外部大小一致
                sub_plot_size = 16 * scale_factor
                num_sub_blocks = 5 # 80 / 16 = 5
                num_draw_start = 2
                num_draw_end = 2
            else:
                sub_dim = 8 if visual_dim > 30 else max(4, visual_dim // 4)
                sub_plot_size = sub_dim * scale_factor
                num_sub_blocks = int(visual_dim // sub_dim)
                if num_sub_blocks < 4: num_sub_blocks = 4
                num_draw_start = 3
                num_draw_end = 3
                
            # 区分一下这四类内部小块的颜色（实心且比外框深）以满足要求
            if type_name in ['Reproj', 'Phaserange', 'Pseudorange', 'Doppler']:
                inner_fc = color
                inner_ec = 'white'
                inner_alpha = 0.9
                inner_lw = 0.5
            else:
                inner_fc = 'none'
                inner_ec = color
                inner_alpha = 0.8
                inner_lw = 0.8

            for i in range(min(num_draw_start, num_sub_blocks)):
                offset = i * sub_plot_size
                sb = patches.Rectangle((current_pos + offset, current_pos + offset), 
                                     sub_plot_size, sub_plot_size,
                                     linewidth=inner_lw, edgecolor=inner_ec, facecolor=inner_fc, alpha=inner_alpha)
                ax.add_patch(sb)
            
            if num_sub_blocks > num_draw_start + num_draw_end:
                start_idx = max(num_draw_start, num_sub_blocks - num_draw_end)
                for i in range(start_idx, num_sub_blocks):
                    offset = i * sub_plot_size
                    sb = patches.Rectangle((current_pos + offset, current_pos + offset), 
                                         sub_plot_size, sub_plot_size,
                                         linewidth=inner_lw, edgecolor=inner_ec, facecolor=inner_fc, alpha=inner_alpha)
                    ax.add_patch(sb)
                
                # 画中间的省略号
                center = current_pos + plot_size / 2
                ax.text(center, center, '...', fontsize=14, ha='center', va='center', color='gray', fontweight='bold')
            elif num_sub_blocks > num_draw_start:
                center = current_pos + plot_size / 2
                ax.text(center, center, '...', fontsize=14, ha='center', va='center', color='gray', fontweight='bold')
        else:
            # 普通实心块
            rect = patches.Rectangle((current_pos, current_pos), plot_size, plot_size, 
                                     linewidth=1, edgecolor='white', facecolor=color, alpha=0.9)
            ax.add_patch(rect)
        
        ticks.append(current_pos + plot_size/2)
        labels.append(str(dim))
        
        current_pos += plot_size
        
        ax.axhline(current_pos, color='#e0e0e0', linestyle='-', linewidth=0.5)
        ax.axvline(current_pos, color='#e0e0e0', linestyle='-', linewidth=0.5)

    total_size = current_pos
    ax.set_xlim(0, total_size)
    ax.set_ylim(total_size, 0) # 反转Y轴，使左上角为(0,0)，符合矩阵常理
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=14, rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlabel('Residual Dimensions', fontsize=20, labelpad=15)
    ax.xaxis.set_label_position('top') 
    
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_ylabel('Residual Dimensions', fontsize=20)

    legend_map = {
        'Marg': 'Marg (15x15)',
        'IMU': 'IMU (15x15)',
        'NHC': 'NHC (2x2)',
        'Pos': 'Pos (3x3)',
        'HMC': 'HMC (1x1)',
        'Vel': 'Vel (3x3)',
        'Reproj': 'Reprojection (nx[2x2])',
        'Pseudorange': 'Pseudorange (nx[1x1])',
        'Phaserange': 'Phaserange (nx[1x1])',
        'Doppler': 'Doppler (nx[1x1])',
        'Frequency': 'Frequency (nx[1x1])',
        'Ambiguity': 'Ambiguity (nx[1x1])'
    }

    # 按使用过的模块生成图例
    handles = []
    
    # 定义特定的顺序：先展示核心几项，再展示其他，最后展示Freq和Amb
    priority_order = ['Reproj', 'Pseudorange', 'Phaserange', 'Doppler', 'IMU', 'Marg']
    
    # 分类存储以便排序
    priority_keys = []
    other_keys = []
    delayed_keys = []
    
    # 按照特定顺序或者当前有的类型
    for k in used_keys:
        if 'Frequency' in k or 'Ambiguity' in k or 'Freq' in k or 'Amb' in k:
            delayed_keys.append(k)
        elif k in priority_order:
            priority_keys.append(k)
        else:
            other_keys.append(k)
            
    # 按照 priority_order 内部定义的顺序排序 priority_keys
    priority_keys.sort(key=lambda x: priority_order.index(x))
    # 其他的按字母排序
    other_keys.sort()
    # 延迟的按字母排序
    delayed_keys.sort()
    
    # 依次添加到 handles 中
    for k in priority_keys + other_keys + delayed_keys:
        c = colors.get(k, colors['Other'])
        handles.append(patches.Patch(color=c, label=legend_map.get(k, k)))
        
    ax.legend(handles=handles, loc='upper right', ncol=1, fontsize=16, facecolor='white', framealpha=0.9)
    
    plt.title('Covariance Matrix $C_{int}(W^{-1})$ Structure RRR', y=-0.1, fontsize=24, fontweight='bold')
    plt.tight_layout()
    
    output_img = '/home/syl/GICI-IM/results/visualization/weight_matrix_structure_rrr.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Done! Saved structured Weight matrix to: file://{output_img}")

if __name__ == "__main__":
    draw_weight_matrix_structure()