import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_weight_matrix_structure():
    # 设置风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # 画布设置
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- 参数定义 ---
    block_size = 0.5      # 块的大小 (宽=高)
    
    # 颜色定义 (保持一致)
    colors = {
        'Marg': "#A34FE7",      
        'IMU': "#f16d6d6c",       
        'Reproj-Pose': "#baf09a",    # 棕色：重投影 (Pose)
        'Reproj-LM': "#cff3b7",      # 浅色：重投影 (Landmark)
        'NHC': '#CBE8FF',       
        'Pos': "#e3ab77",       
        'HMC': '#FFBEC5',       
        'Vel': '#FFFFCC'        
    }

    # 数据定义
    # 按照用户要求：
    # 1. 恢复所有误差块的显示
    # 2. 放大最后三个重投影误差块（详细画出内部结构）
    # 3. 前面的重投影误差块正常显示（不画内部细节）
    # 4. 省略中间部分矩阵块，用省略号代替
    # 5. 前1/2不省略
    rows_data = [
        ('IMU', [5, 6], 15),
        ('Reproj', [5, 17], 90),
        ('IMU', [10, 11], 15),
        ('Reproj', [10, 17], 126),
        ('IMU', [12, 13], 15),
        ('Reproj', [12, 17], 118),
        ('IMU', [13, 14], 15),
        ('Reproj', [13, 17], 134),
        ('IMU', [14, 15], 15),
        ('Reproj', [14, 17], 122),
        ('IMU', [15, 16], 15),
        ('Reproj', [15, 17], 92),
        ('Reproj', [16, 17], 82),
        
        ('HMC', [0], 1),
        ('IMU', [0, 1], 15),
        ('Marg', [0], 15),
        ('NHC', [0], 2),
        ('Pos', [0], 3),
        ('Vel', [0], 3),
        
        ('HMC', [1], 1),
        ('IMU', [1, 2], 15),
        ('NHC', [1], 2),
        ('Pos', [1], 3),
        ('Vel', [1], 3),
        
        # ('HMC', [2], 1),
        # ('IMU', [2, 3], 15),
        # ('NHC', [2], 2),
        # ('Pos', [2], 3),
        # ('Vel', [2], 3),
        
        # ('HMC', [3], 1),
        # ('IMU', [3, 4], 15),
        # ('NHC', [3], 2),
        # ('Pos', [3], 3),
        # ('Vel', [3], 3),
        
        # 省略中间部分
        ('ELLIPSIS', [], 0),
        
        # ('HMC', [9], 1),
        # ('IMU', [9, 10], 15),
        # ('NHC', [9], 2),
        # ('Pos', [9], 3),
        # ('Vel', [9], 3),
        
        ('HMC', [11], 1),
        ('IMU', [11, 12], 15),
        ('NHC', [11], 2),
        ('Pos', [11], 3),
        ('Vel', [11], 3),
    ]

    current_pos = 0
    ticks = []
    labels = []
    
    # Grid lines start
    ax.axhline(0, color='#e0e0e0', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='#e0e0e0', linestyle='-', linewidth=0.5)

    # Count total Reproj blocks to identify the last 3
    total_reproj = sum(1 for r in rows_data if r[0] == 'Reproj')
    reproj_count = 0
    
    # Scale factor to convert dimensions to plot units
    # 90 dim -> 9.0 units
    scale_factor = 0.1 
    
    # Standard size for non-enlarged blocks (same as IMU block size)
    # IMU dim is 15, so standard visual size corresponds to dim 15
    standard_visual_dim = 15 

    for type_name, _, dim in rows_data:
        
        if type_name == 'ELLIPSIS':
            # Draw ellipsis
            ellipsis_size = standard_visual_dim * scale_factor
            center = current_pos + ellipsis_size / 2
            ax.text(center, center, '...', fontsize=20, ha='center', va='center', color='black', fontweight='bold')
            
            # Add a small gap for ellipsis
            current_pos += ellipsis_size
            continue

        if type_name == 'Reproj':
            reproj_count += 1
            
            # Check if it's one of the last 3 Reproj blocks
            if reproj_count > (total_reproj - 3):
                # Use actual dimension for the last 3 Reproj blocks
                visual_dim = dim
                plot_size = visual_dim * scale_factor
                
                # Draw outer frame for the large block
                rect = patches.Rectangle((current_pos, current_pos), plot_size, plot_size, 
                                         linewidth=1, edgecolor=colors['Reproj-Pose'], facecolor=colors['Reproj-LM'], alpha=0.3)
                ax.add_patch(rect)
                
                # Draw internal 3x3 blocks to show structure
                sub_dim = 8
                sub_plot_size = sub_dim * scale_factor
                
                # Calculate how many sub-blocks fit
                num_sub_blocks = dim // sub_dim
                
                # Draw a few at the start (top-left)
                num_draw_start = 4
                for i in range(min(num_draw_start, num_sub_blocks)):
                    offset = i * sub_plot_size
                    sb = patches.Rectangle((current_pos + offset, current_pos + offset), 
                                         sub_plot_size, sub_plot_size,
                                         linewidth=0.2, edgecolor='white', facecolor=colors['Reproj-Pose'])
                    ax.add_patch(sb)
                
                # Draw a few at the end (bottom-right)
                num_draw_end = 4
                if num_sub_blocks > num_draw_start:
                    start_idx = max(num_draw_start, num_sub_blocks - num_draw_end)
                    for i in range(start_idx, num_sub_blocks):
                        offset = i * sub_plot_size
                        sb = patches.Rectangle((current_pos + offset, current_pos + offset), 
                                             sub_plot_size, sub_plot_size,
                                             linewidth=0.2, edgecolor='white', facecolor=colors['Reproj-Pose'])
                        ax.add_patch(sb)
                
                # Add ellipsis if there's a gap
                if num_sub_blocks > (num_draw_start + num_draw_end):
                    center = current_pos + plot_size / 2
                    ax.text(center, center, '...', fontsize=14, ha='center', va='center', color=colors['Reproj-Pose'], fontweight='bold')
            else:
                # For earlier Reproj blocks, use standard size
                visual_dim = standard_visual_dim
                plot_size = visual_dim * scale_factor
                
                # Draw simple block
                rect = patches.Rectangle((current_pos, current_pos), plot_size, plot_size, 
                                         linewidth=1, edgecolor=colors['Reproj-Pose'], facecolor=colors['Reproj-LM'], alpha=0.3)
                ax.add_patch(rect)

        else:
            # For all other types (IMU, HMC, etc.), use standard size
            visual_dim = standard_visual_dim
            plot_size = visual_dim * scale_factor
            
            color = colors[type_name]
            # Draw solid block
            rect = patches.Rectangle((current_pos, current_pos), plot_size, plot_size, 
                                     linewidth=0.5, edgecolor='white', facecolor=color, alpha=0.9)
            ax.add_patch(rect)
        
        # Ticks
        ticks.append(current_pos + plot_size/2)
        # Label all blocks
        labels.append(str(dim))
        
        current_pos += plot_size
        
        # Draw grid lines for this block end
        ax.axhline(current_pos, color='#e0e0e0', linestyle='-', linewidth=0.5)
        ax.axvline(current_pos, color='#e0e0e0', linestyle='-', linewidth=0.5)

    # Axes settings
    total_size = current_pos
    ax.set_xlim(0, total_size)
    ax.set_ylim(total_size, 0) # Invert Y to match matrix layout (0,0 at top-left)
    
    # Labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlabel('Residual Dimensions', fontsize=14, labelpad=10)
    ax.xaxis.set_label_position('top') 
    
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_ylabel('Residual Dimensions', fontsize=14)

    # Legend
    legend_map = {
        'Marg': 'Marg (15x15)',
        'IMU': 'IMU (15x15)',
        'Reproj-Pose': 'Reproj (nx[2x2])', # Use generic name for W
        'NHC': 'NHC (2x2) ',
        'Pos': 'Pos (3x3)',
        'HMC': 'HMC (1x1)',
        'Vel': 'Vel (3x3)'
    }
    # Filter colors used in W
    used_keys = ['Marg', 'IMU', 'Reproj-Pose', 'NHC', 'Pos', 'HMC', 'Vel']
    handles = [patches.Patch(color=colors[k], label=legend_map.get(k, k)) for k in used_keys]
    # Increase fontsize and handleheight/length for larger legend
    ax.legend(handles=handles, loc='lower left', ncol=1, fontsize=14, frameon=True, handleheight=2, handlelength=3)
    
    plt.title('Covariance Matrix $C_{int}(W^{-1})$ Structure', y=-0.05, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weight_matrix_structure.png', dpi=300)
    print("Weight matrix structure visualization saved as 'weight_matrix_structure.png'.")

if __name__ == "__main__":
    draw_weight_matrix_structure()
