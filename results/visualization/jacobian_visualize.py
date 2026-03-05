import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_jacobian_structure():
    # 设置风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # 画布设置
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # --- 参数定义 ---
    num_frames = 17
    col_width_pose = 1.0  # 状态变量列宽
    col_width_lm = 3.0    # 路标点列宽
    row_height = 0.5      # 行高
    
    # 颜色定义
    colors = {
        'Marg': "#A34FE7",      # 灰色：边缘化
        'IMU': "#f16d6d6c",       # 蓝色：IMU
        'Reproj-Pose': "#cfe4c3",    # 棕色：重投影 (Pose)
        'Reproj-LM': "#eaf4e3",      # 浅色：重投影 (Landmark)
        'NHC': '#CBE8FF',       # 紫色：NHC
        'Pos': "#e3ab77",       # 粉色：位置误差
        'HMC': '#FFBEC5',       # 深绿色：HMC
        'Vel': '#FFFFCC'        # 深绿色：速度误差
    }

    # 数据定义 (根据提供的表格)
    # 格式: (Type, [Columns], Dimension)
    # Columns: 0-16 为 Frames, 17 为 Landmarks
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
        
        ('HMC', [2], 1),
        ('IMU', [2, 3], 15),
        ('NHC', [2], 2),
        ('Pos', [2], 3),
        ('Vel', [2], 3),
        
        ('HMC', [3], 1),
        ('IMU', [3, 4], 15),
        ('NHC', [3], 2),
        ('Pos', [3], 3),
        ('Vel', [3], 3),
        
        ('HMC', [4], 1),
        ('IMU', [4, 5], 15),
        ('NHC', [4], 2),
        ('Pos', [4], 3),
        ('Vel', [4], 3),
        
        ('HMC', [6], 1),
        ('IMU', [6, 7], 15),
        ('NHC', [6], 2),
        ('Pos', [6], 3),
        ('Vel', [6], 3),
        
        ('IMU', [7, 8], 15),
        ('Pos', [7], 3),
        ('Vel', [7], 3),
        
        ('HMC', [8], 1),
        ('IMU', [8, 9], 15),
        ('NHC', [8], 2),
        ('Pos', [8], 3),
        ('Vel', [8], 3),
        
        ('HMC', [9], 1),
        ('IMU', [9, 10], 15),
        ('NHC', [9], 2),
        ('Pos', [9], 3),
        ('Vel', [9], 3),
        
        ('HMC', [11], 1),
        ('IMU', [11, 12], 15),
        ('NHC', [11], 2),
        ('Pos', [11], 3),
        ('Vel', [11], 3),
    ]

    current_y = 0
    y_ticks = []
    y_labels = []
    
    # 辅助函数：画块
    def add_block(row_y, col_idx, type_name):
        # Determine color key
        if type_name == 'Reproj':
            color_key = 'Reproj-LM' if col_idx == 17 else 'Reproj-Pose'
        else:
            color_key = type_name

        if col_idx == 17: # Landmarks
            x_start = num_frames * col_width_pose
            w_total = col_width_lm
            
            if type_name == 'Reproj':
                # 特殊处理：Landmarks列的Reproj显示为稀疏小块
                light_color = colors['Reproj-LM'] # 浅色
                
                # 绘制多个对角线分布的小块
                sw = 0.4 # 小块宽度
                sh = row_height * 0.2 # 小块高度
                
                # 1. 左上
                rect1 = patches.Rectangle((x_start + 0.2, row_y + 0.05), sw, sh, 
                                         linewidth=0.5, edgecolor='#aaaaaa', facecolor=light_color)
                ax.add_patch(rect1)
                
                # 2. 左中
                rect2 = patches.Rectangle((x_start + w_total/3 - sw/2, row_y + row_height/3 - sh/2), sw, sh, 
                                         linewidth=0.5, edgecolor='#aaaaaa', facecolor=light_color)
                ax.add_patch(rect2)

                # 3. 中间
                rect3 = patches.Rectangle((x_start + w_total/2 - sw/2, row_y + row_height/2 - sh/2), sw, sh, 
                                         linewidth=0.5, edgecolor='#aaaaaa', facecolor=light_color)
                ax.add_patch(rect3)

                # 4. 右中
                rect4 = patches.Rectangle((x_start + 2*w_total/3 - sw/2, row_y + 2*row_height/3 - sh/2), sw, sh, 
                                         linewidth=0.5, edgecolor='#aaaaaa', facecolor=light_color)
                ax.add_patch(rect4)
                
                # 5. 右下
                rect5 = patches.Rectangle((x_start + w_total - sw - 0.2, row_y + row_height - sh - 0.05), sw, sh, 
                                         linewidth=0.5, edgecolor='#aaaaaa', facecolor=light_color)
                ax.add_patch(rect5)
                
                # 省略号
                ax.text(x_start + w_total/4, row_y + row_height/4, '...', 
                        ha='center', va='center', fontsize=8, color='gray')
                ax.text(x_start + 3*w_total/4, row_y + 3*row_height/4, '...', 
                        ha='center', va='center', fontsize=8, color='gray')
                
                # 虚线边框表示范围
                rect_border = patches.Rectangle((x_start, row_y), w_total, row_height, 
                                         linewidth=0.5, edgecolor='#d0d0d0', facecolor='none', linestyle=':')
                ax.add_patch(rect_border)
                
            else:
                x = x_start
                w = w_total
                rect = patches.Rectangle((x, row_y), w, row_height, 
                                         linewidth=0.5, edgecolor='white', facecolor=colors[color_key], alpha=0.9)
                ax.add_patch(rect)
        else:
            x = col_idx * col_width_pose
            w = col_width_pose
            rect = patches.Rectangle((x, row_y), w, row_height, 
                                     linewidth=0.5, edgecolor='white', facecolor=colors[color_key], alpha=0.9)
            ax.add_patch(rect)

    # 绘图循环
    for type_name, cols, dim in rows_data:
        for col in cols:
            add_block(current_y, col, type_name)
        
        # 记录Y轴标签
        y_ticks.append(current_y + row_height/2)
        y_labels.append(str(dim))
        
        current_y += row_height

    # ================= 坐标轴设置 =================
    total_width = num_frames * col_width_pose + col_width_lm
    ax.set_xlim(0, total_width)
    ax.set_ylim(current_y, 0) # 反转Y轴
    
    # X轴刻度
    x_ticks = [i * col_width_pose + col_width_pose/2 for i in range(num_frames)] + [num_frames * col_width_pose + col_width_lm/2]
    x_labels = ['gPose0', 'gPose1', 'gPose2', 'gPose3', 'gPose4', 'cPose5', 
                'gPose6', 'gPose7', 'gPose8', 'gPose9', 'cPose10', 
                'gPose11', 'cPose12', 'cPose13', 'cPose14', 'cPose15', 'cPose16', 'Landmarks']
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='left')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel('State Variables (Columns)', fontsize=12, labelpad=10)
    
    # Y 轴刻度
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylabel('Residual Dimensions (Rows)', fontsize=12)
    
    # 竖直网格线
    for i in range(num_frames + 1):
        ax.axvline(i * col_width_pose, color='#e0e0e0', linestyle='-', linewidth=0.5)
    ax.axvline(num_frames * col_width_pose, color='black', linestyle='-', linewidth=1.0) # 分隔路标点
    
    # 水平网格线 (每行)
    for i in range(len(rows_data) + 1):
        ax.axhline(i * row_height, color='#e0e0e0', linestyle='-', linewidth=0.5)

    # 图例
    legend_map = {
        'Marg': 'Marg (15x15)',
        'IMU': 'IMU (15x15)',
        'Reproj-Pose': 'Reproj-Pose (2nx6)',
        'Reproj-LM': 'Reproj-LM (nx[2x3])',
        'NHC': 'NHC (2x15)',
        'Pos': 'Pos (3x6)',
        'HMC': 'HMC (1x15)',
        'Vel': 'Vel (3x15)'
    }
    handles = [patches.Patch(color=colors[k], label=legend_map.get(k, k)) for k in colors]
    ax.legend(handles=handles, loc='lower right', ncol=1, fontsize=12, frameon=True, handleheight=2, handlelength=3)    
    
    plt.title('Jacobian Matrix Sparsity Structure', y=-0.05, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('jacobian_structure.png', dpi=300)
    plt.show()
    print("Jacobian structure visualization saved as 'jacobian_structure.png'.")

if __name__ == "__main__":
    draw_jacobian_structure()