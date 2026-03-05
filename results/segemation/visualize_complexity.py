import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import matplotlib.colors as mcolors

# 设置文件路径
file_path_12 = '/home/syl/GICI-IM/results/subset_info_1e_12.txt'
file_path_9 = '/home/syl/GICI-IM/results/subset_info_super.txt'
output_dir = '/home/syl/GICI-IM/results/'

def load_data_from_file(path):
    d = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return np.array([])
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 6: continue
                d.append([float(x) for x in parts])
        return np.array(d)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.array([])

# 加载数据
data_12 = load_data_from_file(file_path_12)
data_9 = load_data_from_file(file_path_9)

# 默认使用 1e_12 的数据用于图 1 和 图 2 (保持原有逻辑)
# data = data_12 if data_12.size > 0 else data_9
data = data_9

if data.size == 0:
    print("No data found in provided files.")
    exit()


# 提取各列数据
time = data[:, 0]
time = time - time[0] # 使用相对时间，从0开始
num_meas = data[:, 1]
n_fault_max = data[:, 2]
subsetsize = data[:, 3]
num_residual = data[:, 4]
num_state_vars = data[:, 5]

# 计算理论计算复杂度
complexity = subsetsize * (num_residual**3 + num_state_vars * (num_residual**2))

# --- 绘图参数设置 ---
FONT_SIZE_GLOBAL = 16
FONT_SIZE_TITLE = 35
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

# --- 图表 1: 优化变量维度与理论计算代价 (合并) ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 18))

# 子图 1: 优化变量维度
ax1.plot(time, num_residual, label='Residuals ($n$)', color='#9467bd')
ax1.plot(time, num_state_vars, label='State Variables ($m$)', color='#ff7f0e')
ax1.set_ylabel('Dimension', fontsize=FONT_SIZE_LABEL)
ax1.set_ylim(bottom=250, top=2500)
ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=FONT_SIZE_LEGEND)
ax1.set_title('Optimization Dimensions over Time', fontsize=FONT_SIZE_TITLE)
ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
ax1.grid(True, linestyle='--', alpha=0.7)

# 子图 2: 理论计算代价
ax2.plot(time, complexity, color='#d62728')
ax2.set_yscale('log')
ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Theoretical Operations', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Theoretical Computational Complexity\n($N_{subsets}x(𝑛^3+𝑚×𝑛^2)$)', fontsize=FONT_SIZE_TITLE)
ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
ax2.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
save_path1 = os.path.join(output_dir, 'dimensions_and_complexity.png')
plt.savefig(save_path1, dpi=300)
print(f"Generated: {save_path1}")


# --- 图表 2: 量测数 vs 子集数 (改进版) ---
# 统计频次
points = list(zip(num_meas, subsetsize, n_fault_max))
counts = Counter(points)
total_points = len(points)

# 解压唯一数据点和对应的频次
unique_data = []
for point, count in counts.items():
    unique_data.append(point + (count,))

unique_data = np.array(unique_data)
u_num_meas = unique_data[:, 0]
u_subsetsize = unique_data[:, 1]
u_n_fault = unique_data[:, 2]
u_freq = unique_data[:, 3] / total_points # 计算相对频率

fig2, ax3 = plt.subplots(figsize=(14, 12))

# 定义形状映射
markers = ['^', '*', 's', 'D', 'p', '*', 'h'] #'o', 
unique_faults = np.unique(u_n_fault)
unique_faults.sort()

# 归一化频次用于颜色映射
norm = mcolors.Normalize(vmin=np.min(u_freq), vmax=np.max(u_freq))
cmap = plt.cm.Reds # 使用红色系渐变，颜色越深代表频次越高

# 循环绘制不同形状
sc_list = []
for i, fault_val in enumerate(unique_faults):
    mask = (u_n_fault == fault_val)
    
    x = u_num_meas[mask]
    y = u_subsetsize[mask]
    c = u_freq[mask]
    
    marker = markers[i % len(markers)]
    
    sc = ax3.scatter(x, y, c=c, cmap=cmap, norm=norm, marker=marker, s=180, 
                    label=f'$N_{{fault,max}}={int(fault_val)}$', edgecolors='k', alpha=0.9, zorder=10-i)
    sc_list.append(sc)

# 添加颜色条
cbar = plt.colorbar(sc_list[-1], ax=ax3)
cbar.set_label('Relative Frequency', fontsize=FONT_SIZE_LABEL)
cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND)

ax3.set_yscale('log')
ax3.set_xlabel('Number of Measurements ($N_{meas}$)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Number of Subsets ($N_{subsets}$)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('$N_{meas}$, $N_{fault,max}$, and $N_{subsets}$ with $P_{thres} = 1x10^{-9}$', fontsize=FONT_SIZE_TITLE)
ax3.legend(loc='upper left', frameon=True, framealpha=0.9, borderpad=1)

ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
ax3.grid(True, which="both", axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
save_path2 = os.path.join(output_dir, 'measurements_vs_subsets.png')
plt.savefig(save_path2, dpi=100)
print(f"Generated: {save_path2}")

print("All plots generated successfully.")

# --- 图表 3: 1e-9 与 1e-12 阈值对比 (量测数 vs 子集数) ---
# 合并数据
if data_9.size > 0 and data_12.size > 0:
    data_combined = np.vstack((data_12, data_9))
elif data_12.size > 0:
    data_combined = data_12
else:
    data_combined = data_9

if data_combined.size > 0:
    # 提取合并数据的列
    c_num_meas = data_combined[:, 1]
    c_n_fault_max = data_combined[:, 2]
    c_subsetsize = data_combined[:, 3]
    
    # 统计频次
    c_points = list(zip(c_num_meas, c_subsetsize, c_n_fault_max))
    c_counts = Counter(c_points)
    c_total_points = len(c_points)
    
    # 解压
    c_unique_data = []
    for point, count in c_counts.items():
        c_unique_data.append(point + (count,))
    
    c_unique_data = np.array(c_unique_data)
    cu_num_meas = c_unique_data[:, 0]
    cu_subsetsize = c_unique_data[:, 1]
    cu_n_fault = c_unique_data[:, 2]
    cu_freq = c_unique_data[:, 3] / c_total_points
    
    fig3, ax4 = plt.subplots(figsize=(18, 12))
    
    # 定义形状
    c_unique_faults = np.unique(cu_n_fault)
    c_unique_faults.sort()
    
    # 颜色
    c_norm = mcolors.Normalize(vmin=np.min(cu_freq), vmax=np.max(cu_freq))
    c_cmap = plt.cm.Reds # 使用红色系

    sc_list_c = []
    for i, fault_val in enumerate(c_unique_faults):
        mask = (cu_n_fault == fault_val)
        
        x = cu_num_meas[mask]
        y = cu_subsetsize[mask]
        c = cu_freq[mask]
        
        marker = markers[i % len(markers)]
        
        sc = ax4.scatter(x, y, c=c, cmap=c_cmap, norm=c_norm, marker=marker, s=180, 
                        label=f'$N_{{fault,max}}={int(fault_val)}$', edgecolors='k', alpha=0.9, zorder=10-i)
        sc_list_c.append(sc)
    
    # Colorbar
    cbar3 = plt.colorbar(sc_list_c[-1], ax=ax4)
    cbar3.set_label('Relative Frequency', fontsize=FONT_SIZE_LABEL)
    cbar3.ax.tick_params(labelsize=FONT_SIZE_LEGEND)
    
    ax4.set_yscale('log')
    ax4.set_xlabel('Number of Measurements ($N_{meas}$)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Number of Subsets ($N_{subsets}$)', fontsize=FONT_SIZE_LABEL)
    # 有无超量测分割下
    ax4.set_title('$N_{meas}$, $N_{fault,max}$, and $N_{subsets}$\n with/without Super-measurement', fontsize=FONT_SIZE_TITLE) #\nwith $P_{thres}=10^{-9}$ vs $10^{-12}$
    ax4.legend(loc='upper left', frameon=True, framealpha=0.9, borderpad=1)
    
    ax4.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax4.grid(True, which="both", axis='y', linestyle='--', alpha=0.5)
    
    save_path3 = os.path.join(output_dir, 'measurements_vs_subsets_combined.png')
    plt.savefig(save_path3, dpi=300)
    print(f"Generated: {save_path3}")
