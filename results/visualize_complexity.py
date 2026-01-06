import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import matplotlib.colors as mcolors

# 设置文件路径
file_path = '/home/dell/sunyulong/GICI-IM/results/subset_info_1e_12.txt'
output_dir = '/home/dell/sunyulong/GICI-IM/results/'

# 读取数据
data = []
try:
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # 格式: time, num_meas, N_fault_max, subsetsize, num_residual, num_state_vars
            data.append([float(x) for x in parts])
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

data = np.array(data)

if data.shape[0] == 0:
    print("No data found in subset_info.txt")
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
markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'h']
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
ax3.set_title('$N_{meas}$, $N_{fault,max}$, and $N_{subsets}$', fontsize=FONT_SIZE_TITLE)
ax3.legend(title='Max Faults ($N_{fault,max}$)', title_fontsize=FONT_SIZE_LEGEND_TITLE, fontsize=FONT_SIZE_LEGEND, loc='upper left', frameon=True, framealpha=0.9, borderpad=1)

ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
ax3.grid(True, which="both", axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
save_path2 = os.path.join(output_dir, 'measurements_vs_subsets.png')
plt.savefig(save_path2, dpi=100)
print(f"Generated: {save_path2}")

print("All plots generated successfully.")
