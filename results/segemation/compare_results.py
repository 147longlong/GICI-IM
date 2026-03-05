import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

def read_data(filepath):
    print(f"Reading {filepath}...")
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

def match_data_by_timestamp(df_base, df_ref):
    """
    Find rows in df_ref that have the closest timestamp to rows in df_base.
    """
    print("Matching data based on timestamps...")
    matched_rows = []
    
    # Ensure Timestamp column exists
    if 'Timestamp' not in df_base.columns or 'Timestamp' not in df_ref.columns:
        raise ValueError("Both files must contain 'Timestamp' column")

    # Sort reference for potentially faster search or just consistency
    df_ref = df_ref.sort_values('Timestamp').reset_index(drop=True)
    
    # Assuming datasets aren't massive, standard loop is fine. 
    # For very large datasets, merge_asof is better.
    matches = pd.merge_asof(df_base.sort_values('Timestamp'), 
                            df_ref.sort_values('Timestamp'), 
                            on='Timestamp', 
                            direction='nearest',
                            suffixes=('_base', '_ref'))
    
    return matches

def plot_pl_comparison(df_matched, output_path):
    print(f"Plotting PL comparison to {output_path}...")
    
    # Use relative time for x-axis
    t = df_matched['Timestamp'] - df_matched['Timestamp'].iloc[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 20), sharex=True)
    
    # Check headers - handle potential suffixes from merge
    # If standard names were used
    lapl_base = df_matched.get('XPL(m)_base', df_matched.get('XPL(m)'))
    lapl_ref = df_matched.get('XPL(m)_ref')
    
    lopl_base = df_matched.get('YPL(m)_base', df_matched.get('YPL(m)'))
    lopl_ref = df_matched.get('YPL(m)_ref')
    
    vpl_base = df_matched.get('VPL(m)_base', df_matched.get('VPL(m)'))
    vpl_ref = df_matched.get('VPL(m)_ref')

    # XPL
    axes[0].plot(t, lapl_base, label='super meas', color='#1f77b4', marker='o', linestyle='', markersize=10)
    axes[0].plot(t, lapl_ref, label='raw', color='#ff7f0e', marker='x', linestyle='', markersize=12, markeredgewidth=2.5)
    axes[0].set_ylabel('LaPL (m)')
    axes[0].legend(loc='upper right')
    axes[0].set_title('LaPL Comparison')
    
    # YPL
    axes[1].plot(t, lopl_base, label='super meas', color='#1f77b4', marker='o', linestyle='', markersize=10)
    axes[1].plot(t, lopl_ref, label='raw', color='#ff7f0e', marker='x', linestyle='', markersize=12, markeredgewidth=2.5)
    axes[1].set_ylabel('LoPL (m)')
    axes[1].set_title('LoPL Comparison')
    
    # VPL
    axes[2].plot(t, vpl_base, label='super meas', color='#1f77b4', marker='o', linestyle='', markersize=10)
    axes[2].plot(t, vpl_ref, label='raw', color='#ff7f0e', marker='x', linestyle='', markersize=12, markeredgewidth=2.5)
    axes[2].set_ylabel('VPL (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('VPL Comparison')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_performance_comparison(df_matched, output_path):
    print(f"Plotting performance comparison to {output_path}...")
    
    t = df_matched['Timestamp'] - df_matched['Timestamp'].iloc[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 16), sharex=True)
    
    subsets_base = df_matched.get('Subsets_base', df_matched.get('Subsets'))
    subsets_ref = df_matched.get('Subsets_ref')
    
    time_base = df_matched.get('TimeTaken(s)_base', df_matched.get('TimeTaken(s)'))
    time_ref = df_matched.get('TimeTaken(s)_ref')

    # Subsets
    axes[0].plot(t, subsets_base, label='super meas', color='#2ca02c', marker='o', linestyle='', markersize=10)
    axes[0].plot(t, subsets_ref, label='raw', color='#d62728', marker='x', linestyle='', markersize=12, markeredgewidth=2.5)
    axes[0].set_ylabel('Number of Subsets')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Subsets Count Comparison')
    # Using log scale if differences are huge
    if subsets_ref.max() > 10 * subsets_base.max() or subsets_base.max() > 10 * subsets_ref.max():
         axes[0].set_yscale('log')
         axes[0].set_ylabel('Number of Subsets (Log Scale)')
    
    # Time Taken
    axes[1].plot(t, time_base, label='super meas', color='#2ca02c', marker='o', linestyle='', markersize=10)
    axes[1].plot(t, time_ref, label='raw', color='#d62728', marker='x', linestyle='', markersize=12, markeredgewidth=2.5)
    axes[1].set_ylabel('Time Taken (s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Computation Time Comparison')
    if time_ref.max() > 10 * time_base.max() or time_base.max() > 10 * time_ref.max():
         axes[1].set_yscale('log')
         axes[1].set_ylabel('Time Taken (s) (Log Scale)')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = '/home/syl/GICI-IM/results'
    file1_path = os.path.join(base_dir, 'results_1.txt')
    file2_path = os.path.join(base_dir, 'results_2.txt')
    
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: {file1_path} not found.")
        return
    if not os.path.exists(file2_path):
        print(f"Error: {file2_path} not found.")
        return

    df1 = read_data(file1_path) # results_1 (Ref)
    df2 = read_data(file2_path) # results_2 (Base)
    
    if df1 is None or df2 is None:
        return

    # Filter data: exclude last point of super meas (df2) and keep only first 5 seconds
    if not df2.empty:
        print("Filtering super meas data: removing last point and limiting to 5 seconds.")
        df2 = df2.iloc[:-1] # Remove last point
        start_time = df2['Timestamp'].min()
        df2 = df2[df2['Timestamp'] - start_time <= 5.0]

    # Match: results_2 is the base, find corresponding in results_1
    try:
        df_matched = match_data_by_timestamp(df2, df1)
        
        # Plot
        plot_pl_comparison(df_matched, os.path.join(base_dir, 'comparison_pl.png'))
        plot_performance_comparison(df_matched, os.path.join(base_dir, 'comparison_performance.png'))
        
        print("Comparison plotting complete.")
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
