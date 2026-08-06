import sys
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_jacobian_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if "Residual Type" in line:
            header_idx = i
            break
    
    if header_idx == -1:
        print("Cannot find header")
        return None, None, None
        
    header_line = lines[header_idx].rstrip('\n')
    cols_part = header_line[35:]
    pad = 21 - (len(cols_part) % 21)
    if pad != 21: cols_part += ' ' * pad
    num_cols = len(cols_part) // 21
    
    col_types = [cols_part[i*21:(i+1)*21].strip() for i in range(num_cols)]
    
    time_line = lines[header_idx+2].rstrip('\n')
    col_times = []
    if len(time_line) > 35 and ('-' in time_line or '.' in time_line):
        time_part = time_line[35:]
        pad = 21 - (len(time_part) % 21)
        if pad != 21: time_part += ' ' * pad
        for i in range(num_cols):
            if i*21 < len(time_part): col_times.append(time_part[i*21:(i+1)*21].strip())
            else: col_times.append("")
    else:
        col_times = [""] * num_cols

    # 1. Condense specific columns: Frequency, Ambiguity, Landmarks
    condensed_cols = []
    col_map = {}
    
    i = 0
    pose_idx = 0
    while i < num_cols:
        c = col_types[i]
        is_freq = 'Frequency' in c
        is_amb = 'Ambiguity' in c
        is_lm = 'Landmarks' in c
        
        if is_freq or is_amb or is_lm:
            j = i
            while j < num_cols and (
                (is_freq and 'Frequency' in col_types[j]) or
                (is_amb and 'Ambiguity' in col_types[j]) or
                (is_lm and 'Landmarks' in col_types[j])):
                j += 1
            
            c_name = 'Frequencies' if is_freq else ('Ambiguities' if is_amb else 'Landmarks')
            col_type_tag = 'Frequency' if is_freq else ('Ambiguity' if is_amb else 'Landmark')
            condensed_cols.append({
                'name': c_name,
                'type': col_type_tag,
                'orig_indices': list(range(i, j)),
                'is_sparse': True,
                'width': 4.0 if is_freq else 4.5,
                'time': ''
            })
            for k in range(i, j): col_map[k] = len(condensed_cols) - 1
            i = j
        else:
            c_name = c
            if 'cPose' in c: c_name = f"cPose{pose_idx}"
            elif 'gPose' in c: c_name = f"gPose{pose_idx}"
            
            condensed_cols.append({
                'name': c_name,
                'type': 'Pose' if 'Pose' in c else c,
                'orig_indices': [i],
                'is_sparse': False,
                'width': 1.5,  # 强制拉宽！
                'time': col_times[i]
            })
            col_map[i] = len(condensed_cols) - 1
            if 'Pose' in c: pose_idx += 1
            i += 1

    # 2. Process Rows & Group vertically
    residual_groups = {}
    residual_patterns = []
    
    for line in lines[header_idx+4:]:
        line = line.rstrip('\n')
        if not line or line.startswith('---') or line.startswith('==='): continue
        row_name_raw = line[0:35].strip()
        if not row_name_raw: continue
        
        # 针对用户不想画具体数值（如 [614,614] 那样的数据行）的诉求，只保留带有 'Error' 的雅可比结构行
        if 'Error' not in row_name_raw:
            continue
            
        cells_str = line[35:]
        pad = 21 - (len(cells_str) % 21)
        if pad != 21: cells_str += ' ' * pad
        
        active_orig_cols = []
        for i in range(num_cols):
            if i*21 >= len(cells_str): break
            cell = cells_str[i*21:(i+1)*21].strip()
            if set(cell) - set('- ') != set(): active_orig_cols.append(i)
                
        # Handle spaces in names like "IMU Error(1x15)" better:
        row_name_full = line.split("  ")[0] if "  " in line else line.split()[0]
        
        # Base residual type
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
            
        # Parse dim from the full token which isn't truncated
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

        # VERY IMPORTANT: Only use active "Pose" columns (time states) to determine groupings
        active_state_cols = set()
        active_all_condensed = set()
        
        for c_orig in active_orig_cols:
            if c_orig in col_map: 
                cond_idx = col_map[c_orig]
                active_all_condensed.add(cond_idx)
                if condensed_cols[cond_idx]['type'] == 'Pose':
                    active_state_cols.add(cond_idx)
                    
        active_state_cols = tuple(sorted(list(active_state_cols)))
        active_all_condensed = tuple(sorted(list(active_all_condensed)))
        
        key = (base, active_state_cols)
        
        if key not in residual_groups:
            residual_groups[key] = {
                'count': 1,
                'dim': row_dim,
                'active_cols': set(active_all_condensed)
            }
            residual_patterns.append(key)
        else:
            residual_groups[key]['count'] += 1
            residual_groups[key]['dim'] += row_dim
            residual_groups[key]['active_cols'].update(active_all_condensed)
            
    return condensed_cols, residual_patterns, residual_groups

def draw_jacobian_structure():
    filepath = '/home/syl/GICI-IM/results/visualization/jacobian_visualization-rrr.txt'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    condensed_cols, residual_patterns, residual_groups = parse_jacobian_file(filepath)
    
    if not condensed_cols: return
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    num_cols = len(condensed_cols)
    num_rows = len(residual_patterns)
    
    x_positions = [0]
    for c in condensed_cols:
        x_positions.append(x_positions[-1] + c['width'])
        
    total_width = x_positions[-1]
    row_height = 0.5
    total_height = num_rows * row_height
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 颜色配置扩展：区分不同 GNSS 残差的不同变量块
    colors = {
        'Marg': "#A34FE7",
        'IMU': "#f16d6d6c", 
        'NHC': '#CBE8FF',
        'Pos': "#e3ab77",
        'HMC': '#FFBEC5',       # 深绿色：HMC
        'Vel': '#FFFFCC',
        
        'Reproj-Pose': "#cfe4c3",
        'Reproj-LM': "#eaf4e3",  
        
        # 伪距
        'Pseudorange-Pose': "#4292C6",      
        'Pseudorange-Freq': "#9ECAE1",      
        'Pseudorange-Ambiguity': "#C6DBEF", 
        
        # 载波相位 (改为紫色系，完美避开带有粉色/红色的 IMU)
        'Phaserange-Pose': "#807DBA",      
        'Phaserange-Freq': "#9E9AC8",      
        'Phaserange-Ambiguity': "#BCBDDC", 
        
        # 多普勒
        'Doppler-Pose': "#E6D7B5", 
        'Doppler-Freq': "#E6D7B5DA", 
        'Doppler-Ambiguity': "#E5F5E0",

        'Other': "#cccccc",
        'Other-Sparse': "#e0e0e0"
    }
    
    y_ticks = []
    y_labels = []
    current_y = 0
    
    # Track used legend keys to only display what's drawn
    used_legend_keys = set()
    
    for key in residual_patterns:
        base_type, _ = key
        group_data = residual_groups[key]
        count = group_data['count']
        dim = group_data['dim']
        active_condensed = group_data['active_cols']
        
        for col_idx in active_condensed:
            c_info = condensed_cols[col_idx]
            x_start = x_positions[col_idx]
            w_total = c_info['width']
            is_sparse = c_info['is_sparse']
            col_type = c_info['type']
            
            # 决定具体色块颜色
            color_key = base_type
            if base_type == 'Reproj':
                if 'Landmark' in col_type: color_key = 'Reproj-LM'
                else: color_key = 'Reproj-Pose'
            elif base_type in ['Pseudorange', 'Phaserange', 'Doppler']:
                if 'Frequency' in col_type: color_key = f'{base_type}-Freq'
                elif 'Ambiguity' in col_type: color_key = f'{base_type}-Ambiguity'
                else: color_key = f'{base_type}-Pose'
            elif is_sparse:
                # Do not group into generic Other-Sparse if it has a real name
                if 'Frequency' in col_type or 'Frequencies' in col_type:
                    color_key = f'{base_type}-Freq' if base_type in ['Pseudorange', 'Phaserange', 'Doppler'] else 'Frequency'
                elif 'Ambiguity' in col_type or 'Ambiguities' in col_type:
                    color_key = f'{base_type}-Ambiguity' if base_type in ['Pseudorange', 'Phaserange', 'Doppler'] else 'Ambiguity'
                elif 'Landmark' in col_type:
                    color_key = 'Reproj-LM'
                else:
                    color_key = f'{base_type}-Sparse'
            
            if color_key not in colors and is_sparse:
                colors[color_key] = "#e0e0e0" # dynamically add it
                
            fc = colors.get(color_key, colors.get(base_type, colors.get('Other')))
            used_legend_keys.add(color_key)
            
            if is_sparse:
                sw = min(0.4, w_total * 0.15)
                sh = min(row_height * 0.2, 0.1)
                
                # 绘制稀疏点阵方块
                ax.add_patch(patches.Rectangle((x_start+w_total*0.1, current_y+row_height*0.1), sw, sh, lw=1.0, ec='gray', fc=fc))
                ax.add_patch(patches.Rectangle((x_start+w_total*0.5-sw/2, current_y+row_height*0.5-sh/2), sw, sh, lw=1.0, ec='gray', fc=fc))
                ax.add_patch(patches.Rectangle((x_start+w_total*0.9-sw, current_y+row_height*0.9-sh), sw, sh, lw=1.0, ec='gray', fc=fc))
                
                # 外围虚线框与省略号
                ax.add_patch(patches.Rectangle((x_start, current_y), w_total, row_height, lw=0.5, ec='#d0d0d0', fc='none', ls=':'))
                ax.text(x_start + w_total/3.5, current_y + row_height/2, '...', ha='center', va='center', fontsize=8, color='gray')
                ax.text(x_start + 2.5*w_total/3.5, current_y + row_height/2, '...', ha='center', va='center', fontsize=8, color='gray')
            else:
                ax.add_patch(patches.Rectangle((x_start, current_y), w_total, row_height, lw=0.5, ec='white', fc=fc, alpha=0.9))
                
        y_ticks.append(current_y + row_height/2)
        y_labels.append(str(dim))
        current_y += row_height
        
    ax.set_xlim(0, total_width)
    ax.set_ylim(current_y, 0)
    
    x_ticks = [(x_positions[i] + x_positions[i+1])/2 for i in range(num_cols)]
    x_labels = []
    for c_info in condensed_cols:
        lbl = c_info['name']
        x_labels.append(lbl)
        
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=16, rotation=45 if num_cols > 20 else 0, ha='center' if num_cols <= 20 else 'left')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('State Variables (Columns)', fontsize=20, labelpad=15)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.set_ylabel('Residual Dimensions (Rows)', fontsize=20)
    
    for x in x_positions: ax.axvline(x, color='#e0e0e0', linestyle='-', linewidth=0.5)
    for y in range(num_rows + 1): ax.axhline(y * row_height, color='#e0e0e0', linestyle='-', linewidth=0.5)
        
    legend_map = {
        'Marg': 'Marg (15x15)', 'IMU': 'IMU (15x15)', 
        'Reproj-Pose': 'Reproj-Pose (2nx6)', 'Reproj-LM': 'Reproj-LM (nx[2x3])',
        'NHC': 'NHC (2x15)', 'Pos': 'Pos (3x6)', 'HMC': 'HMC (1x15)', 'Vel': 'Vel (3x15)',
        'Pseudorange-Pose': 'Pseudorange-Pose (nx6)', 'Pseudorange-Freq': 'bug', 'Pseudorange-Ambiguity': 'bug',
        'Phaserange-Pose': 'Phaserange-Pose (nx6)', 'Phaserange-Freq': 'bug', 'Phaserange-Ambiguity': 'Phaserange-Ambiguity (nx[1x1])',
        'Doppler-Pose': 'Doppler-Pose (nx6)', 'Doppler-Freq': 'Doppler-Freq (nx[1x4])', 'Doppler-Ambiguity': 'bug',
        'Other': 'Other Base', 'Other-Sparse': 'Other Sparse',
        'Frequency': 'Frequency (nx[1x1])', 'Ambiguity': 'Ambiguity (nx[1x1])'
    }
    handles = []
    core_keys = ['IMU', 'HMC', 'NHC']
    active_keys = list(used_legend_keys) + core_keys
    
    # 按照字典顺序添加，把Frequency和Ambiguity推迟到最后添加
    delayed_keys = []
    
    for k, c in colors.items():
        if k in active_keys:
            if 'Frequency' in k or 'Ambiguity' in k or 'Freq' in k or 'Amb' in k:
                delayed_keys.append((k, c))
            else:
                handles.append(patches.Patch(color=c, label=legend_map.get(k, k)))
                
    # 强制将提取出的Freq和Amb模块追加在图例最后
    for k, c in delayed_keys:
        handles.append(patches.Patch(color=c, label=legend_map.get(k, k)))
            
    # 图例放入图内，并设置半透明背景防止遮挡数据
    ax.legend(handles=handles, loc='upper right', ncol=1, fontsize=12, facecolor='white', framealpha=0.9)
    
    # 在图片下方添加图表标题
    plt.title('Jacobian Matrix Sparsity Structure RRR', y=-0.1, fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    
    output_img = '/home/syl/GICI-IM/results/visualization/jacobian_structure_rrr.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Done! Saved structured jacobian to: file://{output_img}")

if __name__ == "__main__":
    draw_jacobian_structure()
