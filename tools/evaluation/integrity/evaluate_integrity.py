import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Increase font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def read_tum(file_path, with_pl=False):
    """
    Reads a TUM file.
    If with_pl is True, expects columns 8, 9, 10 to be xpl, ypl, vpl.
    Returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            
            # Basic TUM: timestamp tx ty tz qx qy qz qw
            timestamp = float(parts[0])
            tx = float(parts[1])
            ty = float(parts[2])
            tz = float(parts[3])
            qx = float(parts[4])
            qy = float(parts[5])
            qz = float(parts[6])
            qw = float(parts[7])
            
            entry = {
                'timestamp': timestamp,
                'pos': np.array([tx, ty, tz]),
                'quat': np.array([qx, qy, qz, qw])
            }
            
            if with_pl:
                # Expecting xpl, ypl, vpl after qw
                # indices: 0:ts, 1-3:pos, 4-7:quat, 8:xpl, 9:ypl, 10:vpl
                if len(parts) >= 11:
                    xpl = float(parts[9]) if parts[9].lower() != 'nan' else np.nan
                    ypl = float(parts[8]) if parts[8].lower() != 'nan' else np.nan
                    vpl = float(parts[10]) if parts[10].lower() != 'nan' else np.nan
                    entry['pl'] = np.array([xpl, ypl, vpl])
                else:
                    entry['pl'] = np.array([np.nan, np.nan, np.nan])
            
            data.append(entry)
    return data

def associate_data(gt_data, est_data, max_diff=0.05):
    """
    Associates estimated data with ground truth data based on timestamp.
    Simple nearest neighbor association.
    """
    gt_timestamps = np.array([d['timestamp'] for d in gt_data])
    matches = []
    
    for est in est_data:
        t = est['timestamp']
        # Find closest timestamp in GT
        idx = (np.abs(gt_timestamps - t)).argmin()
        diff = np.abs(gt_timestamps[idx] - t)
        
        if diff < max_diff:
            matches.append((gt_data[idx], est))
            
    return matches

def get_yaw_from_quat(q):
    """
    Calculates yaw (rotation around Z-axis) from quaternion [x, y, z, w].
    """
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def main():
    gt_file = 'Ground-Truth.tum'
    est_file = 'srr_solution.txt.tum'
    
    if not os.path.exists(gt_file) or not os.path.exists(est_file):
        print(f"Error: Files {gt_file} or {est_file} not found.")
        return

    print(f"Reading {gt_file}...")
    gt_data = read_tum(gt_file, with_pl=False)
    print(f"Reading {est_file}...")
    est_data = read_tum(est_file, with_pl=True)
    
    print(f"Associating data (max time diff 0.05s)...")
    matches = associate_data(gt_data, est_data)
    print(f"Found {len(matches)} matches.")
    
    if not matches:
        print("No matches found. Check timestamps.")
        return

    # Extract errors and PLs
    timestamps = []
    errors_x = []
    errors_y = []
    errors_z = []
    
    pl_x = []
    pl_y = []
    pl_v = []
    
    for gt, est in matches:
        timestamps.append(est['timestamp'])
        
        # Position error
        err = np.abs(est['pos'] - gt['pos'])
        errors_x.append(err[0])
        errors_y.append(err[1])
        errors_z.append(err[2])
        
        # Protection levels
        pl = est['pl']
        pl_x.append(pl[0])
        pl_y.append(pl[1])
        pl_v.append(pl[2])

    timestamps = np.array(timestamps)
    # Normalize time to start from 0
    timestamps = timestamps - timestamps[0]
    
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    errors_z = np.array(errors_z)
    
    pl_x = np.array(pl_x)
    pl_y = np.array(pl_y)
    pl_v = np.array(pl_v)
    
    # Calculate Horizontal and Vertical Errors
    horiz_error = np.sqrt(errors_x**2 + errors_y**2)
    vert_error = errors_z
    
    # Plot 1: Components
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # X Error and XPL
    axs[0].plot(timestamps, errors_x, label='Error X', color='blue', linewidth=2)
    axs[0].plot(timestamps, pl_x, label='XPL', color='red', linestyle='-', linewidth=3)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Error / PL (m)')
    axs[0].set_title('X Position Error and Protection Level')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, which="both", ls="-")
    
    # Y Error and YPL
    axs[1].plot(timestamps, errors_y, label='Error Y', color='green', linewidth=2)
    axs[1].plot(timestamps, pl_y, label='YPL', color='orange', linestyle='-', linewidth=3)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Error / PL (m)')
    axs[1].set_title('Y Position Error and Protection Level')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, which="both", ls="-")
    
    # Vertical Error and VPL
    axs[2].plot(timestamps, vert_error, label='Vertical Error', color='purple', linewidth=2)
    axs[2].plot(timestamps, pl_v, label='VPL', color='magenta', linestyle='-', linewidth=3)
    axs[2].set_yscale('log')
    axs[2].set_ylabel('Error / PL (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('Vertical Position Error and Protection Level')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.savefig('integrity_evaluation_components.png')
    print("Saved plot to integrity_evaluation_components.png")
    
    # Plot 2: Horizontal Error vs PLs
    plt.figure(figsize=(12, 8))
    plt.plot(timestamps, horiz_error, label='Horizontal Error', color='blue', linewidth=2)
    plt.plot(timestamps, pl_x, label='XPL', color='red', linestyle='-', alpha=0.5, linewidth=3)
    plt.plot(timestamps, pl_y, label='YPL', color='orange', linestyle='-', alpha=0.5, linewidth=3)
    
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Error / PL (m)')
    plt.title('Horizontal Position Error vs Protection Levels')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-")
    plt.savefig('integrity_evaluation_horizontal.png')
    print("Saved plot to integrity_evaluation_horizontal.png")

    # Plot 3: Trajectory with PL Boxes (The "Cube" plot)
    print("Generating Trajectory with PL boxes plot...")
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Plot GT
    gt_x = [d['pos'][0] for d in gt_data]
    gt_y = [d['pos'][1] for d in gt_data]
    plt.plot(gt_x, gt_y, 'k-', label='Ground Truth', linewidth=1, alpha=0.6)
    
    # Plot Est
    est_x = [est['pos'][0] for _, est in matches]
    est_y = [est['pos'][1] for _, est in matches]
    plt.plot(est_x, est_y, 'b-', label='Estimated', linewidth=1)
    
    # Draw PL Rectangles
    # Downsample for visibility - adjust step based on data density
    step = 10 
    
    # Define ranges, colors, and scaling
    # Format: (min_hpl, max_hpl, color, label)
    # Colors: Green -> Yellow -> Red -> Dark Red (Academic style)
    ranges = [
        (0.0, 0.05, '#006400', 'HPL < 0.05m'),          # Dark Green
        (0.05, 0.1, '#32CD32', '0.05 <= HPL < 0.1m'),   # Lime Green
        (0.1, 0.5, '#CCCC00', '0.1 <= HPL < 0.5m'),     # Dark Yellow
        (0.5, 1.0, '#FFA500', '0.5 <= HPL < 1.0m'),     # Orange
        (1.0, 5.0, '#FF4500', '1.0 <= HPL < 5.0m'),     # Orange Red
        (5.0, 10.0, '#FF0000', '5.0 <= HPL < 10.0m'),   # Red
        (10.0, float('inf'), '#800000', 'HPL >= 10.0m') # Maroon
    ]
    
    scale_magnified = 80.0
    scale_normal = 1.0
    
    # Create dummy handles for the legend
    for start, end, color, label in ranges:
        scale = scale_normal if start >= 10.0 else scale_magnified
        scale_str = f" (x{scale})"
        plt.plot([], [], color=color, linestyle='-', linewidth=1.5, label=label + scale_str)
    
    for i in range(0, len(matches), step):
        gt, est = matches[i]
        x, y, z = gt['pos']
        xpl, ypl, vpl = est['pl']
        q = gt['quat']
        
        if np.isnan(xpl) or np.isnan(ypl):
            continue

        # Calculate HPL for color classification (before scaling)
        hpl = np.sqrt(xpl**2 + ypl**2)
        
        box_color = 'black'
        current_scale = scale_magnified
        
        # Determine color and scale
        for start, end, color, label in ranges:
            if start <= hpl < end:
                box_color = color
                if start >= 1.0:
                    current_scale = scale_normal
                break

        # Apply scaling for visualization
        xpl *= current_scale
        ypl *= current_scale
            
        # Yaw
        yaw = get_yaw_from_quat(q)
        
        # Define rectangle corners in local frame (centered at 0)
        # Assuming XPL is along local X, YPL along local Y
        # Corners: (xpl, ypl), (-xpl, ypl), (-xpl, -ypl), (xpl, -ypl)
        
        corners_local = np.array([
            [xpl, ypl],
            [-xpl, ypl],
            [-xpl, -ypl],
            [xpl, -ypl],
            [xpl, ypl] # Close loop
        ])
        
        # Rotation matrix
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate and translate
        corners_global = corners_local @ R.T
        corners_global[:, 0] += x
        corners_global[:, 1] += y
        
        plt.plot(corners_global[:, 0], corners_global[:, 1], color=box_color, linewidth=1.0, alpha=0.8)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory and Protection Levels (XPL/YPL Boxes)')
    plt.axis('equal')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.savefig('integrity_evaluation_trajectory.png')
    print("Saved plot to integrity_evaluation_trajectory.png")
    
    # plt.show()

if __name__ == "__main__":
    main()
