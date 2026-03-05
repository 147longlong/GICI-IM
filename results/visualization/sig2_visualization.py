#!/usr/bin/env python3
"""
Visualize Matrix matrix from visual integrity monitoring
Shows correlation structure for same landmark observations
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def parse_matrix_file(filename):
    """
    Parse the Matrix_output.txt file and extract matrices
    Returns a list of dictionaries with timestamp and matrix data
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by timestamp blocks
    blocks = content.split('----------------------------------------')
    
    data_list = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract timestamp
        # timestamp_match = re.search(r'Timestamp:\s*([\d\.]+)', block)
        # if not timestamp_match:
        #     continue
            
        # timestamp = float(timestamp_match.group(1))
        timestamp = 0.0 # Default timestamp
        
        # Extract shape
        shape_match = re.search(r'Matrix shape:\s*(\d+)\s*x\s*(\d+)', block)
        if not shape_match:
            continue
            
        rows = int(shape_match.group(1))
        cols = int(shape_match.group(2))
        
        # Extract matrix data
        matrix_start = block.find('Matrix:')
        if matrix_start == -1:
            continue
            
        matrix_text = block[matrix_start:].split('\n', 1)[1]
        
        # Parse matrix
        matrix_lines = matrix_text.strip().split('\n')
        matrix_data = []
        
        for line in matrix_lines:
            if not line.strip() or line.startswith('---'):
                break
            # Parse numbers from line
            numbers = [float(x) for x in line.split()]
            matrix_data.append(numbers)
        
        if len(matrix_data) == rows and all(len(row) == cols for row in matrix_data):
            matrix = np.array(matrix_data)
            data_list.append({
                'timestamp': timestamp,
                'matrix': matrix,
                'rows': rows,
                'cols': cols
            })
    
    return data_list

def visualize_matrix(matrix, timestamp, output_dir=None, pose_boundaries=None):
    """
    Visualize the Matrix matrix with different plots
    pose_boundaries: list of indices where pose boundaries occur (only for ReprojectionError rows)
    """
    # Create figure 1: Correlation structure (alone)
    fig1, ax1 = plt.subplots(figsize=(14, 12))
    
    # 1. Correlation structure for same landmarks (if applicable)
    # This will show blocks of correlations including 1.0 values
    vmax = np.max(matrix)
    vmin = np.min(matrix)
    
    # Create an academic-style colormap for non-negative variance values
    # White for 0, then gradient from light to dark colors
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import matplotlib.colors as mcolors
    
    # Define academic colormap: white -> light yellow -> orange -> dark red
    # This is similar to scientific visualization standards
    colors = plt.cm.YlOrRd(np.linspace(0.1, 1.0, 256))  # Yellow-Orange-Red
    colors[0] = [1, 1, 1, 1]  # Set first color (for 0 values) to white
    cmap_academic = ListedColormap(colors)
    
    # Use linear normalization from 0 to max value
    im1 = ax1.imshow(matrix, cmap=cmap_academic, aspect='auto', 
                    vmin=0, vmax=vmax)
    ax1.set_title('Covariance Matrix Structure', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Column Index (Parameters)', fontsize=16)
    ax1.set_ylabel('Row Index (Residuals)', fontsize=16)
    
    # Add colorbar with clear labels
    cbar = plt.colorbar(im1, ax=ax1, label='Variance Value')
    cbar.ax.tick_params(labelsize=14)
    
    # Add pose grouping boundaries if provided
    # Only draw boundaries for ReprojectionError rows
    if pose_boundaries and len(pose_boundaries) > 0:
        # Add vertical lines for column boundaries
        for boundary in pose_boundaries:
            if boundary > 0 and boundary < matrix.shape[1]:
                ax1.axvline(x=boundary - 0.5, color='gray', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label='ReprojectionError Boundary' if boundary == pose_boundaries[0] else "")
        
        # Add horizontal lines for row boundaries (same boundaries)
        # These are specifically for ReprojectionError rows
        for boundary in pose_boundaries:
            if boundary > 0 and boundary < matrix.shape[0]:
                ax1.axhline(y=boundary - 0.5, color='gray', linestyle='--', 
                           linewidth=1.5, alpha=0.7)
        
        # Add legend if boundaries exist
        if len(pose_boundaries) > 0:
            ax1.legend(loc='upper right', fontsize=10)
    
    # Add grid lines for better readability
    ax1.grid(True, alpha=0.2, color='gray', linestyle=':', linewidth=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'sig2_correlation.png')
        fig1.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved correlation structure to: {output_file}")
    
    plt.close(fig1)
    
    # Create figure 2: Diagonal elements and off-diagonal histogram together
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f'Matrix Analysis\n', fontsize=16)
    
    # 2. Diagonal elements
    diagonal = np.diag(matrix)
    axes[0].plot(diagonal, 'o-', linewidth=2, markersize=6, color='#1f77b4')
    axes[0].set_title('Diagonal Elements', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Index', fontsize=11)
    axes[0].set_ylabel('Variance', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 3. Off-diagonal elements (non-zero correlations) - as histogram
    off_diagonal = matrix - np.diag(diagonal)
    non_zero_mask = np.abs(off_diagonal) > 1e-10
    if np.any(non_zero_mask):
        off_diag_values = off_diagonal[non_zero_mask]
        axes[1].hist(off_diag_values, bins=20, alpha=0.7, edgecolor='black', color='#2ca02c')
        axes[1].set_title('Distribution of Off-Diagonal Elements', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Correlation Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No off-diagonal\ncorrelations', 
                       ha='center', va='center', transform=axes[1].transAxes, 
                       fontsize=14, color='gray')
        axes[1].set_title('Off-Diagonal Elements', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'sig2_diag_and_hist.png')
        fig2.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved diagonal and histogram to: {output_file}")
    
    plt.close(fig2)

def analyze_correlation_structure(matrix, timestamp):
    """
    Analyze the correlation structure to identify blocks corresponding to same landmarks
    """
    print(f"\n=== Analysis for Timestamp: {timestamp} ===")
    print(f"Matrix shape: {matrix.shape}")
    
    # Check if matrix is diagonal
    off_diag = matrix - np.diag(np.diag(matrix))
    max_off_diag = np.max(np.abs(off_diag))
    print(f"Maximum off-diagonal value: {max_off_diag:.6e}")
    
    if max_off_diag < 1e-10:
        print("Matrix is diagonal (no correlations between observations)")
        return
    
    # Find non-zero off-diagonal blocks
    # Group rows/columns that have correlations
    corr_matrix = np.abs(off_diag) > 1e-10
    
    # Find connected components (blocks of correlated observations)
    visited = set()
    blocks = []
    
    for i in range(matrix.shape[0]):
        if i in visited:
            continue
        
        # Find all rows/columns correlated with i
        block = set([i])
        to_check = [i]
        
        while to_check:
            current = to_check.pop(0)
            # Find rows correlated with current
            for j in range(matrix.shape[0]):
                if j not in block and (corr_matrix[current, j] or corr_matrix[j, current]):
                    block.add(j)
                    to_check.append(j)
        
        if len(block) > 1:
            blocks.append(sorted(list(block)))
            visited.update(block)
    
    print(f"Found {len(blocks)} correlated blocks:")
    for idx, block in enumerate(blocks):
        print(f"  Block {idx+1}: rows {block}")
        # Extract submatrix for this block
        submatrix = matrix[np.ix_(block, block)]
        print(f"    Submatrix shape: {submatrix.shape}")
        print(f"    Diagonal values: {np.diag(submatrix)}")
        print(f"    Off-diagonal values: {submatrix - np.diag(np.diag(submatrix))}")
    
    return blocks

def extract_pose_boundaries_from_jacobian(jacobian_file):
    """
    Extract pose boundaries from Jacobian analysis file
    Returns list of boundary indices where pose groups change
    Only includes boundaries for ReprojectionError rows
    """
    if not os.path.exists(jacobian_file):
        print(f"Warning: Jacobian file {jacobian_file} not found!")
        return None
    
    try:
        with open(jacobian_file, 'r') as f:
            content = f.read()
        
        # Find the Jacobian Analysis section
        analysis_start = content.find("Jacobian Analysis:")
        if analysis_start == -1:
            print("Warning: Jacobian Analysis section not found!")
            return None
        
        analysis_section = content[analysis_start:]
        
        # Parse the table to find pose boundaries for ReprojectionError only
        lines = analysis_section.split('\n')
        
        # Store row ranges for ReprojectionError only
        reprojection_rows = set()  # All rows that belong to ReprojectionError
        
        for line in lines:
            if not line.strip() or line.startswith('===') or line.startswith('---'):
                continue
            
            # Look for lines with ReprojectionError
            if '(ReprojectionError)' in line:
                # Extract row range
                row_match = re.search(r'\[(\d+),(\d+)\]', line)
                if row_match:
                    row_start = int(row_match.group(1))
                    row_end = int(row_match.group(2))
                    # Add all rows in this range to the set
                    reprojection_rows.update(range(row_start, row_end + 1))
        
        if not reprojection_rows:
            print("Warning: No ReprojectionError rows found in Jacobian analysis!")
            return None
        
        # Sort the rows to find boundaries
        sorted_rows = sorted(list(reprojection_rows))
        
        # Find boundaries where there's a gap in row numbers
        # This indicates a new pose group
        pose_boundaries = []
        for i in range(1, len(sorted_rows)):
            if sorted_rows[i] != sorted_rows[i-1] + 1:
                # There's a gap, this is a boundary
                pose_boundaries.append(sorted_rows[i])
        
        # Add the end boundary after the last ReprojectionError row
        if sorted_rows:
            last_row = sorted_rows[-1]
            pose_boundaries.append(last_row + 1)
        
        print(f"Found {len(reprojection_rows)} ReprojectionError rows")
        print(f"Extracted {len(pose_boundaries)} pose boundaries for ReprojectionError: {pose_boundaries}")
        return pose_boundaries
        
    except Exception as e:
        print(f"Error parsing Jacobian file: {e}")
        return None

def main():
    # File paths
    matrix_file = "/home/syl/GICI-IM/results/sig2_all_output.txt"
    jacobian_file = "/home/syl/GICI-IM/results/jacobian_visualization.txt"
    output_dir = "/home/syl/GICI-IM/results/"
    
    # Parse data
    print("Parsing Matrix output file...")
    data_list = parse_matrix_file(matrix_file)
    
    if not data_list:
        print("No data found in file!")
        return
    
    print(f"Found {len(data_list)} matrices to visualize")
    
    # Extract pose boundaries from Jacobian analysis
    print("\nExtracting pose boundaries from Jacobian analysis...")
    pose_boundaries = extract_pose_boundaries_from_jacobian(jacobian_file)
    
    if pose_boundaries is None:
        print("Warning: Could not extract pose boundaries. Proceeding without grouping lines.")
    
    # Process each matrix
    for i, data in enumerate(data_list):
        print(f"\nProcessing matrix {i+1}/{len(data_list)}")
        
        # Analyze structure
        blocks = analyze_correlation_structure(data['matrix'], data['timestamp'])
        
        # Visualize with pose boundaries
        visualize_matrix(data['matrix'], data['timestamp'], output_dir, pose_boundaries)
        
        # Ask if user wants to continue
        if i < len(data_list) - 1:
            response = input(f"\nPress Enter to continue to next matrix, or 'q' to quit: ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    main()