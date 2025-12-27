import glob
import os
import sys

def check_large_phi_errors(data_dir='/media/syl/longlong/GICI-Dataset', threshold_j=300):
    """检查pairwise_sd_errors.txt中是否存在j > threshold_j的配对"""
    
    # 查找所有pairwise_sd_errors.txt文件
    file_paths = glob.glob(f'{data_dir}/*/pairwise_sd_errors.txt')
    
    if not file_paths:
        print(f"在 {data_dir} 中未找到 pairwise_sd_errors.txt 文件")
        return
    
    print(f"找到 {len(file_paths)} 个文件需要检查")
    print("=" * 80)
    
    total_files = 0
    total_pairs = 0
    large_j_pairs = 0
    
    for file_path in file_paths:
        total_files += 1
        print(f"\n检查文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"  文件不存在: {file_path}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                header = f.readline()  # 跳过标题行
                file_pairs = 0
                file_large_pairs = 0
                
                for line_num, line in enumerate(f, 2):  # 从第2行开始计数（第1行是标题）
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    
                    pair_type = parts[0]
                    try:
                        error = float(parts[1])
                    except ValueError:
                        continue
                    
                    # 解析配对
                    pair_parts = pair_type.split('-')
                    if len(pair_parts) != 2:
                        continue
                    
                    try:
                        i = int(pair_parts[0])
                        j = int(pair_parts[1])
                        phi = j - i
                        
                        file_pairs += 1
                        
                        if j > threshold_j:
                            file_large_pairs += 1
                            large_j_pairs += 1
                            
                            # 打印前10个大j值的配对作为示例
                            if file_large_pairs <= 10:
                                print(f"  发现大j值配对: {pair_type} (i={i}, j={j}, phi={phi}), error={error:.6e}")
                    
                    except ValueError:
                        continue
                
                total_pairs += file_pairs
                
                if file_large_pairs > 0:
                    print(f"  本文件统计: 总配对数={file_pairs}, j>{threshold_j}的配对数={file_large_pairs}")
                else:
                    print(f"  本文件统计: 总配对数={file_pairs}, 未发现j>{threshold_j}的配对")
                    
        except Exception as e:
            print(f"  读取文件时出错: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("汇总统计:")
    print(f"  检查的文件总数: {total_files}")
    print(f"  所有文件的配对总数: {total_pairs}")
    print(f"  j > {threshold_j} 的配对总数: {large_j_pairs}")
    
    if large_j_pairs == 0:
        print(f"\n结论: 在所有文件中未发现 j > {threshold_j} 的配对")
    else:
        print(f"\n结论: 在所有文件中发现了 {large_j_pairs} 个 j > {threshold_j} 的配对")
    
    return large_j_pairs > 0


if __name__ == "__main__":
    # 可以通过命令行参数指定阈值
    threshold = 300
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
        except ValueError:
            print(f"无效的阈值参数: {sys.argv[1]}，使用默认值 {threshold}")
    
    print(f"开始检查 j > {threshold} 的配对...")
    has_large_j = check_large_phi_errors(threshold_j=threshold)
    
    if has_large_j:
        sys.exit(0)  # 发现大j值配对
    else:
        sys.exit(1)  # 未发现大j值配对