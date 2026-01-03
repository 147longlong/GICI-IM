import numpy as np
import re

# 读取和解析矩阵
with open('/home/dell/sunyulong/GICI-IM/results/debug/sig2_acc_debug.txt', 'r') as f:
    content = f.read()

lines = content.strip().split('\n')
matrix_data = []

for line in lines:
    if re.search(r'[-+]?\d+\.?\d*', line) and not line.startswith('Matrix'):
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        if numbers:
            row = [float(num) for num in numbers]
            matrix_data.append(row)

A = np.array(matrix_data)
# if A.shape != (89, 89):
#     A = A[:89*89].reshape(89, 89)

print(f"原始矩阵 A 的形状: {A.shape}")

def find_linear_dependent_rows_columns(A):
    """找出线性相关的行和列"""
    print("\n" + "="*60)
    print("线性相关分析")
    print("="*60)
    
    n = A.shape[0]
    results = {
        'dependent_rows': [],
        'dependent_cols': [],
        'row_rank': 0,
        'col_rank': 0,
        'row_dependencies': [],
        'col_dependencies': []
    }
    
    # 方法1: 使用QR分解找线性相关的行
    print("\n1. 使用QR分解分析行相关性:")
    Q, R = np.linalg.qr(A.T, mode='complete')  # 对A^T做QR分解，R的行对应A的列
    
    # R的对角线元素接近0表示对应的列（即A的行）线性相关
    diag_R = np.abs(np.diag(R))
    tolerance = max(A.shape) * np.finfo(float).eps * np.max(diag_R)
    
    print(f"QR分解阈值: {tolerance:.2e}")
    
    dependent_rows_qr = []
    for i in range(min(R.shape[0], R.shape[1])):
        if diag_R[i] < tolerance:
            dependent_rows_qr.append(i)
    
    print(f"QR检测到的线性相关行索引: {dependent_rows_qr}")
    results['dependent_rows'].extend(dependent_rows_qr)
    
    # 方法2: 使用SVD分析
    print("\n2. 使用SVD分析行和列空间:")
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.linalg.matrix_rank(A)
    results['row_rank'] = rank
    results['col_rank'] = rank
    
    print(f"矩阵秩: {rank}")
    print(f"零空间维数: {n - rank}")
    
    # 零空间向量对应线性相关的行/列组合
    if n > rank:
        # 零空间在V的列中
        null_space = Vt[rank:, :].T  # 零空间基向量
        print(f"零空间基向量形状: {null_space.shape}")
        
        # 对于每个零空间向量，找出非零分量对应的行
        for i in range(null_space.shape[1]):
            null_vec = null_space[:, i]
            # 找出显著的非零分量
            significant_indices = np.where(np.abs(null_vec) > 1e-10)[0]
            if len(significant_indices) > 0:
                print(f"零空间向量 {i+1}: 涉及行 {significant_indices.tolist()}")
                results['row_dependencies'].append({
                    'vector_idx': i,
                    'rows': significant_indices.tolist(),
                    'coefficients': null_vec[significant_indices].tolist()
                })
    
    # 方法3: 逐行检查相关性
    print("\n3. 逐行检查线性相关性:")
    independent_rows = []
    dependent_row_info = []
    
    for i in range(n):
        if i == 0:
            # 第一行总是独立的（除非全零）
            if not np.allclose(A[i, :], 0):
                independent_rows.append(i)
                print(f"行 {i}: 独立")
            else:
                print(f"行 {i}: 全零行")
                dependent_row_info.append((i, "全零行"))
        else:
            # 检查当前行是否可由已选独立行线性表示
            if len(independent_rows) > 0:
                # 构建当前子矩阵
                sub_A = A[independent_rows + [i], :]
                sub_rank = np.linalg.matrix_rank(sub_A)
                
                if sub_rank == len(independent_rows):
                    # 当前行是线性相关的
                    # 尝试找到线性组合系数
                    try:
                        # 解线性系统: A_indep * x = A[i, :]
                        A_indep = A[independent_rows, :]
                        coeff = np.linalg.lstsq(A_indep.T, A[i, :], rcond=None)[0]
                        
                        # 计算残差
                        residual = A[i, :] - A_indep.T @ coeff
                        residual_norm = np.linalg.norm(residual)
                        
                        if residual_norm < tolerance * np.linalg.norm(A[i, :]):
                            print(f"行 {i}: 相关于行 {independent_rows}, 系数: {coeff[:5]}..., 残差: {residual_norm:.2e}")
                            dependent_row_info.append((i, independent_rows.copy(), coeff))
                        else:
                            independent_rows.append(i)
                            print(f"行 {i}: 独立")
                    except:
                        independent_rows.append(i)
                        print(f"行 {i}: 独立")
                else:
                    independent_rows.append(i)
                    print(f"行 {i}: 独立")
            else:
                independent_rows.append(i)
                print(f"行 {i}: 独立")
    
    print(f"\n独立行数量: {len(independent_rows)}")
    print(f"独立行索引: {independent_rows}")
    
    # 方法4: 检查列相关性（类似方法）
    print("\n4. 检查列相关性:")
    independent_cols = []
    dependent_col_info = []
    
    for j in range(n):
        if j == 0:
            if not np.allclose(A[:, j], 0):
                independent_cols.append(j)
                print(f"列 {j}: 独立")
            else:
                print(f"列 {j}: 全零列")
                dependent_col_info.append((j, "全零列"))
        else:
            if len(independent_cols) > 0:
                sub_A = A[:, independent_cols + [j]]
                sub_rank = np.linalg.matrix_rank(sub_A)
                
                if sub_rank == len(independent_cols):
                    try:
                        A_indep = A[:, independent_cols]
                        coeff = np.linalg.lstsq(A_indep, A[:, j], rcond=None)[0]
                        residual = A[:, j] - A_indep @ coeff
                        residual_norm = np.linalg.norm(residual)
                        
                        if residual_norm < tolerance * np.linalg.norm(A[:, j]):
                            print(f"列 {j}: 相关于列 {independent_cols}, 系数: {coeff[:5]}..., 残差: {residual_norm:.2e}")
                            dependent_col_info.append((j, independent_cols.copy(), coeff))
                        else:
                            independent_cols.append(j)
                            print(f"列 {j}: 独立")
                    except:
                        independent_cols.append(j)
                        print(f"列 {j}: 独立")
                else:
                    independent_cols.append(j)
                    print(f"列 {j}: 独立")
            else:
                independent_cols.append(j)
                print(f"列 {j}: 独立")
    
    print(f"\n独立列数量: {len(independent_cols)}")
    print(f"独立列索引: {independent_cols}")
    
    # 方法5: 使用行列式检测
    print("\n5. 使用行列式检测:")
    print("检查每个可能的子矩阵...")
    
    # 检查是否有全零行或列
    zero_rows = []
    zero_cols = []
    
    for i in range(n):
        if np.allclose(A[i, :], 0):
            zero_rows.append(i)
        if np.allclose(A[:, i], 0):
            zero_cols.append(i)
    
    if zero_rows:
        print(f"全零行: {zero_rows}")
        results['dependent_rows'].extend(zero_rows)
    
    if zero_cols:
        print(f"全零列: {zero_cols}")
        results['dependent_cols'].extend(zero_cols)
    
    # 检查是否有重复行
    print("\n6. 检查重复行:")
    duplicate_rows = {}
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(A[i, :], A[j, :]):
                if i not in duplicate_rows:
                    duplicate_rows[i] = []
                duplicate_rows[i].append(j)
    
    if duplicate_rows:
        print("发现重复行:")
        for row1, duplicates in duplicate_rows.items():
            print(f"  行 {row1} 与行 {duplicates} 完全相同")
            results['dependent_rows'].extend([row1] + duplicates)
    
    # 检查是否有重复列
    print("\n7. 检查重复列:")
    duplicate_cols = {}
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(A[:, i], A[:, j]):
                if i not in duplicate_cols:
                    duplicate_cols[i] = []
                duplicate_cols[i].append(j)
    
    if duplicate_cols:
        print("发现重复列:")
        for col1, duplicates in duplicate_cols.items():
            print(f"  列 {col1} 与列 {duplicates} 完全相同")
            results['dependent_cols'].extend([col1] + duplicates)
    
    # 收集所有结果
    results['dependent_rows'] = list(set(results['dependent_rows']))
    results['dependent_cols'] = list(set(results['dependent_cols']))
    
    return results

def analyze_matrix_structure(A):
    """分析矩阵的结构特点"""
    print("\n" + "="*60)
    print("矩阵结构分析")
    print("="*60)
    
    n = A.shape[0]
    
    # 1. 检查矩阵的稀疏性
    zero_threshold = 1e-15
    nonzero_count = np.sum(np.abs(A) > zero_threshold)
    total_elements = n * n
    sparsity = 1.0 - nonzero_count / total_elements
    
    print(f"\n1. 稀疏性分析:")
    print(f"  总元素数: {total_elements}")
    print(f"  非零元素数: {nonzero_count}")
    print(f"  稀疏度: {sparsity:.2%}")
    
    # 2. 检查对角线元素
    print(f"\n2. 对角线分析:")
    diag_elements = np.diag(A)
    zero_diag = np.where(np.abs(diag_elements) < zero_threshold)[0]
    if len(zero_diag) > 0:
        print(f"  零对角线元素位置: {zero_diag.tolist()}")
    
    # 3. 检查行和列的范数
    print(f"\n3. 行/列范数分析:")
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)
    
    # 找出范数接近零的行和列
    zero_norm_threshold = 1e-10
    zero_norm_rows = np.where(row_norms < zero_norm_threshold)[0]
    zero_norm_cols = np.where(col_norms < zero_norm_threshold)[0]
    
    if len(zero_norm_rows) > 0:
        print(f"  接近零范数的行: {zero_norm_rows.tolist()}")
    if len(zero_norm_cols) > 0:
        print(f"  接近零范数的列: {zero_norm_cols.tolist()}")
    
    # 4. 检查矩阵块结构
    print(f"\n4. 矩阵块结构分析:")
    
    # 寻找可能的块对角结构
    block_size = 2  # 从2x2块开始检查
    while block_size <= n:
        if n % block_size == 0:
            num_blocks = n // block_size
            is_block_diagonal = True
            
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if i != j:
                        block = A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        if not np.allclose(block, 0, atol=1e-10):
                            is_block_diagonal = False
                            break
                if not is_block_diagonal:
                    break
            
            if is_block_diagonal:
                print(f"  发现{block_size}x{block_size}块对角结构，共{num_blocks}个块")
                break
        
        block_size += 1
    
    return {
        'sparsity': sparsity,
        'zero_diag_indices': zero_diag.tolist(),
        'zero_norm_rows': zero_norm_rows.tolist(),
        'zero_norm_cols': zero_norm_cols.tolist()
    }

def save_analysis_results(A, dep_results, struct_results):
    """保存分析结果"""
    print("\n" + "="*60)
    print("保存分析结果")
    print("="*60)
    
    with open('matrix_dependency_analysis.txt', 'w') as f:
        f.write("矩阵线性相关性分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"矩阵形状: {A.shape}\n")
        f.write(f"矩阵秩: {dep_results['row_rank']}\n")
        f.write(f"零空间维数: {A.shape[0] - dep_results['row_rank']}\n\n")
        
        f.write("1. 线性相关行:\n")
        if dep_results['dependent_rows']:
            for row_idx in sorted(dep_results['dependent_rows']):
                f.write(f"   行 {row_idx}: 可能线性相关\n")
        else:
            f.write("   未发现明显的线性相关行\n")
        
        f.write("\n2. 线性相关列:\n")
        if dep_results['dependent_cols']:
            for col_idx in sorted(dep_results['dependent_cols']):
                f.write(f"   列 {col_idx}: 可能线性相关\n")
        else:
            f.write("   未发现明显的线性相关列\n")
        
        f.write("\n3. 零空间向量分析:\n")
        if dep_results['row_dependencies']:
            for dep in dep_results['row_dependencies']:
                f.write(f"   零空间向量 {dep['vector_idx']+1}:\n")
                f.write(f"     涉及行: {dep['rows']}\n")
                f.write(f"     系数: {dep['coefficients']}\n")
        else:
            f.write("   未详细分析零空间\n")
        
        f.write("\n4. 矩阵结构分析:\n")
        f.write(f"   稀疏度: {struct_results['sparsity']:.2%}\n")
        if struct_results['zero_diag_indices']:
            f.write(f"   零对角线位置: {struct_results['zero_diag_indices']}\n")
        if struct_results['zero_norm_rows']:
            f.write(f"   零范数行: {struct_results['zero_norm_rows']}\n")
        if struct_results['zero_norm_cols']:
            f.write(f"   零范数列: {struct_results['zero_norm_cols']}\n")
    
    print("分析结果已保存到 matrix_dependency_analysis.txt")

# 执行分析
print("开始分析矩阵...")
print("="*60)

# 分析线性相关性
dependency_results = find_linear_dependent_rows_columns(A)

# 分析矩阵结构
structure_results = analyze_matrix_structure(A)

# 保存结果
save_analysis_results(A, dependency_results, structure_results)

print("\n分析完成！")
print("\n建议:")
print("1. 检查报告中标记的线性相关行/列")
print("2. 考虑移除或修改这些行/列以使矩阵满秩")
print("3. 如果这是物理问题的刚度矩阵，可能需要添加边界条件")