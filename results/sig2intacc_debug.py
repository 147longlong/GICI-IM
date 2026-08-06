import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import sparse

# 读取和解析矩阵
with open('/home/syl/GICI-IM/results/jacobian/sig2_int_output1679304413.400000.txt', 'r') as f:
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
print(f"原始矩阵 A 的形状: {A.shape}")

def deep_analyze_matrix(A, matrix_name="矩阵A"):
    """
    深度分析矩阵为什么不对称且不正定
    """
    print("\n" + "="*80)
    print(f"深度分析: {matrix_name}")
    print("="*80)
    
    n = A.shape[0]
    results = {}
    
    # 1. 详细不对称性分析
    print("\n1. 不对称性详细分析:")
    diff = A - A.T
    abs_diff = np.abs(diff)
    
    # 找到不对称最严重的位置
    max_diff_idx = np.unravel_index(np.argmax(abs_diff), A.shape)
    max_diff_value = diff[max_diff_idx]
    
    print(f"   最大不对称位置: ({max_diff_idx[0]}, {max_diff_idx[1]})")
    print(f"   不对称值: {max_diff_value:.6e}")
    print(f"   该位置原始值: A[{max_diff_idx[0]}, {max_diff_idx[1]}] = {A[max_diff_idx[0], max_diff_idx[1]]:.6e}")
    print(f"   对应转置值: A[{max_diff_idx[1]}, {max_diff_idx[0]}] = {A[max_diff_idx[1], max_diff_idx[0]]:.6e}")
    
    # 统计不对称程度
    threshold = 1e-10
    significant_asym = np.sum(abs_diff > threshold)
    print(f"   显著不对称元素数量 (> {threshold}): {significant_asym} / {n*n} ({significant_asym/(n*n):.2%})")
    
    # 不对称元素分布
    print(f"   不对称值范围: [{np.min(diff):.6e}, {np.max(diff):.6e}]")
    print(f"   不对称绝对值均值: {np.mean(abs_diff):.6e}")
    
    # 2. 对称化后的矩阵分析
    print("\n2. 对称化处理:")
    A_sym = 0.5 * (A + A.T)
    
    # 验证对称化效果
    sym_error = np.max(np.abs(A_sym - A_sym.T))
    print(f"   对称化后最大不对称误差: {sym_error:.6e}")
    
    # 3. 负特征值根源分析
    print("\n3. 负特征值根源分析:")
    
    # 计算特征值
    try:
        eigvals = np.linalg.eigvalsh(A_sym)
        sorted_eigvals = np.sort(eigvals)
        
        # 负特征值统计
        neg_threshold = -1e-10
        negative_eigvals = eigvals[eigvals < neg_threshold]
        positive_eigvals = eigvals[eigvals > 0]
        zero_eigvals = eigvals[np.abs(eigvals) <= 1e-10]
        
        print(f"   总特征值数量: {len(eigvals)}")
        print(f"   负特征值数量 (< {neg_threshold:.0e}): {len(negative_eigvals)}")
        print(f"   正特征值数量: {len(positive_eigvals)}")
        print(f"   零特征值数量: {len(zero_eigvals)}")
        
        if len(negative_eigvals) > 0:
            print(f"\n   负特征值详细信息:")
            print(f"   最小特征值: {np.min(eigvals):.6e}")
            print(f"   最大负特征值: {np.max(negative_eigvals):.6e}")
            print(f"   负特征值平均值: {np.mean(negative_eigvals):.6e}")
            print(f"   负特征值绝对值总和: {np.sum(np.abs(negative_eigvals)):.6e}")
            
            # 负特征值对应的特征向量分析
            print(f"\n   负特征值对应的特征向量分析:")
            eigvals_full, eigvecs = np.linalg.eigh(A_sym)
            
            # 找到最小特征值对应的特征向量
            min_idx = np.argmin(eigvals_full)
            min_eigvec = eigvecs[:, min_idx]
            
            # 特征向量中绝对值最大的分量
            max_component_idx = np.argmax(np.abs(min_eigvec))
            print(f"   最小特征值 λ_min = {eigvals_full[min_idx]:.6e}")
            print(f"   对应特征向量中最大分量: 位置 {max_component_idx}, 值 {min_eigvec[max_component_idx]:.6f}")
            
            # 检查这些位置在原始矩阵中
            print(f"   检查矩阵中相关位置的值:")
            print(f"     A[{max_component_idx}, {max_component_idx}] = {A_sym[max_component_idx, max_component_idx]:.6e}")
            
            # 检查该行的其他值
            row = A_sym[max_component_idx, :]
            significant_cols = np.where(np.abs(row) > 0.1)[0]  # 找到大于0.1的元素
            for col in significant_cols[:5]:  # 显示前5个
                print(f"     A[{max_component_idx}, {col}] = {row[col]:.6e}")
    
    except Exception as e:
        print(f"   特征值计算失败: {e}")
    
    # 4. 对角线元素分析
    print("\n4. 对角线元素详细分析:")
    diag = np.diag(A_sym)
    
    print(f"   对角线元素范围: [{np.min(diag):.6e}, {np.max(diag):.6e}]")
    print(f"   对角线平均值: {np.mean(diag):.6e}")
    print(f"   对角线标准差: {np.std(diag):.6e}")
    
    # 检查负对角线元素
    neg_diag = diag[diag < 0]
    if len(neg_diag) > 0:
        print(f"\n   发现负对角线元素!")
        print(f"   负对角线数量: {len(neg_diag)}")
        print(f"   负对角线最小值: {np.min(neg_diag):.6e}")
        print(f"   负对角线位置: {np.where(diag < 0)[0].tolist()}")
        
        # 详细检查第一个负对角线位置
        first_neg_idx = np.where(diag < 0)[0][0]
        print(f"\n   第一个负对角线元素分析 (位置 {first_neg_idx}):")
        print(f"     A[{first_neg_idx}, {first_neg_idx}] = {diag[first_neg_idx]:.6e}")
        
        # 检查该行的其他元素
        row_vals = A_sym[first_neg_idx, :]
        col_vals = A_sym[:, first_neg_idx]
        
        # 找出该行中绝对值较大的元素
        large_vals = np.where(np.abs(row_vals) > 1.0)[0]
        for col in large_vals[:5]:
            if col != first_neg_idx:
                print(f"     A[{first_neg_idx}, {col}] = {row_vals[col]:.6e}")
    
    # 5. 矩阵结构可视化（前100x100部分）
    print("\n5. 矩阵结构检查:")
    
    # 检查矩阵是否稀疏
    zero_threshold = 1e-10
    nnz = np.sum(np.abs(A_sym) > zero_threshold)
    sparsity = 1.0 - nnz / (n * n)
    
    print(f"   非零元素数量: {nnz} / {n*n} ({nnz/(n*n):.2%})")
    print(f"   稀疏度: {sparsity:.2%}")
    
    # 检查块对角结构
    print(f"\n   检查块对角结构:")
    block_size = 10
    num_blocks = n // block_size
    
    for i in range(min(3, num_blocks)):  # 检查前3个块
        block = A_sym[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size]
        block_energy = np.sum(np.abs(block))
        print(f"   块 [{i*block_size}:{(i+1)*block_size}, {i*block_size}:{(i+1)*block_size}] 能量: {block_energy:.6e}")
    
    # 6. 数值稳定性分析
    print("\n6. 数值稳定性分析:")
    
    # 条件数
    if 'eigvals' in locals():
        cond_num = np.abs(np.max(eigvals) / np.min(eigvals))
        print(f"   条件数 (κ): {cond_num:.2e}")
        
        if cond_num > 1e10:
            print(f"   ⚠ 矩阵病态！条件数过大")
    
    # 行列式（仅用于小矩阵）
    if n <= 200:
        try:
            det = np.linalg.det(A_sym)
            print(f"   行列式: {det:.6e}")
            
            if np.abs(det) < 1e-50:
                print(f"   ⚠ 行列式非常接近零，矩阵可能奇异")
        except:
            print(f"   行列式计算失败")
    
    # 7. 修复建议
    print("\n7. 问题诊断与修复建议:")
    
    issues = []
    
    if significant_asym > 0.01 * n * n:  # 超过1%的元素不对称
        issues.append("矩阵严重不对称")
        print(f"   ❌ 问题1: 矩阵严重不对称")
        print(f"      建议: 检查矩阵构建代码，确保对称操作正确")
    
    if len(negative_eigvals) > 0:
        issues.append(f"有{len(negative_eigvals)}个负特征值")
        print(f"   ❌ 问题2: 矩阵有负特征值")
        print(f"      最小特征值: {np.min(eigvals):.6e}")
        
        if np.min(eigvals) < -1.0:
            print(f"      严重: 负特征值绝对值较大，不是数值误差")
            print(f"      可能原因: 矩阵构建逻辑错误或数据错误")
        else:
            print(f"      可能是数值误差，建议正则化")
    
    if len(neg_diag) > 0:
        issues.append(f"有{len(neg_diag)}个负对角线元素")
        print(f"   ❌ 问题3: 矩阵有负对角线元素")
        print(f"      协方差/精度矩阵的对角线必须是正数!")
    
    # 修复策略
    print(f"\n   修复策略:")
    if "矩阵严重不对称" in issues:
        print(f"   1. 必须修复不对称问题:")
        print(f"      - 检查矩阵加法/乘法操作是否破坏对称性")
        print(f"      - 检查是否有非对称的更新操作")
        print(f"      - 确保使用 A = 0.5*(A + A.T) 保持对称")
    
    if "有负特征值" in issues:
        print(f"\n   2. 处理负特征值:")
        if np.min(eigvals) < -0.1:  # 明显的负特征值
            print(f"      - 检查矩阵构建的数学公式")
            print(f"      - 验证输入数据的正确性")
            print(f"      - 可能需要重新设计算法")
        else:
            print(f"      - 添加正则化: A_reg = A + εI")
            print(f"      - 使用 Higham 方法投影到半正定锥")
    
    if "有负对角线元素" in issues:
        print(f"\n   3. 修复负对角线:")
        print(f"      - 对角线元素必须非负")
        print(f"      - 检查对角线计算代码")
        print(f"      - 可以用绝对值或小正数替代: diag = np.maximum(diag, 1e-8)")
    
    results.update({
        'A_sym': A_sym,
        'eigenvalues': eigvals if 'eigvals' in locals() else None,
        'diagonal': diag,
        'asymmetry_stats': {
            'max_diff': max_diff_value,
            'significant_asym': significant_asym,
            'asym_percentage': significant_asym/(n*n)
        },
        'issues': issues,
        'negative_eigvals': negative_eigvals if 'negative_eigvals' in locals() else [],
        'negative_diag': neg_diag if 'neg_diag' in locals() else []
    })
    
    return results

def visualize_matrix_issues(A, A_sym, results):
    """可视化矩阵问题"""
    print("\n" + "="*80)
    print("矩阵可视化分析")
    print("="*80)
    
    n = A.shape[0]
    
    # 只显示前100x100（如果矩阵太大）
    display_size = min(100, n)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('矩阵问题可视化分析', fontsize=16)
    
    # 1. 原始矩阵的热图（前100x100）
    im1 = axes[0, 0].imshow(np.abs(A[:display_size, :display_size]), 
                          cmap='hot', aspect='auto')
    axes[0, 0].set_title('原始矩阵绝对值热图')
    axes[0, 0].set_xlabel('列索引')
    axes[0, 0].set_ylabel('行索引')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 不对称性热图
    diff = A - A.T
    im2 = axes[0, 1].imshow(np.abs(diff[:display_size, :display_size]), 
                          cmap='Reds', aspect='auto')
    axes[0, 1].set_title('不对称性热图')
    axes[0, 1].set_xlabel('列索引')
    axes[0, 1].set_ylabel('行索引')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 对称化后矩阵热图
    im3 = axes[0, 2].imshow(np.abs(A_sym[:display_size, :display_size]), 
                          cmap='viridis', aspect='auto')
    axes[0, 2].set_title('对称化后矩阵热图')
    axes[0, 2].set_xlabel('列索引')
    axes[0, 2].set_ylabel('行索引')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 特征值分布
    if results['eigenvalues'] is not None:
        eigvals = results['eigenvalues']
        axes[1, 0].plot(np.sort(eigvals), 'b-', linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('特征值排序分布')
        axes[1, 0].set_xlabel('排序索引')
        axes[1, 0].set_ylabel('特征值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 标记负特征值
        neg_mask = eigvals < 0
        if np.any(neg_mask):
            neg_indices = np.where(neg_mask)[0]
            axes[1, 0].scatter(neg_indices, eigvals[neg_mask], 
                             color='red', s=50, zorder=5, label=f'负特征值 ({len(neg_indices)}个)')
            axes[1, 0].legend()
    
    # 5. 对角线元素分布
    diag = np.diag(A_sym)
    axes[1, 1].plot(range(len(diag)), diag, 'g-', linewidth=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('对角线元素分布')
    axes[1, 1].set_xlabel('对角线索引')
    axes[1, 1].set_ylabel('对角线值')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 标记负对角线
    neg_diag_mask = diag < 0
    if np.any(neg_diag_mask):
        neg_diag_indices = np.where(neg_diag_mask)[0]
        axes[1, 1].scatter(neg_diag_indices, diag[neg_diag_mask], 
                         color='red', s=30, zorder=5, label=f'负对角线 ({len(neg_diag_indices)}个)')
        axes[1, 1].legend()
    
    # 6. 不对称值分布直方图
    if n <= 1000:  # 只对小矩阵做完整直方图
        diff_flat = diff.flatten()
        axes[1, 2].hist(diff_flat[np.abs(diff_flat) > 1e-10], bins=50, 
                       color='orange', edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('不对称值分布直方图')
        axes[1, 2].set_xlabel('不对称值')
        axes[1, 2].set_ylabel('频数')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # 对大矩阵，只显示统计信息
        axes[1, 2].text(0.5, 0.5, f'矩阵太大 ({n}x{n})\n无法显示完整直方图\n最大不对称: {np.max(np.abs(diff)):.2e}',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('不对称统计信息')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('matrix_analysis_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存到 matrix_analysis_visualization.png")
    plt.show()

def apply_corrections(A_sym, results):
    """应用修正措施"""
    print("\n" + "="*80)
    print("应用修正措施")
    print("="*80)
    
    n = A_sym.shape[0]
    A_corrected = A_sym.copy()
    
    corrections_applied = []
    
    # 1. 修复负对角线（如果存在）
    diag = np.diag(A_corrected)
    if np.any(diag < 0):
        print("1. 修复负对角线元素:")
        neg_diag_indices = np.where(diag < 0)[0]
        print(f"   发现 {len(neg_diag_indices)} 个负对角线元素")
        
        # 策略：用小的正数替换负值
        min_positive = np.min(diag[diag > 0]) if np.any(diag > 0) else 1e-8
        replacement = max(1e-8, min_positive * 0.01)
        
        for idx in neg_diag_indices:
            old_value = A_corrected[idx, idx]
            A_corrected[idx, idx] = replacement
            print(f"   位置 [{idx}, {idx}]: {old_value:.6e} → {replacement:.6e}")
        
        corrections_applied.append(f"修复{len(neg_diag_indices)}个负对角线")
    
    # 2. 检查并应用正则化（如果特征值太负）
    if results['eigenvalues'] is not None:
        min_eig = np.min(results['eigenvalues'])
        
        if min_eig < -1e-6:  # 明显的负特征值
            print(f"\n2. 处理明显负特征值:")
            print(f"   最小特征值: {min_eig:.6e}")
            
            # 计算需要的正则化量
            epsilon = abs(min_eig) + 1e-8
            print(f"   需要添加正则化量: {epsilon:.6e}")
            
            # 应用正则化
            A_corrected += epsilon * np.eye(n)
            corrections_applied.append(f"添加正则化 ε={epsilon:.2e}")
            
            print(f"   已应用: A_corrected = A + {epsilon:.2e} * I")
    
    # 3. 验证修正后的矩阵
    if corrections_applied:
        print(f"\n3. 验证修正结果:")
        
        # 检查对角线
        new_diag = np.diag(A_corrected)
        new_neg_diag = np.sum(new_diag < 0)
        print(f"   负对角线数量: {new_neg_diag} (修正前: {len(results['negative_diag'])})")
        
        # 检查特征值
        try:
            new_eigvals = np.linalg.eigvalsh(A_corrected)
            new_min_eig = np.min(new_eigvals)
            new_neg_eig = np.sum(new_eigvals < -1e-10)
            
            print(f"   最小特征值: {new_min_eig:.6e} (修正前: {min_eig:.6e})")
            print(f"   负特征值数量: {new_neg_eig} (修正前: {len(results['negative_eigvals'])})")
            
            # 尝试Cholesky分解
            try:
                L = np.linalg.cholesky(A_corrected)
                print(f"   ✓ Cholesky分解成功！")
                corrections_applied.append("Cholesky分解成功")
            except np.linalg.LinAlgError as e:
                print(f"   ✗ Cholesky分解仍然失败: {e}")
                
                # 如果仍然失败，可能需要更强的正则化
                if new_min_eig < 1e-8:
                    additional_epsilon = 1e-8 - new_min_eig
                    print(f"   需要额外正则化: {additional_epsilon:.6e}")
                    A_corrected += additional_epsilon * np.eye(n)
                    corrections_applied.append(f"额外正则化 {additional_epsilon:.2e}")
                    
                    # 再次验证
                    try:
                        L = np.linalg.cholesky(A_corrected)
                        print(f"   ✓ 额外正则化后Cholesky分解成功！")
                    except:
                        print(f"   ✗ 仍然失败，建议检查矩阵根本问题")
        
        except Exception as e:
            print(f"   特征值计算失败: {e}")
    else:
        print("无需修正，矩阵已正常")
    
    return A_corrected, corrections_applied

# 主执行流程
print("开始深度分析矩阵问题...")
print("="*80)

# 1. 深度分析
results = deep_analyze_matrix(A, "sig2_acc 矩阵")

# 2. 可视化分析（如果矩阵不是太大）
if A.shape[0] <= 200:
    visualize_matrix_issues(A, results['A_sym'], results)
else:
    print(f"\n矩阵太大 ({A.shape[0]}x{A.shape[0]})，跳过可视化")

# 3. 应用修正
A_corrected, corrections = apply_corrections(results['A_sym'], results)

# 4. 生成最终报告
print("\n" + "="*80)
print("最终分析报告")
print("="*80)

print(f"\n原始矩阵问题总结:")
for i, issue in enumerate(results['issues'], 1):
    print(f"  {i}. {issue}")

print(f"\n应用的修正措施:")
if corrections:
    for i, correction in enumerate(corrections, 1):
        print(f"  {i}. {correction}")
else:
    print("  无修正措施应用")

print(f"\n最终矩阵状态:")
try:
    final_eigvals = np.linalg.eigvalsh(A_corrected)
    final_min_eig = np.min(final_eigvals)
    final_max_eig = np.max(final_eigvals)
    
    print(f"  特征值范围: [{final_min_eig:.6e}, {final_max_eig:.6e}]")
    print(f"  条件数: {final_max_eig/abs(final_min_eig):.2e}")
    
    if final_min_eig > 1e-10:
        print(f"  ✓ 矩阵现在是正定的")
    elif final_min_eig > -1e-10:
        print(f"  ⚠ 矩阵接近半正定")
    else:
        print(f"  ✗ 矩阵仍然有负特征值")
    
    # 尝试Cholesky
    try:
        L = np.linalg.cholesky(A_corrected)
        print(f"  ✓ Cholesky分解成功")
    except:
        print(f"  ✗ Cholesky分解仍然失败")
        
except Exception as e:
    print(f"  最终验证失败: {e}")

# 5. 保存修正后的矩阵（如果需要）
save_corrected = input("\n是否要保存修正后的矩阵到文件? (y/n): ")
if save_corrected.lower() == 'y':
    filename = "matrix_corrected.npy"
    np.save(filename, A_corrected)
    print(f"修正后的矩阵已保存到 {filename}")
    
    # 同时保存为文本格式
    np.savetxt("matrix_corrected.txt", A_corrected, fmt='%.12e')
    print(f"文本格式已保存到 matrix_corrected.txt")

print("\n" + "="*80)
print("分析完成！")
print("="*80)

print("\n关键发现:")
print(f"1. 不对称性: 最大误差 {results['asymmetry_stats']['max_diff']:.2e}")
print(f"2. 负特征值: {len(results['negative_eigvals'])}个，最小 {np.min(results['eigenvalues']):.6e}")
print(f"3. 负对角线: {len(results['negative_diag'])}个")

print("\n建议下一步:")
print("1. 检查矩阵构建代码，找出不对称的原因")
print("2. 如果是协方差矩阵，检查数据预处理和计算过程")
print("3. 考虑使用修正后的矩阵进行后续计算")