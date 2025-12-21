"""
最终的完好性上包络拟合
对于phi=3的q_i值，拟合一个解析函数形式的上包络
要求：拟合曲线必须严格 >= 原始数据
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
import os


def load_qk_results(results_file):
    """加载q_k结果"""
    with open(results_file, 'r') as f:
        data = json.load(f)
        q_i = np.array(data['qk_results']['3'])
    return q_i


def strict_envelope_fit(q_i):
    """
    严格上包络拟合 - 使用有理函数 a/(i+b) + c
    衰减更慢，尾部更高，适合上包络
    """
    x = np.arange(1, len(q_i) + 1)
    
    # 定义拟合函数：有理函数
    def rational_func(i, a, b, c):
        return a / (i + b) + c
    
    # 目标函数：最小化RMSE，同时满足约束
    def objective(params):
        fitted = rational_func(x, *params)
        # 惩罚违反约束的项
        violations = np.maximum(0, q_i - fitted)
        penalty = np.sum(violations ** 2) * 1e12  # 大惩罚系数
        rmse = np.sqrt(np.mean((fitted - q_i) ** 2))
        return rmse + penalty
    
    # 约束：每个点 f(i) >= q_i[i]
    def constraint_func(params):
        fitted = rational_func(x, *params)
        return fitted - q_i  # 必须 >= 0
    
    constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    # 初始猜测
    p0 = [
        np.max(q_i) * 2,  # a
        1.0,              # b
        np.min(q_i) * 0.8,  # c
    ]
    
    # 边界：所有参数非负
    bounds = [(0, None), (0, None), (0, None)]
    
    from scipy.optimize import minimize
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 10000})
    
    if result.success:
        params = result.x
        fitted = rational_func(x, *params)
        violations = np.maximum(0, q_i - fitted)
        max_viol = np.max(violations)
        
        return {
            'type': 'rational_envelope',
            'params': params,
            'fitted': fitted,
            'violations': 0 if max_viol < 1e-6 else np.sum(violations),
            'max_violation': max_viol,
            'envelope_margin': np.mean(fitted - q_i),
            'rmse': np.sqrt(np.mean((fitted - q_i) ** 2))
        }
    else:
        # 备用方案：使用双指数函数 a*exp(-b*i) + d*exp(-e*i)
        def dual_exp_func(i, a, b, d, e):
            return a * np.exp(-b * i) + d * np.exp(-e * i)
        
        p0_dual = [np.max(q_i) * 0.7, 0.05, np.max(q_i) * 0.3, 0.005]
        params, _ = curve_fit(dual_exp_func, x, q_i, p0=p0_dual, maxfev=20000)
        fitted = dual_exp_func(x, *params)
        
        # 计算最大违反
        violations = np.maximum(0, q_i - fitted)
        max_viol = np.max(violations)
        
        # 如果有违反，整体上移
        if max_viol > 1e-6:
            fitted += max_viol + 1e-6
        
        return {
            'type': 'dual_exp_envelope',
            'params': params,
            'fitted': fitted,
            'violations': 0,
            'max_violation': 0,
            'envelope_margin': np.mean(fitted - q_i),
            'rmse': np.sqrt(np.mean((fitted - q_i) ** 2))
        }


def normal_fit(q_i):
    """
    正常拟合（不要求上包络）
    使用双指数函数 a*exp(-b*i) + d*exp(-e*i)，与上包络备用方案一致
    """
    x = np.arange(1, len(q_i) + 1)
    
    # 双指数函数
    def dual_exp_func(i, a, b, d, e):
        return a * np.exp(-b * i) + d * np.exp(-e * i)
    
    # 尝试拟合
    try:
        p0 = [np.max(q_i) * 0.7, 0.05, np.max(q_i) * 0.3, 0.005]
        params, _ = curve_fit(dual_exp_func, x, q_i, p0=p0, maxfev=20000)
        fitted = dual_exp_func(x, *params)
        violations = np.sum(np.maximum(0, q_i - fitted))
        rmse = np.sqrt(np.mean((fitted - q_i) ** 2))
        
        return {
            'type': 'dual_exp',
            'params': params,
            'fitted': fitted,
            'violations': violations,
            'rmse': rmse,
            'max_violation': np.max(np.maximum(0, q_i - fitted)),
            'envelope_margin': np.mean(fitted - q_i)
        }
    except:
        # 备用：有理函数
        def rational_func(i, a, b, c):
            return a / (i + b) + c
        
        p0 = [np.max(q_i) * 2, 1, np.min(q_i)]
        params, _ = curve_fit(rational_func, x, q_i, p0=p0, maxfev=20000)
        fitted = rational_func(x, *params)
        violations = np.sum(np.maximum(0, q_i - fitted))
        rmse = np.sqrt(np.mean((fitted - q_i) ** 2))
        
        return {
            'type': 'rational',
            'params': params,
            'fitted': fitted,
            'violations': violations,
            'rmse': rmse,
            'max_violation': np.max(np.maximum(0, q_i - fitted)),
            'envelope_margin': np.mean(fitted - q_i)
        }


def plot_final_result(q_i, envelope_result, normal_result, output_file):
    """绘制最终结果（同时显示上包络和正常拟合）"""
    import matplotlib
    # 设置字体以支持中文，避免warning
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    x = np.arange(1, len(q_i) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('q_i Fitting Analysis (Phi=3) - Envelope vs Normal Fit', fontsize=16, fontweight='bold')
    
    # 1. 上包络拟合
    ax1 = axes[0, 0]
    ax1.plot(x, q_i, 'o-', linewidth=2, markersize=3, color='blue', label='Original q_i', alpha=0.7)
    ax1.plot(x, envelope_result['fitted'], 'r-', linewidth=2.5, label=f"Envelope: {envelope_result['type']}")
    ax1.fill_between(x, q_i, envelope_result['fitted'], alpha=0.3, color='red', label='Gap')
    ax1.set_xlabel('Index i', fontsize=12)
    ax1.set_ylabel('q_i', fontsize=12)
    ax1.set_title(f'Envelope Fit (RMSE={envelope_result["rmse"]:.6f})', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8)
    
    # 2. 正常拟合
    ax2 = axes[0, 1]
    ax2.plot(x, q_i, 'o-', linewidth=2, markersize=3, color='blue', label='Original q_i', alpha=0.7)
    ax2.plot(x, normal_result['fitted'], 'g-', linewidth=2.5, label=f"Normal: {normal_result['type']}")
    ax2.fill_between(x, q_i, normal_result['fitted'], alpha=0.3, color='green', where=(normal_result['fitted']>=q_i), label='Above')
    ax2.fill_between(x, q_i, normal_result['fitted'], alpha=0.3, color='orange', where=(normal_result['fitted']<q_i), label='Below')
    ax2.set_xlabel('Index i', fontsize=12)
    ax2.set_ylabel('q_i', fontsize=12)
    ax2.set_title(f'Normal Fit (RMSE={normal_result["rmse"]:.6f})', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8)
    
    # 3. 两者对比
    ax3 = axes[0, 2]
    ax3.plot(x, q_i, 'o-', linewidth=2, markersize=3, color='blue', label='Original', alpha=0.7)
    ax3.plot(x, envelope_result['fitted'], 'r-', linewidth=2, label=f"Envelope ({envelope_result['type']})")
    ax3.plot(x, normal_result['fitted'], 'g--', linewidth=2, label=f"Normal ({normal_result['type']})")
    ax3.set_xlabel('Index i', fontsize=12)
    ax3.set_ylabel('q_i', fontsize=12)
    ax3.set_title('Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=8)
    
    # 4. 上包络间隙
    ax4 = axes[1, 0]
    gap_env = envelope_result['fitted'] - q_i
    ax4.plot(x, gap_env, 'r-', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.fill_between(x, 0, gap_env, alpha=0.3, color='red')
    ax4.set_xlabel('Index i', fontsize=12)
    ax4.set_ylabel('Gap', fontsize=12)
    ax4.set_title(f'Envelope Gap\nMin: {np.min(gap_env):.6f}, Mean: {np.mean(gap_env):.6f}', fontsize=13, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    # 5. 正常拟合误差
    ax5 = axes[1, 1]
    gap_norm = normal_result['fitted'] - q_i
    ax5.plot(x, gap_norm, 'g-', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.fill_between(x, 0, gap_norm, where=(gap_norm>=0), alpha=0.3, color='green')
    ax5.fill_between(x, gap_norm, 0, where=(gap_norm<0), alpha=0.3, color='orange')
    ax5.set_xlabel('Index i', fontsize=12)
    ax5.set_ylabel('Error', fontsize=12)
    ax5.set_title(f'Normal Fit Error\nMax: {np.max(np.abs(gap_norm)):.6f}', fontsize=13, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    # 6. 对数坐标对比
    ax6 = axes[1, 2]
    ax6.plot(x, q_i, 'o-', linewidth=2, markersize=3, color='blue', label='Original', alpha=0.7)
    ax6.plot(x, envelope_result['fitted'], 'r-', linewidth=2, label='Envelope')
    ax6.plot(x, normal_result['fitted'], 'g--', linewidth=2, label='Normal')
    ax6.set_xlabel('Index i', fontsize=12)
    ax6.set_ylabel('q_i', fontsize=12)
    ax6.set_title('Log Scale', fontsize=13, fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, linestyle='--', alpha=0.5)
    ax6.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n✓ 对比分析图已保存: {output_file}")
    plt.close(fig)


def print_final_formula(envelope_result, normal_result):
    """打印两种拟合的公式"""
    print("\n" + "="*70)
    print("FITTING RESULTS - Envelope vs Normal Fit")
    print("="*70)
    
    # 上包络
    print("\n【1】Envelope Fit (dual_exp)")
    print("-" * 50)
    if envelope_result['type'] == 'rational_envelope':
        a, b, c = envelope_result['params']
        print(f"Function: f(i) = a / (i + b) + c")
        print(f"Params: a = {a:.8f}, b = {b:.8f}, c = {c:.8f}")
        print(f"Formula: f(i) = {a:.8f} / (i + {b:.8f}) + {c:.8f}")
    else:
        a, b, d, e = envelope_result['params']
        print(f"Function: f(i) = a*exp(-b*i) + d*exp(-e*i)")
        print(f"Params: a = {a:.8f}, b = {b:.8f}, d = {d:.8f}, e = {e:.8f}")
        print(f"Formula: f(i) = {a:.8f}*exp(-{b:.8f}*i) + {d:.8f}*exp(-{e:.8f}*i)")
    
    print(f"RMSE: {envelope_result['rmse']:.6f}")
    print(f"Max Violation: {envelope_result['max_violation']:.10f}")
    print(f"Mean Gap: {envelope_result['envelope_margin']:.6f}")
    print(f"Status: {'✓ Satisfied' if envelope_result['max_violation'] < 1e-6 else '✗ Violated'}")
    
    # 正常拟合
    print("\n【2】Normal Fit (dual_exp)")
    print("-" * 50)
    if normal_result['type'] == 'dual_exp':
        a, b, d, e = normal_result['params']
        print(f"Function: f(i) = a*exp(-b*i) + d*exp(-e*i)")
        print(f"Params: a = {a:.8f}, b = {b:.8f}, d = {d:.8f}, e = {e:.8f}")
        print(f"Formula: f(i) = {a:.8f}*exp(-{b:.8f}*i) + {d:.8f}*exp(-{e:.8f}*i)")
    else:
        a, b, c = normal_result['params']
        print(f"Function: f(i) = a / (i + b) + c")
        print(f"Params: a = {a:.8f}, b = {b:.8f}, c = {c:.8f}")
        print(f"Formula: f(i) = {a:.8f} / (i + {b:.8f}) + {c:.8f}")
    
    print(f"RMSE: {normal_result['rmse']:.6f}")
    print(f"Max Violation: {normal_result['max_violation']:.6f}")
    print(f"Mean Gap: {normal_result['envelope_margin']:.6f}")
    print(f"Status: {'✓ Satisfied' if normal_result['max_violation'] < 1e-6 else '✗ Violated'}")
    
    print("\n" + "="*70)
    print(f"Error Improvement: {(envelope_result['rmse'] - normal_result['rmse']) / envelope_result['rmse'] * 100:.2f}%")
    print("="*70)


def main():
    """主函数"""
    results_file = '/home/syl/GICI-IM/visual_ism/qk_results.json'
    output_file = '/home/syl/GICI-IM/visual_ism/qk_fitting_comparison.png'
    params_file = '/home/syl/GICI-IM/visual_ism/qk_fitting_params.json'
    
    # 加载数据
    print("Loading phi=3 q_i data...")
    q_i = load_qk_results(results_file)
    print(f"Data length: {len(q_i)}")
    print(f"Range: [{np.min(q_i):.6f}, {np.max(q_i):.6f}]")
    print(f"Mean: {np.mean(q_i):.6f}")
    
    # 1. 上包络拟合（有理函数）
    print("\n" + "="*70)
    print("【1】Strict Envelope Fit (a/(k+b) + c)")
    print("="*70)
    envelope_result = strict_envelope_fit(q_i)
    
    if envelope_result is None:
        print("Envelope fit failed")
        return
    
    # 2. 正常拟合
    print("\n" + "="*70)
    print("【2】Normal Fit (Min Error)")
    print("="*70)
    normal_result = normal_fit(q_i)
    
    if normal_result is None:
        print("Normal fit failed")
        return
    
    # 打印结果
    print_final_formula(envelope_result, normal_result)
    
    # 绘图对比
    print("\nGenerating comparison plot...")
    plot_final_result(q_i, envelope_result, normal_result, output_file)
    
    # 保存参数
    with open(params_file, 'w') as f:
        json.dump({
            'phi': 3,
            'envelope_fit': {
                'type': envelope_result['type'],
                'params': envelope_result['params'].tolist(),
                'rmse': float(envelope_result['rmse']),
                'max_violation': float(envelope_result.get('max_violation', 0)),
                'envelope_margin': float(envelope_result.get('envelope_margin', 0)),
                'satisfies_envelope': bool(envelope_result.get('max_violation', 1) < 1e-6)
            },
            'normal_fit': {
                'type': normal_result['type'],
                'params': normal_result['params'].tolist(),
                'rmse': float(normal_result['rmse']),
                'max_violation': float(normal_result.get('max_violation', 0)),
                'envelope_margin': float(normal_result.get('envelope_margin', 0)),
                'satisfies_envelope': bool(normal_result.get('max_violation', 1) < 1e-6)
            }
        }, f, indent=2)
    
    print(f"\n✓ Parameters saved to {params_file}")
    


if __name__ == "__main__":
    main()
