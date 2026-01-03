import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import linalg

class SigmaInfAnalyzer:
    def __init__(self, debug_dir):
        self.debug_dir = Path(debug_dir)
        self.load_data()
        self.setup_plotting()
        
    def setup_plotting(self):
        """设置绘图参数"""
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        sns.set_style("whitegrid")
        
    def load_data(self):
        """加载数据文件"""
        print("加载调试数据...")
        data_files = {
            'sig2_int': 'sig2_int_debug.txt',
            'sig2_acc': 'sig2_acc_debug.txt', 
            's1vec': 's1vec_debug.txt',
            's2vec': 's2vec_debug.txt',
            's3vec': 's3vec_debug.txt',
            'sigma': 'sigma_debug.txt',
            'sigma_ss': 'sigma_ss_debug.txt'
        }
        
        self.data = {}
        for key, filename in data_files.items():
            filepath = self.debug_dir / filename
            if filepath.exists():
                try:
                    # 方法1: 先读取整个文件，找到有效数据部分
                    self.data[key] = self.smart_load_matrix(filepath)
                    print(f"  {key}: shape = {self.data[key].shape}")
                    
                except Exception as e:
                    print(f"  加载 {filename} 时出错: {e}")
                    print(f"  尝试备用加载方法...")
                    self.data[key] = self.robust_load_matrix(filepath)
            else:
                print(f"  警告: 文件 {filename} 不存在")
                
        # 检查维度
        n_sets = len(self.data.get('sigma', []))
        print(f"\n共有 {n_sets} 个集合")
        
    def smart_load_matrix(self, filepath):
        """智能加载矩阵，处理格式问题"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 1. 找到数据开始的行
        data_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Matrix shape:") or line == "Matrix:":
                continue
            # 尝试解析第一行数据
            parts = line.split()
            try:
                [float(x) for x in parts]
                data_start = i
                break
            except:
                continue
        
        # 2. 收集有效数据行
        valid_lines = []
        expected_cols = None
        
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            parts = line.split()
            # 检查是否为数字
            try:
                row_data = [float(x) for x in parts]
                if expected_cols is None:
                    expected_cols = len(row_data)
                    valid_lines.append(row_data)
                elif len(row_data) == expected_cols:
                    valid_lines.append(row_data)
                else:
                    # 列数不一致，停止读取
                    print(f"    在行 {i+1} 检测到列数变化: {expected_cols} -> {len(row_data)}，停止读取")
                    break
            except ValueError:
                # 非数字行，停止读取
                print(f"    在行 {i+1} 检测到非数字内容，停止读取")
                break
        
        if not valid_lines:
            raise ValueError(f"未找到有效数据: {filepath}")
        
        return np.array(valid_lines)

    def robust_load_matrix(self, filepath):
        """更健壮的矩阵加载方法"""
        print(f"    使用健壮加载方法读取 {filepath.name}...")
        
        # 读取文件内容
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 分割成行
        lines = content.split('\n')
        
        # 解析数据
        data_rows = []
        in_data_section = False
        expected_cols = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 跳过标题行
            if "Matrix shape:" in line or line == "Matrix:":
                in_data_section = True
                continue
                
            # 检查是否为数据行
            parts = line.split()
            if not parts:
                continue
                
            # 尝试转换为数字
            try:
                row_data = []
                for part in parts:
                    row_data.append(float(part))
                
                # 如果是第一行有效数据，设置期望列数
                if expected_cols is None:
                    expected_cols = len(row_data)
                    data_rows.append(row_data)
                elif len(row_data) == expected_cols:
                    data_rows.append(row_data)
                else:
                    # 列数变化，可能是文件末尾的附加信息
                    print(f"      行 {i+1}: 列数从 {expected_cols} 变为 {len(row_data)}，忽略后续内容")
                    break
                    
            except ValueError:
                # 非数字行，可能是文件末尾的其他内容
                print(f"      行 {i+1}: 包含非数字内容，忽略后续内容")
                break
        
        if not data_rows:
            # 如果上述方法失败，尝试使用np.genfromtxt的灵活模式
            print(f"      尝试使用genfromtxt读取...")
            try:
                data = np.genfromtxt(filepath, 
                                    skip_header=2,  # 跳过"Matrix shape:"和"Matrix:"
                                    invalid_raise=False)  # 忽略错误行
                
                # 移除NaN行
                mask = ~np.isnan(data).all(axis=1)
                data = data[mask]
                return data
            except Exception as e:
                print(f"      genfromtxt也失败: {e}")
                return np.array([])
        
        return np.array(data_rows)
            
    def find_inf_sigma_rows(self):
        """找出sigma为inf的行"""
        sigma = self.data.get('sigma')
        if sigma is None:
            print("未找到sigma数据")
            return []
            
        inf_rows = []
        inf_positions = []
        
        for i in range(len(sigma)):
            for k in range(sigma.shape[1]):
                if np.isinf(sigma[i, k]) or np.isnan(sigma[i, k]):
                    inf_rows.append(i)
                    inf_positions.append((i, k))
                    
        # 去重并排序
        inf_rows = sorted(set(inf_rows))
        
        print(f"\n发现 {len(inf_rows)} 个集合包含inf/nan值")
        print(f"总共 {len(inf_positions)} 个inf/nan位置")
        
        if inf_positions:
            print("\n前20个inf/nan位置:")
            for i, (row_idx, state_idx) in enumerate(inf_positions[:20]):
                print(f"  集合 {row_idx}, 状态量 {state_idx}: sigma = {sigma[row_idx, state_idx]}")
                
        return inf_rows, inf_positions
    
    def analyze_inf_causes(self, inf_rows, inf_positions):
        """深入分析inf的原因"""
        print("\n" + "="*80)
        print("深入分析sigma为inf的原因")
        print("="*80)
        
        # 获取相关数据
        sig2_int = self.data['sig2_int']
        s1vec = self.data['s1vec']
        s2vec = self.data['s2vec']
        s3vec = self.data['s3vec']
        
        # 分析第一个inf位置的详细信息
        if inf_positions:
            first_inf = inf_positions[0]
            row_idx, state_idx = first_inf
            
            print(f"\n详细分析第一个inf位置 (集合 {row_idx}, 状态量 {state_idx}):")
            
            # 获取对应的s向量
            if state_idx == 0:
                s_vec = s1vec[row_idx] if len(s1vec.shape) > 1 else s1vec
            elif state_idx == 1:
                s_vec = s2vec[row_idx] if len(s2vec.shape) > 1 else s2vec
            else:
                s_vec = s3vec[row_idx] if len(s3vec.shape) > 1 else s3vec
                
            print(f"  s向量维度: {s_vec.shape}")
            print(f"  s向量范数: {np.linalg.norm(s_vec):.6e}")
            
            # 计算方差
            var = s_vec.T @ sig2_int @ s_vec
            print(f"  计算方差: s^T * sig2_int * s = {var:.6e}")
            
            # 检查sig2_int的性质
            print(f"\n  检查sig2_int矩阵:")
            print(f"    维度: {sig2_int.shape}")
            print(f"    条件数: {np.linalg.cond(sig2_int):.6e}")
            
            # 检查特征值
            eigenvalues = np.linalg.eigvalsh(sig2_int)  # 使用eigvalsh确保对称矩阵
            print(f"    最小特征值: {np.min(eigenvalues):.6e}")
            print(f"    最大特征值: {np.max(eigenvalues):.6e}")
            print(f"    负特征值数量: {np.sum(eigenvalues < 0)}")
            print(f"    零特征值数量: {np.sum(np.abs(eigenvalues) < 1e-10)}")
            
            # 检查s向量和sig2_int的关系
            print(f"\n  检查s向量和sig2_int的关系:")
            
            # 计算sig2_int * s
            sig2_int_s = sig2_int @ s_vec
            print(f"    sig2_int * s 的范数: {np.linalg.norm(sig2_int_s):.6e}")
            
            # 检查s是否在sig2_int的零空间
            zero_space_check = np.linalg.norm(sig2_int_s)
            if zero_space_check < 1e-10:
                print(f"    ⚠️  s向量可能在sig2_int的零空间中!")
                print(f"    sig2_int * s 的范数: {zero_space_check:.6e}")
            else:
                print(f"    s向量不在sig2_int的零空间中")
                
            # 检查方差是否为负
            if var < 0:
                print(f"\n  ⚠️  警告: 方差为负值! var = {var:.6e}")
                print(f"    这会导致sqrt(var)为NaN或inf")
            elif var == 0:
                print(f"\n  ⚠️  警告: 方差为零! var = {var:.6e}")
                print(f"    这会导致sqrt(var)为0，但代码可能设置为inf")
            else:
                print(f"\n  方差为正: var = {var:.6e}")
                print(f"    标准差应为: sqrt(var) = {np.sqrt(var):.6f}")
                
        # 分析多个inf位置的模式
        if len(inf_positions) > 5:
            print(f"\n{'='*80}")
            print("分析inf位置的模式")
            print(f"{'='*80}")
            
            # 按状态量统计
            state_counts = {}
            for _, state_idx in inf_positions:
                state_counts[state_idx] = state_counts.get(state_idx, 0) + 1
                
            print(f"inf按状态量分布:")
            for state_idx in range(3):
                count = state_counts.get(state_idx, 0)
                print(f"  状态量 {state_idx}: {count} 个inf ({count/len(inf_positions)*100:.1f}%)")
                
            # 检查s向量的统计特性
            print(f"\ninf位置s向量的统计特性:")
            s_norms = []
            for row_idx, state_idx in inf_positions[:100]:  # 只检查前100个
                if state_idx == 0:
                    s_vec = s1vec[row_idx] if len(s1vec.shape) > 1 else s1vec
                elif state_idx == 1:
                    s_vec = s2vec[row_idx] if len(s2vec.shape) > 1 else s2vec
                else:
                    s_vec = s3vec[row_idx] if len(s3vec.shape) > 1 else s3vec
                    
                s_norms.append(np.linalg.norm(s_vec))
                
            if s_norms:
                print(f"  s向量范数 - 最小值: {np.min(s_norms):.6e}")
                print(f"  s向量范数 - 最大值: {np.max(s_norms):.6e}")
                print(f"  s向量范数 - 平均值: {np.mean(s_norms):.6e}")
                
    def visualize_analysis(self, inf_rows, inf_positions):
        """可视化分析结果"""
        fig, axes = plt.subplots
        
        # 数据准备
        sigma = self.data['sigma']
        s1vec = self.data['s1vec']
        s2vec = self.data['s2vec']
        s3vec = self.data['s3vec']
        
        # 1. sigma值的分布
        ax = axes[0, 0]
        finite_sigma = sigma[~np.isinf(sigma) & ~np.isnan(sigma)]
        if len(finite_sigma) > 0:
            ax.hist(finite_sigma.flatten(), bins=50, alpha=0.7, label='有限值')
        ax.set_title('Sigma值分布')
        ax.set_xlabel('Sigma值')
        ax.set_ylabel('频数')
        ax.legend()
        ax.text(0.05, 0.95, f'inf/nan数量: {np.sum(np.isinf(sigma) | np.isnan(sigma))}',
                transform=ax.transAxes, verticalalignment='top')
        
        # 2. inf位置的分布
        ax = axes[0, 1]
        if inf_positions:
            rows, states = zip(*inf_positions)
            ax.scatter(rows, states, alpha=0.6, s=10)
            ax.set_title('Inf位置分布')
            ax.set_xlabel('集合索引')
            ax.set_ylabel('状态量索引')
            ax.set_yticks([0, 1, 2])
        else:
            ax.text(0.5, 0.5, '未发现inf值', ha='center', va='center')
            
        # 3. s向量范数的分布
        ax = axes[0, 2]
        s_norms = []
        for i in range(min(1000, len(s1vec))):
            s_norms.append(np.linalg.norm(s1vec[i] if len(s1vec.shape) > 1 else s1vec))
            
        ax.hist(s_norms, bins=50, alpha=0.7)
        ax.set_title('S向量范数分布 (s1vec)')
        ax.set_xlabel('范数')
        ax.set_ylabel('频数')
        
        # 4. 方差计算过程
        ax = axes[1, 0]
        if inf_positions:
            # 计算前10个inf位置的方差
            variances = []
            for row_idx, state_idx in inf_positions[:10]:
                if state_idx == 0:
                    s_vec = s1vec[row_idx] if len(s1vec.shape) > 1 else s1vec
                elif state_idx == 1:
                    s_vec = s2vec[row_idx] if len(s2vec.shape) > 1 else s2vec
                else:
                    s_vec = s3vec[row_idx] if len(s3vec.shape) > 1 else s3vec
                    
                var = s_vec.T @ self.data['sig2_int'] @ s_vec
                variances.append(var)
                
            ax.bar(range(len(variances)), variances, alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Inf位置的方差计算')
            ax.set_xlabel('inf位置索引')
            ax.set_ylabel('方差值')
            
        # 5. sig2_int的特征值分布
        ax = axes[1, 1]
        eigenvalues = np.linalg.eigvalsh(self.data['sig2_int'])
        positive_eigs = eigenvalues[eigenvalues > 0]
        zero_eigs = eigenvalues[np.abs(eigenvalues) < 1e-10]
        negative_eigs = eigenvalues[eigenvalues < 0]
        
        labels = ['正特征值', '零特征值', '负特征值']
        counts = [len(positive_eigs), len(zero_eigs), len(negative_eigs)]
        
        ax.bar(labels, counts, alpha=0.7, color=['green', 'orange', 'red'])
        ax.set_title('sig2_int特征值分布')
        ax.set_ylabel('数量')
        
        # 6. 各状态量的inf比例
        ax = axes[1, 2]
        if inf_positions:
            state_counts = {}
            for _, state_idx in inf_positions:
                state_counts[state_idx] = state_counts.get(state_idx, 0) + 1
                
            states = list(range(3))
            counts = [state_counts.get(s, 0) for s in states]
            
            ax.bar(states, counts, alpha=0.7)
            ax.set_title('各状态量的Inf数量')
            ax.set_xlabel('状态量索引')
            ax.set_ylabel('Inf数量')
            ax.set_xticks(states)
            
        plt.tight_layout()
        plt.savefig(self.debug_dir / 'sigma_inf_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def detailed_diagnosis(self, target_rows=None):
        """对特定行进行详细诊断"""
        print(f"\n{'='*80}")
        print("详细诊断模式")
        print(f"{'='*80}")
        
        sig2_int = self.data['sig2_int']
        
        if target_rows is None:
            # 自动选择几个有问题的行
            sigma = self.data['sigma']
            inf_rows = []
            for i in range(len(sigma)):
                if np.any(np.isinf(sigma[i]) | np.isnan(sigma[i])):
                    inf_rows.append(i)
                    if len(inf_rows) >= 3:
                        break
            target_rows = inf_rows[:3]
            
        for row_idx in target_rows:
            print(f"\n{'─'*40}")
            print(f"诊断集合 {row_idx}:")
            print(f"{'─'*40}")
            
            for state_idx in range(3):
                print(f"\n  状态量 {state_idx}:")
                
                # 获取s向量
                if state_idx == 0:
                    s_vec = self.data['s1vec'][row_idx] if len(self.data['s1vec'].shape) > 1 else self.data['s1vec']
                elif state_idx == 1:
                    s_vec = self.data['s2vec'][row_idx] if len(self.data['s2vec'].shape) > 1 else self.data['s2vec']
                else:
                    s_vec = self.data['s3vec'][row_idx] if len(self.data['s3vec'].shape) > 1 else self.data['s3vec']
                    
                # 计算中间结果
                sig2_int_s = sig2_int @ s_vec
                dot_product = np.dot(s_vec, sig2_int_s)
                
                print(f"    s向量范数: {np.linalg.norm(s_vec):.6e}")
                print(f"    sig2_int * s 范数: {np.linalg.norm(sig2_int_s):.6e}")
                print(f"    点积 s·(sig2_int·s): {dot_product:.6e}")
                
                # 检查可能的数值问题
                if np.abs(dot_product) < 1e-100:
                    print(f"    ⚠️  点积接近零!")
                elif dot_product < 0:
                    print(f"    ⚠️  点积为负!")
                    
                # 检查s向量元素
                print(f"    s向量 - 最小值: {np.min(s_vec):.6e}, 最大值: {np.max(s_vec):.6e}")
                print(f"    s向量 - 绝对值之和: {np.sum(np.abs(s_vec)):.6e}")
                
                # 计算实际的sigma
                if dot_product > 0:
                    actual_sigma = np.sqrt(dot_product)
                    stored_sigma = self.data['sigma'][row_idx, state_idx]
                    print(f"    计算的标准差: {actual_sigma:.6f}")
                    print(f"    存储的标准差: {stored_sigma}")
                    
                    if np.isinf(stored_sigma) or np.isnan(stored_sigma):
                        print(f"    ❗ 存储值为inf/nan，但计算值为有限值!")
                        print(f"    可能的原因: 存储时使用了错误的计算方式")
                        
    def run_full_analysis(self):
        """运行完整分析"""
        print("="*80)
        print("Sigma Inf 分析器")
        print("="*80)
        
        # 1. 找出inf的行
        inf_rows, inf_positions = self.find_inf_sigma_rows()
        
        if not inf_positions:
            print("\n✓ 未发现sigma为inf/nan的值")
            return
            
        # 2. 分析原因
        self.analyze_inf_causes(inf_rows, inf_positions)
        
        # 3. 可视化
        print("\n生成可视化分析图表...")
        self.visualize_analysis(inf_rows, inf_positions)
        
        # 4. 详细诊断
        print("\n进行详细诊断...")
        self.detailed_diagnosis(inf_rows[:5])
        
        print(f"\n{'='*80}")
        print("分析完成")
        print(f"{'='*80}")
        
        # 总结建议
        print("\n可能的问题和解决建议:")
        print("1. sig2_int可能不是正定矩阵，包含零或负特征值")
        print("2. s向量可能在某些方向上与sig2_int的零空间对齐")
        print("3. 数值精度问题导致方差计算为负或零")
        print("4. 代码中计算sigma时可能未正确处理边界情况")
        print("\n建议:")
        print("- 检查sig2_int是否应为正定矩阵")
        print("- 在计算sqrt前添加小正数避免零方差: sqrt(max(var, 1e-12))")
        print("- 检查s向量的计算是否正确")
        print("- 考虑使用Cholesky分解验证正定性")

# 使用示例
if __name__ == "__main__":
    # 设置调试文件目录
    debug_dir = "/home/dell/sunyulong/GICI-IM/results/debug"
    
    # 创建分析器
    analyzer = SigmaInfAnalyzer(debug_dir)
    
    # 运行完整分析
    analyzer.run_full_analysis()