import matplotlib.pyplot as plt
import numpy as np

# 文件路径
file_path = "/home/syl/GICI-IM/results/debug/sigma_1679304413.400000.raw.txt"

# 使用生成器过滤掉非数值行
def is_valid_line(line):
    try:
        float(line.split()[0])  # 尝试将第一列转换为浮点数
        return True
    except ValueError:
        return False

# 读取并过滤数据
try:
    with open(file_path, "r") as f:
        lines = f.readlines()[2:]  # 跳过前两行
        valid_lines = filter(is_valid_line, lines)
        data = np.loadtxt(valid_lines, usecols=(0, 1, 2))
except Exception as e:
    print(f"无法读取文件: {e}")
    exit(1)

# 检查数据列数
if data.shape[1] < 3:
    print("文件中列数不足三列，无法绘制图像。")
    exit(1)

# 提取三列数据
sigma1 = data[:, 0]
sigma2 = data[:, 1]
sigma3 = data[:, 2]

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(sigma1, label="Sigma 1", color="blue")
plt.plot(sigma2, label="Sigma 2", color="green")
plt.plot(sigma3, label="Sigma 3", color="red")

# 添加图例和标签
plt.title("Sigma Values")
plt.xlabel("Index")
plt.ylabel("Sigma")
plt.legend()
plt.grid()

# 显示图像
plt.savefig("/home/syl/GICI-IM/results/debug/T_plot.raw.png")
print("图像已保存到 /home/syl/GICI-IM/results/debug/T_plot.raw.png")