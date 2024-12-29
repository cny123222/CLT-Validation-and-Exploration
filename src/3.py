import numpy as np
from scipy.stats import norm, kstest
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 13
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")

# 设定显著性水平为 0.05
significance_level = 0.05
# 最大样本量，假设到这个样本量时都收敛了
max_sample_size = 100
# 最小样本量起始值
min_sample_size = 1
# 实验次数
num_experiments = 10000
# 多重检验的次数
num_tests = 10


# 函数：生成样本均值分布
def generate_means(dist_func, sample_size, num_experiments):
    means = [np.mean(dist_func(sample_size)) for _ in range(num_experiments)]
    return np.array(means)


# 函数：计算 KS 统计量及判断是否达到显著性水平
def compute_ks_statistic(data, ref_dist):
    statistic, p_value = kstest(data, ref_dist.cdf)
    is_significant = p_value < significance_level
    return statistic, p_value, is_significant


# 使用二分搜索找到 t 分布不同自由度达到不显著的最小样本量，并使用 Benjamini-Hochberg 校正
def find_min_sample_size_for_non_significance_binary_search():
    min_sample_sizes = {}
    # 设定不同的自由度范围
    degrees_of_freedom = range(3, 16)
    for df in degrees_of_freedom:
        print(f"正在处理自由度为 {df} 的 t 分布")
        dist_func = lambda size: np.random.standard_t(df=df, size=size)
        mu = 0  # t 分布的均值（当 df > 1 时，均值为 0）
        sigma = np.sqrt(df / (df - 2)) if df > 2 else 1  # t 分布的方差（当 df > 2 时，方差为 df / (df - 2)）

        left = min_sample_size
        right = max_sample_size

        while left < right:
            mid = left + (right - left) // 2
            p_values = []
            for _ in range(num_tests):
                means = generate_means(dist_func, mid, num_experiments)
                means = (means - mu) / (sigma / np.sqrt(mid))
                ref_dist = norm(loc=0, scale=1)
                _, p_value, _ = compute_ks_statistic(means, ref_dist)
                p_values.append(p_value)
            # Benjamini-Hochberg 校正
            sorted_p_values = np.sort(p_values)
            m = len(p_values)
            for i, p_value in enumerate(sorted_p_values):
                if p_value <= (i + 1) * significance_level / m:
                    adjusted_p_value = p_value * m / (i + 1)
                    if adjusted_p_value > significance_level:
                        break
            is_significant = p_value < significance_level

            if is_significant:
                left = mid + 1
            else:
                right = mid

        min_sample_sizes[df] = left
    return min_sample_sizes


# 调用函数找到最小样本量
min_sample_sizes = find_min_sample_size_for_non_significance_binary_search()


# 提取自由度和最小样本量用于绘图
df_list = list(min_sample_sizes.keys())
min_sample_size_list = list(min_sample_sizes.values())


# 绘制折线图
plt.plot(df_list, min_sample_size_list, marker='o')
plt.yscale('log')
plt.xlim(2, 16)
plt.ylim(0, 100)
plt.xlabel('t分布的自由度m', fontproperties=font)
plt.ylabel(r'达到不显著的最小样本量$n_{\min}$', fontproperties=font)
plt.grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig(f"CLT-Validation-and-Exploration/figures/t3.png", dpi=300)
# plt.show()


# 打印结果
for df, min_sample_size in min_sample_sizes.items():
    print(f"自由度为 {df} 的 t 分布达到不显著的最小样本量为: {min_sample_size}")