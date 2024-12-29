import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 12
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")

# 定义分布及其理论参数
distributions = {
    "Uniform": {
        "func": lambda size: np.random.uniform(0, 1, size),
        "mean": 0.5,
        "variance": 1 / 12,
    },
    "Exponential": {
        "func": lambda size: np.random.exponential(1, size),
        "mean": 1,
        "variance": 1,
    },
    "t-Distribution": {
        "func": lambda size: np.random.standard_t(df=5, size=size),
        "mean": 0,
        "variance": 5 / (5 - 2)
    },
    "Chi-Square": {
        "func": lambda size: np.random.chisquare(df=3, size=size),
        "mean": 3,
        "variance": 2 * 3
    },
    # "Poisson": {
    #     "func": lambda size: np.random.poisson(5, size),
    #     "mean": 5,
    #     "variance": 5,
    # },
    # "Binomial": {
    #     "func": lambda size: np.random.binomial(n=10, p=0.5, size=size),
    #     "mean": 10 * 0.5,
    #     "variance": 10 * 0.5 * 0.5,
    # },
    "Pareto": {
        "func": lambda size: np.random.pareto(a=3, size=size) + 1,
        "mean": 3 / (3 - 1),
        "variance": ((1 / (3 - 1)) ** 2) * (3 / (3 - 2)),
    },
    "Lognormal": {
        "func": lambda size: np.random.lognormal(mean=0, sigma=1, size=size),
        "mean": np.exp(0 + 1 ** 2 / 2),
        "variance": (np.exp(1 ** 2) - 1) * np.exp(2 * 0 + 1 ** 2),
    }
}

dist_eng_to_chi = {
    "Uniform": "均匀分布",
    "Exponential": "指数分布",
    "t-Distribution": "t分布",
    "Chi-Square": "分布", 
    "Pareto": "Pareto分布",
    "Lognormal": "对数正态分布"
}


# 样本量和实验次数
sample_sizes = [1, 10, 100, 1000, 10000]
num_experiments = 10000
normalize = True


# 函数：生成样本均值分布
def generate_means(dist_func, sample_size, num_experiments):
    means = [np.mean(dist_func(sample_size)) for _ in range(num_experiments)]
    return np.array(means)


# 函数：计算 KL 散度
def compute_kl_divergence(data, ref_dist):
    hist, bin_edges = np.histogram(data, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ref_pdf = ref_dist.pdf(bin_centers)
    ref_pdf[ref_pdf == 0] = 1e-10  # 避免分母为 0
    kl_div = entropy(hist, ref_pdf)
    return kl_div


# 结果存储
results = {}


# 分布迭代
for dist_name, dist_params in distributions.items():
    results[dist_name] = {}
    dist_func = dist_params["func"]
    mu = dist_params["mean"]
    sigma = np.sqrt(dist_params["variance"])

    for n in sample_sizes:
        means = generate_means(dist_func, n, num_experiments)

        # 是否归一化
        if normalize:
            means = (means - mu) / (sigma / np.sqrt(n))

        # 使用正态分布计算 KL 散度
        ref_dist = norm(loc=0 if normalize else mu, scale=1 if normalize else sigma / np.sqrt(n))

        # 计算 KL 散度
        kl_div = compute_kl_divergence(means, ref_dist)
        results[dist_name][n] = {"means": means, "kl_div": kl_div}


# 可视化函数：绘制子图
def plot_distributions_subplots(results):
    for dist_name, dist_results in results.items():
        # 创建子图：每个样本量对应一个子图
        fig, axes = plt.subplots(1, len(sample_sizes), figsize=(12, 4), sharey=True)

        mu = distributions[dist_name]["mean"]
        sigma = np.sqrt(distributions[dist_name]["variance"])

        for i, (n, metrics) in enumerate(dist_results.items()):
            means = metrics["means"]
            kl_div = metrics["kl_div"]

            ax = axes[i]
            # 绘制直方图
            ax.hist(means, bins=30, density=True, alpha=0.6, label=f"n={n}")

            # 绘制正态分布曲线
            x = np.linspace(min(means), max(means), 1000)
            ax.plot(x, norm.pdf(x, loc=0 if normalize else mu, scale=1 if normalize else sigma / np.sqrt(n)), linewidth=2, label="Fitted Normal")

            # 设置子图标题和图例
            ax.set_title(f"n={n}\nKL={kl_div:.4f}", fontsize=14)
            ax.set_xlabel("标准化的样本均值", fontsize=11, fontproperties=font)
            if i == 0:
                ax.set_ylabel("概率密度", fontproperties=font)

            # 调整刻度字体大小
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # 设置刻度字体样式
            for label in ax.get_xticklabels():
                label.set_fontproperties('serif')
            for label in ax.get_yticklabels():
                label.set_fontproperties('serif')

        plt.tight_layout()
        plt.savefig(f"CLT-Validation-and-Exploration/figures/{dist_name}-new.png", dpi=300)
        # plt.show()


# 调用绘制函数
plot_distributions_subplots(results)


# 可视化函数：绘制 KL 散度随样本量变化的折线图
def plot_kl_divergence_line(results):
    plt.figure(figsize=(8, 6))

    for dist_name, dist_results in results.items():
        sample_sizes = list(dist_results.keys())
        kl_divs = [metrics["kl_div"] for metrics in dist_results.values()]

        plt.plot(sample_sizes, kl_divs, marker='o', label=dist_eng_to_chi[dist_name])

    plt.xscale('log')  # 对数坐标

    plt.xlabel("样本量n", fontsize=15, fontproperties=font)
    plt.ylabel("KL散度", fontsize=15, fontproperties=font)
    plt.grid(True, alpha=0.6)
    plt.legend(fontsize=15, prop=font)
    plt.tight_layout()
    plt.savefig("CLT-Validation-and-Exploration/figures/comp2.png", dpi=300)
    # plt.show()


# 调用函数绘制 KL 散度折线图
plot_kl_divergence_line(results)