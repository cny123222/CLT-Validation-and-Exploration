import numpy as np
from scipy.stats import norm, kstest
from matplotlib.font_manager import FontProperties

# Define settings for hypothesis testing
significance_level = 0.05
max_sample_size = 100000
min_sample_size = 1
num_experiments = 10000
num_tests = 10

# Define distributions and their parameters
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
        "variance": 5 / (5 - 2),
    },
    "Chi-Square": {
        "func": lambda size: np.random.chisquare(df=3, size=size),
        "mean": 3,
        "variance": 2 * 3,
    },
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

# Function to generate sample means
def generate_means(dist_func, sample_size, num_experiments):
    means = [np.mean(dist_func(sample_size)) for _ in range(num_experiments)]
    return np.array(means)

# Function to compute KS statistic and significance
def compute_ks_statistic(data, ref_dist):
    statistic, p_value = kstest(data, ref_dist.cdf)
    is_significant = p_value < significance_level
    return statistic, p_value, is_significant

# Find minimum sample size for non-significance using binary search
def find_min_sample_size_for_distributions(distributions):
    min_sample_sizes = {}
    for dist_name, dist_params in distributions.items():
        print(f"Processing distribution: {dist_name}")
        dist_func = dist_params["func"]
        mu = dist_params["mean"]
        sigma = np.sqrt(dist_params["variance"])

        left = min_sample_size
        right = max_sample_size

        while left < right:
            mid = left + (right - left) // 2
            print(f"Current sample size: {mid}")
            p_values = []
            for _ in range(num_tests):
                means = generate_means(dist_func, mid, num_experiments)
                means = (means - mu) / (sigma / np.sqrt(mid))
                ref_dist = norm(loc=0, scale=1)
                _, p_value, _ = compute_ks_statistic(means, ref_dist)
                p_values.append(p_value)
            # Benjamini-Hochberg correction
            sorted_p_values = np.sort(p_values)
            m = len(p_values)
            is_significant = False
            for i, p_value in enumerate(sorted_p_values):
                if p_value <= (i + 1) * significance_level / m:
                    adjusted_p_value = p_value * m / (i + 1)
                    if adjusted_p_value > significance_level:
                        break
                    is_significant = True

            if is_significant:
                left = mid + 1
            else:
                right = mid

        min_sample_sizes[dist_name] = left
        print(f"Minimum sample size for {dist_name}: {left}")
    return min_sample_sizes

# Run the analysis
min_sample_sizes = find_min_sample_size_for_distributions(distributions)
print(min_sample_sizes)