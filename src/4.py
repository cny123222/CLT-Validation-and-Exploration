# Re-import necessary libraries after state reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 15
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")

# Define the distributions
x_pareto = np.linspace(1, 10, 1000)
pareto_pdf = (3 * (1 / x_pareto**4))

x_lognormal = np.linspace(0.01, 10, 1000)
lognormal_pdf = (1 / (x_lognormal * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_lognormal)) ** 2) / 2)

# Plot the distributions
plt.figure(figsize=(8, 6))

# Pareto distribution
plt.plot(x_pareto, pareto_pdf, label=r"Pareto分布 ($a=3$)", color="blue")

# Lognormal distribution
plt.plot(x_lognormal, lognormal_pdf, label=r"对数正态分布 ($\mu=0$, $\sigma=1$)", color="green")

# Add labels and legend
plt.xlabel("x")
plt.ylabel("概率密度", fontproperties=font)
plt.ylim(-0.1, 1.5)
plt.legend(prop=font)
plt.tight_layout()
plt.grid(True, alpha=0.6)

# Show plot
plt.savefig("CLT-Validation-and-Exploration/figures/den.png", dpi=300)
# plt.show()