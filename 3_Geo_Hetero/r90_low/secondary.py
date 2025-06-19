import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from collections import Counter

# Load and preprocess
data = pd.read_csv("./simulation_outputs/secondary_infections.csv", header=None)
data = data.iloc[1:, 1:]
flattened = data.values.flatten()
filtered = [int(x) for x in flattened if pd.notna(x) and x < 10]

# Histogram data
counts = Counter(filtered)
total = sum(counts.values())
x_vals = np.arange(10)
y_vals = np.array([counts.get(i, 0) / total for i in x_vals])

# Fit negative binomial
mean = np.mean(filtered)
var = np.var(filtered)
p = mean / var if var > mean else 0.5
r = mean * p / (1 - p) if (1 - p) != 0 else 1
nbinom_probs = nbinom.pmf(x_vals, r, p)

# Plot
plt.figure(figsize=(6, 4))
bar_width = 0.6  # <== narrower bars with gap
plt.bar(x_vals, y_vals, width=bar_width, color='black', label='Data', align='center', edgecolor='white')
plt.plot(x_vals, nbinom_probs, 'r-', marker='d', label='Fit', linewidth=1.5)

# Axes & ticks
plt.xlabel("Number of secondary infections")
plt.ylabel("Probability")
plt.xticks(x_vals)  # ticks at integer values, centered
plt.xlim(-0.5, 9.5)
plt.legend()
plt.tight_layout()
plt.savefig("./Epiabm_plots/secondary_infections.png")
