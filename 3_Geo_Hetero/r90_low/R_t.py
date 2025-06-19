import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

# Load data
data = pd.read_csv("./simulation_outputs/secondary_infections.csv", header=None)
data = data.iloc[1:, 1:]  # Remove metadata row/col
data = data.apply(pd.to_numeric, errors='coerce')
data_np = data.to_numpy()

days, people = data_np.shape
Rt_case = np.full(days, np.nan)

# Calculate true Rt_case (use ALL values, including ≥15)
for day in range(days):
    exposed_today = data_np[day, :]
    valid = ~np.isnan(exposed_today)
    if np.any(valid):
        Rt_case[day] = np.mean(exposed_today[valid])  # no filtering!

# Interpolate missing Rt_case values
x = np.arange(days)
valid = ~np.isnan(Rt_case)
interp_func = interp1d(x[valid], Rt_case[valid], kind='linear', fill_value="extrapolate")
Rt_case = interp_func(x)

# Create heatmap matrix for secondary infections (0–17 only)
max_secondary = 18
heatmap = np.zeros((max_secondary, days))  # y=0..17

for day in range(days):
    sec_vals = data_np[day, :]
    valid = ~np.isnan(sec_vals)
    sec_clean = sec_vals[valid]
    sec_clean = sec_clean[sec_clean < max_secondary]  # discard ≥15 only for heatmap
    for val in sec_clean.astype(int):
        heatmap[val, day] += 1

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

img = ax.imshow(
    heatmap[::-1],  # flip Y-axis: 0 at bottom
    aspect='auto',
    cmap='BuGn',
    norm=LogNorm(vmin=1, vmax=np.max(heatmap)),
    extent=[0, days, 0, max_secondary]
)

# Plot Rt line
ax.plot(x, Rt_case, color='black', linewidth=2, label=r'True $R_t^{\mathrm{case}}$')

# Labels and colorbar
ax.set_xlabel("Time [days]")
ax.set_ylabel("Secondary infections")
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Primary infectors")
ax.legend()
plt.tight_layout()
plt.savefig("./Epiabm_plots/true_Rt_case.png")
