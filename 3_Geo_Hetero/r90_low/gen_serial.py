import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Generation time and serial interval
gen_raw = pd.read_csv("./simulation_outputs/generation_times.csv", skiprows=1, header=None)
ser_raw = pd.read_csv("./simulation_outputs/serial_intervals.csv", skiprows=1, header=None)

# Flatten to 1D arrays, drop NaNs, and convert to integers
gen_values = pd.to_numeric(gen_raw.values.flatten(), errors='coerce')
ser_values = pd.to_numeric(ser_raw.values.flatten(), errors='coerce')

gen_values = gen_values[~np.isnan(gen_values)].astype(int)
ser_values = ser_values[~np.isnan(ser_values)].astype(int)

bins = np.arange(0, 31)

plt.figure(figsize=(8, 5))
plt.hist(gen_values, bins=bins, density=True, alpha=0.8, label="Generation times",
         color="orchid", edgecolor="black")
plt.hist(ser_values, bins=bins, density=True, alpha=0.8, label="Serial intervals",
         color="mediumaquamarine", edgecolor="black")
plt.xlabel("Time [days]")
plt.ylabel("Probability")
plt.legend()
plt.xlim([0, 30])
plt.xticks(bins[:-1] + 0.5, bins[:-1])  # Center tick labels
plt.tight_layout()
plt.savefig("./Epiabm_plots/generation_serial.png")
