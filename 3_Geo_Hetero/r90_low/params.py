import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('./simulation_outputs/SEIRD_timeseries.csv')
time = df.iloc[:, 0].values
S = df.iloc[:, 1].values
E = df.iloc[:, 2].values
I = df.iloc[:, 3].values
R = df.iloc[:, 4].values
N = S[0] + E[0] + I[0] + R[0]  # Total population (assuming constant)

# Compute derivatives using central difference
dSdt = np.gradient(S, time)
dEdt = np.gradient(E, time)
dIdt = np.gradient(I, time)
dRdt = np.gradient(R, time)

# Compute γ(t)
gamma_t = np.divide(dRdt, I, out=np.zeros_like(I, dtype=float), where=I > 0)

# Compute κ(t)
kappa_t = np.divide(-dSdt-dEdt, E, out=np.zeros_like(E, dtype=float), where=E > 0)

# Compute β(t)
SI = S * I
beta_t = np.divide(-N * dSdt, SI, out=np.zeros_like(SI, dtype=float), where=SI > 0)

# True values from Epiabm
kappa_true = 1 / 4.59
gamma_true = 1 / (14 - 4.59)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# β(t)
axes[0].plot(time, beta_t, color='orange')
axes[0].set_ylabel("Infection rate, β")

# κ(t)
axes[1].plot(time, kappa_t, color='blue')
axes[1].axhline(kappa_true, linestyle='--', color='blue', label='Epiabm value')
axes[1].legend()
axes[1].set_ylabel("Incubation rate, κ")

# γ(t)
axes[2].plot(time, gamma_t, color='green')
axes[2].axhline(gamma_true, linestyle='--', color='green', label='Epiabm value')
axes[2].legend()
axes[2].set_ylabel("Recovery rate, γ")
axes[2].set_xlabel("Time [days]")

plt.tight_layout()
plt.savefig("./Epiabm_plots/params.png")
