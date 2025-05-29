import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SEIRDModel:
    def __init__(self, beta, kappa, gamma, mu):
        self.beta = beta
        self.kappa = kappa
        self.gamma = gamma
        self.mu = mu

    def derivatives(self, y, t):
        S, E, I, R, D, C = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.kappa * E
        dIdt = self.kappa * E - (self.gamma + self.mu) * I
        dRdt = self.gamma * I
        dDdt = self.mu * I
        dCdt = self.beta * S * I
        return [dSdt, dEdt, dIdt, dRdt, dDdt, dCdt]

    def simulate(self, initial_conditions, days):
        t = np.linspace(0, days, days + 1)
        result = odeint(self.derivatives, initial_conditions, t)
        return t, result

# Benchmark parameters
E_time = 5.5
I_time = 7
IFR = 0.0066
kappa_benchmark = 1 / E_time
ksi = 1 / I_time
gamma_benchmark = ksi * (1 - IFR)
mu_benchmark = ksi * IFR
beta_benchmark = 2.4 * (gamma_benchmark + mu_benchmark)

# Initial conditions and duration
initial_conditions = [1 - 1e-6, 0, 1e-6, 0, 0, 0]
days = 365
t = np.linspace(0, days, days + 1)

# Create variations
factors = [0.5, 0.75, 1.0, 1.25, 1.5]

# Setup subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
titles = [
    "Varying β (infection rate)",
    "Varying κ (latent rate)",
    "Varying γ (recovery rate)",
    "Varying μ (mortality rate)"
]

# Plot 1: Varying beta
for f in factors:
    beta = f * beta_benchmark
    model = SEIRDModel(beta, kappa_benchmark, gamma_benchmark, mu_benchmark)
    t, result = model.simulate(initial_conditions, days)
    S, E, I, R, D, _ = result.T
    axes[0].plot(t, I, label=f"β={beta:.3f}")
axes[0].set_title(titles[0])
axes[0].legend()

# Plot 2: Varying kappa
for f in factors:
    kappa = f * kappa_benchmark
    model = SEIRDModel(beta_benchmark, kappa, gamma_benchmark, mu_benchmark)
    t, result = model.simulate(initial_conditions, days)
    S, E, I, R, D, _ = result.T
    axes[1].plot(t, I, label=f"κ={kappa:.3f}")
axes[1].set_title(titles[1])
axes[1].legend()

# Plot 3: Varying gamma
for f in factors:
    gamma = f * gamma_benchmark
    model = SEIRDModel(beta_benchmark, kappa_benchmark, gamma, mu_benchmark)
    t, result = model.simulate(initial_conditions, days)
    S, E, I, R, D, _ = result.T
    axes[2].plot(t, I, label=f"γ={gamma:.3f}")
axes[2].set_title(titles[2])
axes[2].legend()

# Plot 4: Varying mu
for f in factors:
    mu = f * mu_benchmark
    model = SEIRDModel(beta_benchmark, kappa_benchmark, gamma_benchmark, mu)
    t, result = model.simulate(initial_conditions, days)
    S, E, I, R, D, _ = result.T
    axes[3].plot(t, I, label=f"μ={mu:.4f}")
axes[3].set_title(titles[3])
axes[3].legend()

# Formatting
for ax in axes:
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected proportion (I)")
    ax.grid(True)

plt.tight_layout()
plt.savefig("Sensitivity_I.png")