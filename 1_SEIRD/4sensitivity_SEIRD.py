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

# Initial conditions
initial_conditions = [1 - 1e-6, 0, 1e-6, 0, 0, 0]
days = 365

factor = 1.5 # Change in parameter value

# Titles and parameter sets
titles = [
    "Changed β",
    "Changed κ",
    "Changed γ",
    "Changed μ"
]

parameter_sets = [
    (beta_benchmark * factor, kappa_benchmark, gamma_benchmark, mu_benchmark),
    (beta_benchmark, kappa_benchmark * factor, gamma_benchmark, mu_benchmark),
    (beta_benchmark, kappa_benchmark, gamma_benchmark * factor, mu_benchmark),
    (beta_benchmark, kappa_benchmark, gamma_benchmark, mu_benchmark * factor * 5)
]

# Colors for compartments
colors = {'S': 'blue', 'E': 'orange', 'I': 'green', 'R': 'red', 'D': 'purple'}

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i in range(4):
    beta, kappa, gamma, mu = parameter_sets[i]
    model = SEIRDModel(beta, kappa, gamma, mu)
    t, result = model.simulate(initial_conditions, days)
    S, E, I, R, D, _ = result.T
    for label, data in zip(['S', 'E', 'I', 'R', 'D'], [S, E, I, R, D]):
        axes[i].plot(t, data, label=label, color=colors[label])
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Time (days)")
    axes[i].set_ylabel("Proportion of population")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig("Sensitivity_SEIRD.png", dpi=300)
