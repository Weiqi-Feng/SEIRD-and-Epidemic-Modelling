import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# SEIRDC model
class SEIRDModel:
    def __init__(self, beta, kappa, gamma, mu):
        self.beta = beta
        self.kappa = kappa
        self.gamma = gamma
        self.mu = mu

    def derivatives(self, y, t):
        S, E, I, R, D= y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.kappa * E
        dIdt = self.kappa * E - (self.gamma + self.mu) * I
        dRdt = self.gamma * I
        dDdt = self.mu * I
        dCdt = self.beta * S * I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]

    def simulate(self, initial_conditions, days):
        t = np.linspace(0, days, days + 1)
        result = odeint(self.derivatives, initial_conditions, t)
        return t, result

# Parameters
E_time = 5.5 # average latent period
IFR = 0.0066 # Infection fatality rate: IFR = mu / (gamma + mu)
I_time = 7 # average infectious period
ksi = 1 / I_time # ksi = gamma + mu = 1 / I_time
# Model parameters
kappa = 1 / E_time
gamma = ksi * (1 - IFR) # gamma = 𝜁(1-IFR) = 0.0133
mu = ksi * IFR # mu = 𝜁IFR

# Initial conditions
initial_conditions = [1 - 1e-6, 0, 1e-6, 0, 0]
days = 365
t = np.linspace(0, days, days + 1)

# Range of beta values
beta_vals = np.linspace(0.1, 0.7, 15)
norm = mcolors.Normalize(vmin=min(beta_vals) / (gamma + mu), vmax=max(beta_vals) / (gamma + mu))
cmap = cm.get_cmap("viridis")

# Plot
plt.figure(figsize=(10, 6))

for beta in beta_vals:
    R0 = beta / (gamma + mu)
    color = cmap(norm(R0))
    model = SEIRDModel(beta, kappa, gamma, mu)
    t, result = model.simulate(initial_conditions, days)
    I = result[:, 2]
    linestyle = '--' if R0 > 1 else '-'
    plt.plot(t, I, linestyle=linestyle, color=color, label=f"R₀={R0:.2f}")

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, label='R₀')

# Add custom legend for line styles
custom_lines = [
    Line2D([0], [0], color='gray', linestyle='-', label='R₀ ≤ 1'),
    Line2D([0], [0], color='gray', linestyle='--', label='R₀ > 1')
]
plt.legend(handles=custom_lines, title="Line Style", loc='upper right')

# Labels and formatting
plt.xlabel("Time (Days)")
plt.ylabel("Infectious proportion")
plt.title("Sensitivity Analysis of Infectious Proportion vs $R_0$")
plt.grid(True)
plt.tight_layout()
plt.savefig("Sensitivity_beta.png")