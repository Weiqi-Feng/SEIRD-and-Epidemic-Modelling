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

# Results from other research
R0 = 2.4  # Initial E produced by an infectious person: R0 = beta / (gamma + mu)
E_time = 5.5  # average latent period
IFR = 0.0066  # Infection fatality rate: IFR = mu / (gamma + mu)
I_time = 7  # average infectious period
ksi = 1 / I_time  # ksi = gamma + mu = 1 / I_time
# Model parameters
kappa = 1 / E_time
gamma = ksi * (1 - IFR)  # gamma = 𝜁(1-IFR) = 0.0133
mu = ksi * IFR  # mu = 𝜁IFR
beta = R0 * (gamma + mu)  # infection rate beta = R0 * (gamma + mu)

# Initial conditions: [S, E, I, R, D, C]
initial_conditions = [1 - 1e-6, 0, 1e-6, 0, 0, 0]

# Time span: 365 days (1 year)
days = 365

# Run simulation
model = SEIRDModel(beta, kappa, gamma, mu)
t, result = model.simulate(initial_conditions, days)
S, E, I, R, D, C = result.T  # Transpose to unpack columns

# Plot results
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Days')
plt.ylabel('Fraction of Population')
plt.title('SEIRD Model Over One Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('SEIRD.png')

# Compute daily incidence and daily deaths
daily_incidence = np.diff(C, prepend=0)
daily_deaths = np.diff(D, prepend=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, daily_incidence, label='Daily Incidence', color='blue')
plt.plot(t, daily_deaths, label='Daily Deaths', color='red')
plt.xlabel('Days')
plt.ylabel('Daily Count (Fraction of Population)')
plt.title('Daily Incidence and Deaths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Daily_incidence_deaths.png')