import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import integrate, optimize
import pints

# Step one: Define the model
# True parameters
E_time = 5.5
IFR = 0.0066
I_time = 7
ksi = 1 / I_time
kappa = 1 / E_time
gamma = ksi * (1 - IFR)
mu = ksi * IFR
beta = 2.4 * (gamma + mu)
# Initial conditions and time series
y0 = [1 - 1e-6, 0, 1e-6, 0, 0]  # S, E, I, R, D
times = np.linspace(0, 365, 365)
true_params = [beta, kappa, gamma, mu]
compartments = ['S', 'E', 'I', 'R', 'D']

# SEIRD ODE system
def seird(y, t, beta, kappa, gamma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - kappa * E
    dIdt = kappa * E - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# odeint
def simulate(func, parameters, y0, times):
    sol = scipy.integrate.odeint(func, y0, times, args=tuple(parameters))
    return sol

# True trajectories
actual_values = simulate(seird, true_params, y0, times)

# Step two: Bayesian inference
# Add noise to all compartments
sigma = 0.02
noisy_data = actual_values + np.random.normal(0, sigma, actual_values.shape)
# Plot noisy data vs true
plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(times, noisy_data[:, i], '.', label=f'{compartments[i]} noisy')
    plt.plot(times, actual_values[:, i], label=f'{compartments[i]} true')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('Noisy vs True Data')
plt.tight_layout()
plt.savefig('1noisy.png')

# --- MLE estimation ---
def sumofsquares(y_model, y_data):
    return np.sum((y_data - y_model)**2)

def scalar_to_minimise(params):
    y_model = simulate(seird, params, y0, times)
    return sumofsquares(y_model, noisy_data) / y_model.size

# Optimise it with scipy
result = scipy.optimize.minimize(scalar_to_minimise, [0.2, 0.1, 0.05, 0.001],
                                 bounds=[(0.01,1), (0.01,1), (0.001,1), (0.0001,0.01)])
print('True params:', true_params)
print('MLE estimated params:', result.x)

# Plot MLE fit
recon_model = simulate(seird, result.x, y0, times)
plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(times, noisy_data[:, i], '.', label=f'{compartments[i]} noisy')
    plt.plot(times, recon_model[:, i], '--', linewidth=2.5, label=f'{compartments[i]} MLE fit')
    plt.plot(times, actual_values[:, i], label=f'{compartments[i]} true')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('MLE Fit to Noisy Data')
plt.tight_layout()
plt.savefig('2mle.png')

# Step three: Using PINTS
class PintsSEIRD(pints.ForwardModel):
    def n_parameters(self):
        return 4
    def n_outputs(self):
        return 5
    def simulate(self, parameters, times):
        return simulate(seird, parameters, y0, times)

problem = pints.MultiOutputProblem(PintsSEIRD(), times, noisy_data)
error_measure = pints.SumOfSquaresError(problem)
optimisation = pints.OptimisationController(error_measure, [0.3, 0.15, 0.05, 0.001], method=pints.XNES)
optimisation.set_log_to_screen(False)
parameters, error = optimisation.run()

print('Pints MLE params:', parameters)
plt.figure(figsize=(10,6))
plt.plot(times, noisy_data, '.', label='Measured values')
plt.plot(times, recon_model, label='Custom inferred values')
plt.plot(times, PintsSEIRD().simulate(parameters, times), '--', lw=2, label='Pints inferred values')
plt.legend()
plt.savefig('3pints.png')

# Step four: Using PINTS for MCMC sampling
log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma=sigma)
startpoints = [
    [0.2, 0.1, 0.1, 0.0005],  # Chain 1 start
    [0.4, 0.3, 0.2, 0.001], # Chain 2 start
    [0.6, 0.5, 0.3, 0.0015] # Chain 3 start
]
mcmc = pints.MCMCController(log_likelihood, 3, startpoints, method=pints.HaarioBardenetACMC)
mcmc.set_max_iterations(3000)
mcmc.set_log_to_screen(False)
samples = mcmc.run()

import pints.plot
# pints.plot.trace(samples)
# plt.savefig('4mcmc_params.png')

# burnin = int(samples.shape[1] * 0.5)  # 50% burn-in
samples_post = samples[:, 1500:]

# Plot trace of post-burn-in samples
pints.plot.trace(samples_post)
plt.suptitle("Trace Plot (Post Burn-in)")
plt.tight_layout()
plt.savefig('5params_burn_in.png')

# Combine chains and plot series prediction
samples_combined = np.vstack(samples_post)
pints.plot.series(samples_combined, problem)
plt.suptitle("Posterior Predictive Series (Post Burn-in)")
plt.tight_layout()
plt.savefig('6seird_burn_in.png')

# Compute R̂ values
rhat_values = pints.rhat(samples_post)

# Print nicely
print("Gelman-Rubin R̂ statistics (via PINTS):")
for i, r in enumerate(rhat_values):
    print(f"R̂ for parameter {i+1} = {r:.4f}")