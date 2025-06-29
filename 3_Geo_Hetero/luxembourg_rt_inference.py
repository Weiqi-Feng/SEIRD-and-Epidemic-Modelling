import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pints
from scipy.interpolate import interp1d

mpl.rcParams.update({'font.size': 15})

radius = "homo"  # Heterogeneity
pop_size = 662854  # Total population of Luxembourg

# 1 Epiabm output visualisation
# 1.1 Use secondary_infections.csv to calculate real Rt_case
secondary_infections_data = pd.read_csv(f"NI_outputs/{radius}/secondary_infections.csv", dtype="float32", low_memory=False)
plt.figure(figsize=(10, 6))
plt.plot(secondary_infections_data["time"], secondary_infections_data["R_t"])
plt.xlabel("Time")
plt.ylabel("$R_t$")
plt.title(f"$R_t$ values over time for Luxembourg simulation, {radius}")
plt.savefig(f"NI_outputs/{radius}/simulation_flow_R_t.png")
plt.close()

# 1.2 Secondary infections histogram
fig, ax = plt.subplots()
secondary_infections_data.dropna()
secondary_infections_only = secondary_infections_data.iloc[:, 2:-1].to_numpy()
secondary_infections_array = secondary_infections_only.flatten()
x_arr = np.arange(0,40, 1)
ax.hist(secondary_infections_array, range=(0, 40), bins=x_arr, log=False)
ax.set_xlabel("Number of secondary infections")
ax.set_ylabel("Frequency")
ax.set_title(f"Histogram of secondary infections, {radius}")
fig.savefig(f"NI_outputs/{radius}/simulation_flow_secondary_infections.png", bbox_inches="tight")
plt.close()

# # 1.3 Serial intervals and generation times
serial_interval_df = pd.read_csv(f"NI_outputs/{radius}/serial_intervals.csv",
                                 index_col=0)
serial_interval_df.dropna()
serial_interval_array = serial_interval_df.to_numpy().flatten()
generation_time_df = pd.read_csv(f"NI_outputs/{radius}/generation_times.csv",
                                 index_col=0)
generation_time_df.dropna()
generation_time_array = generation_time_df.to_numpy().flatten()

fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
ax.hist(generation_time_array, density=True, range=(0, 30), log=False, color="darkviolet", label="Generation times", 
        alpha=0.5, bins=np.arange(0, 30, 1), rwidth=0.8)
ax.hist(serial_interval_array, density=True, range=(0, 30), log=False, color="seagreen", label="Serial intervals", 
        alpha=0.5, bins=np.arange(0, 30, 1), rwidth=0.8)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Probability")
ax.legend()
fig.savefig(f"NI_outputs/{radius}/simulation_flow_si_and_gt.png", bbox_inches="tight")
plt.close()

# 1.4 Crude estimates of β, κ and γ
small_seir_df = pd.read_csv(f"NI_outputs/{radius}/seir.csv", index_col=0)
susceptible = small_seir_df["Susceptible"]
exposed = small_seir_df["Exposed"]
infected = small_seir_df["Infected"]
recovered = small_seir_df["Recovered"]
times = np.arange(0, len(susceptible), 1)

# SEIR plot
fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
ax.plot(times, susceptible, color="blue", label="Susceptible", lw=2.5)
ax.plot(times, exposed, color="forestgreen", label="Exposed", lw=2.5)
ax.plot(times, infected, color="goldenrod", label="Infected", lw=2.5)
ax.plot(times, recovered, color="firebrick", label="Recovered", lw=2.5)
ax.legend()
ax.set_xlabel("Time [days]")
ax.set_ylabel("Number of people")
ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6), useMathText=True)
# plt.title("Compartmental data for Northern Ireland simulation with r = 0.2")
fig.savefig(f"NI_outputs/{radius}/SEIR.png", bbox_inches="tight")
plt.close()

mpl.rcParams.update({'font.size': 11})
dS_dt = np.gradient(susceptible)
dE_dt = np.gradient(exposed)
dI_dt = np.gradient(infected)
dR_dt = np.gradient(recovered)

beta_estimate = - dS_dt * pop_size / (susceptible * infected)
gamma_estimate = dR_dt / infected
kappa_estimate_eq_3 = (dI_dt + gamma_estimate * infected) / exposed
kappa_estimate_eq_2 = (beta_estimate / pop_size * susceptible * infected - dE_dt) / exposed

epiabm_gamma, epiabm_kappa = 1/9.41, 1/4.59

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 7.5), sharex=True)
axs[0].plot(times, beta_estimate, color="goldenrod")
axs[0].set_ylabel("Infection rate, $\\beta$")
axs[1].plot(times, kappa_estimate_eq_2, color="blue")
axs[1].axhline(epiabm_kappa, color="blue", alpha=0.7, linestyle="--", label="Epiabm value")
axs[1].set_ylabel("Incubation rate, $\\kappa$")
axs[1].legend()
axs[2].plot(times, gamma_estimate, color="green")
axs[2].axhline(epiabm_gamma, color="green", alpha=0.7, linestyle="--", label="Epiabm value")
axs[2].set_ylabel("Recovery rate, $\\gamma$")
axs[2].set_xlabel("Time [days]")
axs[2].legend()
axs[0].set_ylim(0, 3.5)
fig.subplots_adjust(hspace=0.1)
fig.savefig(f"NI_outputs/{radius}/crude_estimates_2.png", bbox_inches="tight")
plt.close()

# 2 Model Inference
# 2.1 Set up
all_data = np.array([susceptible, exposed, infected, recovered]).transpose()
initial_infected = all_data[0, 2]
I0 = all_data[0, 2]
all_data = all_data / pop_size # normalise to population size

import seirmo

class SEIRModel(pints.ForwardModel):
    def __init__(self, pop_size, initial_infected, fixed_hyperparams=None, prevalence=False,
                 S0=None, E0=None, R0=None, sampling_intervals=None):
        """
        Creates the SEIR model, with options to fix parameters, use prevalence only, and sample various time intervals
        of the epidemic.
        """

        super(SEIRModel, self).__init__()

        seir_model = seirmo.SEIRModel()
        self._model = seirmo.ReducedModel(seir_model)
        fixed_parameters = {"S0": 1 - initial_infected / pop_size, "E0": 0, "I0": initial_infected / pop_size, "R0": 0}
        if S0:
            fixed_parameters = {"S0": S0 / pop_size, "E0": E0 / pop_size, "I0": I0 / pop_size, "R0": R0 / pop_size}
        self._fixed_hyperparams = fixed_hyperparams
        if fixed_hyperparams is not None:
            fixed_parameters = fixed_hyperparams | fixed_parameters
        self._model.fix_parameters(fixed_parameters)
        self._n_outputs = 1 if prevalence else 4
        # sampling_intervals is a list of tuple pairs denoting the start and end timepoints
        # for which to take a sample (to mimic the REACT-1 survey), for example, missing
        # every week
        self._sampling_intervals = sampling_intervals

    def n_outputs(self):
        # Returns number of model outputs.
        # Returns the S, E, I and R values at each timestep
        return self._n_outputs

    def n_parameters(self):
        # Returns number of parameters, i.e. some of beta, kappa and gamma
        return 3 - len(self._fixed_hyperparams)

    def n_fixed_parameters(self):
        # Returns number of fixed parameters, i.e. 4 initial conditions (S(0), E(0), I(0) and R(0)) and potentially other ones
        return self._n_outputs + len(self._fixed_hyperparams)

    def simulate(self, parameters, sampled_times):
        # This ensures that we are taking the prevalence
        if self._n_outputs == 4:
            self._model.set_outputs(["S", "E", "I", "R"])
        else:
            self._model.set_outputs(["I"])
        # We want to ensure that we are simulating for all times and then afterwards take the sample
        times = np.arange(sampled_times[0], sampled_times[-1] + 1, 1)
        compartmental_results = self._model.simulate(parameters=parameters, times=times)
        if self._sampling_intervals:
            compartmental_lists = [compartmental_results[pair[0]:pair[1]] for pair in self._sampling_intervals]
            compartmental_results = np.array([result for result_list in compartmental_lists for result in result_list])
        return compartmental_results

trial = "bgk_AR1_prev_10"  # name of trial
prevalence = True  # only Infected
sampling_intervals = None

fixed_hyperparams = {}
all_data = all_data if not prevalence else all_data[:, 2]
pints_model = SEIRModel(pop_size=pop_size, initial_infected=initial_infected,
                        fixed_hyperparams=fixed_hyperparams, prevalence=prevalence,
                        sampling_intervals=sampling_intervals)
if sampling_intervals:
    time_lists = [times[pair[0]:pair[1]] for pair in sampling_intervals]
    truncated_times = np.array([time for time_list in time_lists for time in time_list])
    data_lists = [all_data[pair[0]:pair[1]] for pair in sampling_intervals]
    truncated_data = np.array([data for data_list in data_lists for data in data_list])
else:
    truncated_times = times[:]
    truncated_data = all_data[:]
problem = pints.MultiOutputProblem(pints_model, truncated_times, truncated_data)

# 2.2 Optimisation
boundaries = pints.RectangularBoundaries([0.01, 0.01, 0.01],
                                         [10, 1, 1])
log_prior = pints.UniformLogPrior(boundaries)

composed_log_prior = pints.ComposedLogPrior(
    log_prior, # beta, kappa and gamma
    # pints.UniformLogPrior(0, 1), # rho_S
    # pints.TruncatedGaussianLogPrior(0, 1, 0, np.inf), # sigma_S
    # pints.UniformLogPrior(0, 1), # rho_E
    # pints.TruncatedGaussianLogPrior(0, 1, 0, np.inf), # sigma_E
    pints.UniformLogPrior(0, 1), # rho_I
    pints.TruncatedGaussianLogPrior(0, 1, 0, np.inf), # sigma_I
    # pints.UniformLogPrior(0, 1), # rho_R
    # pints.TruncatedGaussianLogPrior(0, 1, 0, np.inf) # sigma_R
)

composed_boundaries = pints.RectangularBoundaries([0.01, 0.01, 0.01, 0, 0],
                                                  [10, 1, 1, 4, 4])

num_opts = 10
# Create log-likelihood
log_likelihood = pints.AR1LogLikelihood(problem)
param_names = ["beta", "kappa", "gamma",
               "rho_I", "sigma_I"]

for j in range(num_opts):
    print(f"Iteration {j}")
    xs = composed_log_prior.sample(1)
    opt = pints.OptimisationController(log_likelihood, xs, boundaries=composed_boundaries, method=pints.CMAES)
    opt.set_max_unchanged_iterations(200, 0.00000001)
    opt.set_max_iterations(2500)
    opt_params, opt_values = opt.run()
    opt_param_df = pd.DataFrame({param_names[i]: [opt_params[i]] for i in range(len(param_names))})
    opt_param_df.to_csv(f"optimisation_outputs/{radius}/{trial}/opt_params_{j}.csv")

# 2.3 Plot optimisation results
opt_params = []
for j in range(num_opts):
    opt_param_df = pd.read_csv(f"optimisation_outputs/{radius}/{trial}/opt_params_{j}.csv", index_col=0)
    opt_params.append(opt_param_df.to_numpy()[0])

errors = []
for j in range(num_opts):
    print("---------------------------------------------------------------------------")
    error = log_likelihood(opt_params[j])
    print(f"Optimisation {j}, log-likelihood = {round(error, 5)}")
    print("---------------------------------------------------------------------------")
    errors.append(error)
    output_list = []
    for i in range(pints_model.n_parameters()):
        opt_param = round(opt_params[j][i], 5)
        output_list.append(opt_param)
        print(f"Parameter: {param_names[i]}, Optimised: {opt_param}")

chosen_opt = 2

num_params = pints_model.n_parameters()
opt_soln = pints_model.simulate(opt_params[chosen_opt][:num_params], truncated_times)

opt_I = opt_soln
plt.plot(truncated_times, opt_I, "-r", label="Optimised")
plt.plot(truncated_times, infected / pop_size, "-b", alpha=0.5, label="True data")
plt.xlabel("Time [days]")
plt.ylabel("Population")
plt.legend()
plt.title(f"Susceptible population for optimisation {chosen_opt}")
plt.savefig(f"optimisation_outputs/{radius}/{trial}/susceptible_{chosen_opt}.png")

# 2.4 PINTS MultiOutputProblem and GaussianLogLikelihood
# Create log-prior and log-posterior
log_posterior = pints.LogPosterior(log_likelihood, composed_log_prior)

num_chains = 4
xs = np.vstack([opt_params[chosen_opt] * 1.0,
                opt_params[chosen_opt] * 0.7,
                opt_params[chosen_opt] * 0.3,
                opt_params[chosen_opt] * 0.2])
transform = pints.RectangularBoundariesTransformation([0.01, 0, 0, 0, 0],
                                                      [10, 1, 1, 1, 1])
n_params = log_prior.n_parameters()
mcmc = pints.MCMCController(log_posterior, num_chains, xs, transformation=transform)
max_iterations = 16000
mcmc.set_max_iterations(max_iterations)
mcmc.set_initial_phase_iterations(1000)
# This is to try and get more of a spread in the initial phase
sigma0_multiplier = None
if sigma0_multiplier is not None:
    for s in mcmc.samplers():
        s._sigma = np.diag(s._sigma.dot(sigma0_multiplier))

chains = mcmc.run()

param_names = ['Infection Rate (beta)', 'Incubation Rate (kappa)', 'Recovery Rate (gamma)',
               'rho_I', 'sigma_I']

for i, chain in enumerate(chains):
    df = pd.DataFrame(chain, columns=param_names)
    df.to_csv(f"chain_data/{radius}/{trial}/chain_{i}.csv")

# Plot the chains
list_of_chains = []
for i in range(4):
    df = pd.read_csv(f"chain_data/{radius}/{trial}/chain_{i}.csv", index_col=0)
    chain = df.to_numpy()
    list_of_chains.append(chain)
chains = np.array(list_of_chains)

mpl.rcParams.update({'font.size': 11})
import pints.plot
pints.plot.trace(chains, parameter_names=param_names)
plt.savefig(f"chain_data/{radius}/{trial}/trace.png", dpi=300)

# Disgnosing using R_stat
r_hat_values = pints.rhat(chains, warm_up=0.5)
for i in range(len(param_names)):
    print(f"R_hat value for {param_names[i]}: {round(r_hat_values[i], 5)}")

chains = chains[:, int(max_iterations / 2):] # Discard half of the iterations from now on

# 2.5 Predicted time series plots
# Take chain 1 as an example
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(times, infected, label='True data', color='black')
ax.plot(times, pints_model.simulate(chains[0, 0][:3], times) * pop_size, label='Guess', color='red')
ax.legend()
ax.set_xlabel('Time [days]')
ax.set_ylabel('Infected')
fig.savefig(f"inference_outputs/{radius}/{trial}/example_estimated.png", bbox_inches='tight')
plt.close()

pints.plot.series(chains[0, :], problem)
plt.savefig(f"inference_outputs/{radius}/{trial}/example_series.png")

# 2.6 Estimating Rt
param_names = ["Infection Rate ($\\beta$)", "Incubation Rate ($\\kappa$)", "Recovery Rate ($\\gamma$)", "rho_I", "sigma_I"]

chains_df = pd.DataFrame(columns=param_names + ['chain'])
for i in range(4):
    chain_i_df = pd.DataFrame(chains[i], columns=param_names)
    # for population_parameter in ["sigma_S", "sigma_E", "sigma_I", "sigma_R"]:
    for population_parameter in ["sigma_I"]:
        chain_i_df[population_parameter] = pop_size * chain_i_df[population_parameter]
    chain_i_df['chain'] = [i] * len(chain_i_df)
    chains_df = pd.concat([chains_df, chain_i_df], ignore_index=True)

epiabm_params = [1/9.41, 1/4.59]

beta_posterior = chains_df["Infection Rate ($\\beta$)"].to_numpy()
gamma_posterior = chains_df["Recovery Rate ($\\gamma$)"].to_numpy()
kappa_posterior = chains_df["Incubation Rate ($\\kappa$)"].to_numpy()
# print(f"Mean beta: {np.mean(chains_df["Infection Rate ($\\beta$)"])}," 
#       f" std dev beta: {np.std(chains_df["Infection Rate ($\\beta$)"])}")
# print(f"Mean gamma: {np.mean(chains_df["Recovery Rate ($\\gamma$)"])}," 
#       f" std dev gamma: {np.std(chains_df["Recovery Rate ($\\gamma$)"])}")
# print(f"Mean kappa: {np.mean(chains_df["Incubation Rate ($\\kappa$)"])}," 
#       f" std dev kappa: {np.std(chains_df["Incubation Rate ($\\kappa$)"])}")
num_rows, num_columns = 2, 2
num_params = 3

histograms = chains_df.hist(column=["Infection Rate ($\\beta$)", "Recovery Rate ($\\gamma$)", "Incubation Rate ($\\kappa$)"],
                            color="blue", alpha=0.7, figsize=(8, 7))
for row in range(num_rows):
    figs_in_row = num_columns
    if row == num_rows - 1:
        r = num_params % num_columns
        figs_in_row = r if r != 0 else num_params
    for column in range(figs_in_row):
        index = row * num_columns + column
        ax = histograms[row][column]
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        # crude_params = param_df[crude_param_names[index]]
        # ax.hist(crude_params, alpha=0.4, label="Crude")
        if index >= 1:
            ax.axvline(epiabm_params[index - 1], color='r', linestyle='dashed', linewidth=1, label="Epiabm value")
            ax.legend()
# ax.legend()
# plt.title("Posterior distributions for the governing parameters")
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(f"inference_outputs/{radius}/{trial}/posteriors.png", bbox_inches='tight')

# 2.7 Contingency table for secondary infections
max_secondary_infections = 18
final_active_day = 90
times = np.array(secondary_infections_data["time"], dtype="int8")
secondary_infections = secondary_infections_data.iloc[:, 1:-1].to_numpy()
contingency_list = []
for t in times[:final_active_day + 1]:
    print(f"Day {t}", end=" ")
    t_list = [0] * (max_secondary_infections + 1)
    secondary_infections_t = secondary_infections[t, :]
    secondary_infections_t = secondary_infections_t[~np.isnan(secondary_infections_t)]
    for entry in secondary_infections_t:
        if entry <= max_secondary_infections:
            t_list[int(entry)] += 1
    contingency_list.append(t_list)
contingency_array = np.array(contingency_list).transpose()
contingency_df = pd.DataFrame(contingency_array, columns=times[:final_active_day + 1])
contingency_df.to_csv(f"NI_outputs/{radius}/contingency.csv")

contingency_array = pd.read_csv(f"NI_outputs/{radius}/contingency.csv", index_col=0).to_numpy()
final_active_day = 80

true_Rt = secondary_infections_data["R_t"].to_numpy()
nan_days = np.isnan(true_Rt)
x = lambda z: z.nonzero()[0]
true_Rt[nan_days] = np.interp(x(nan_days), x(~nan_days), true_Rt[~nan_days]) # linearly interpolate

active_times = np.arange(-0.5, final_active_day, 1)
frequencies = np.arange(-0.5, max_secondary_infections, 1)
fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
cmap_name = "GnBu"
whole_map = mpl.colormaps[cmap_name]
truncated_map = mpl.colors.LinearSegmentedColormap.from_list(f"{cmap_name}_subset",
                                                             whole_map(np.linspace(0.3, 1.0, 100)))
cf = ax.pcolormesh(active_times, frequencies, contingency_array[:, :final_active_day + 1],
                   alpha=1, cmap=truncated_map,
                   norm=mpl.colors.LogNorm())
cf.set_clim(1, 1e5)
ax.plot(secondary_infections_data["time"][:final_active_day + 1], 
        true_Rt[:final_active_day + 1], 'k', 
        lw=2.5, label="True $R_t^{\\mathrm{case}}$")
fig.colorbar(cf, label="Primary infectors")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Secondary infections")
# ax.set_title("Distribution of secondary infections over time")
ax.legend()
fig.savefig(f"NI_outputs/{radius}/secondary_infections_E_mesh.png", bbox_inches="tight")

# 2.8 Compartmental plots
import random
num_samples = 1000
pints_model._n_outputs = 4
SEIR_lists = [[], [], [], []]
beta_samples, kappa_samples, gamma_samples = [], [], []
for j in range(num_samples):
    k = random.randint(0, len(beta_posterior))
    beta, kappa, gamma = beta_posterior[k], kappa_posterior[k], gamma_posterior[k]
    # Note that below, the name "beta" refers to the parameter "kappa" AHHHHH
    # beta, kappa, gamma = beta_posterior[k], fixed_hyperparams["beta"], fixed_hyperparams["gamma"]
    beta_samples.append(beta)
    kappa_samples.append(kappa)
    gamma_samples.append(gamma)
    # seir_data = pints_model.simulate([beta], truncated_times)
    seir_data = pints_model.simulate([beta, kappa, gamma], truncated_times)
    for i in range(len(SEIR_lists)):
        SEIR_lists[i].append(seir_data[:, i])
lower_bounds = []
means = []
upper_bounds = []
for SEIR_list in SEIR_lists:
    SEIR_np = np.array(SEIR_list)
    lower_bounds.append(np.percentile(SEIR_np, 2.5, axis=0) * pop_size)
    means.append(np.mean(SEIR_np, axis=0) * pop_size)
    upper_bounds.append(np.percentile(SEIR_np, 97.5, axis=0) * pop_size)

# Make the plots
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 30/3), sharex=True)
lower_t, upper_t = 0, 91
true_data = [susceptible[lower_t:upper_t], exposed[lower_t:upper_t], infected[lower_t:upper_t], recovered[lower_t:upper_t]]
ylabels = ["Susceptible", "Exposed", "Infected", "Recovered"]
for i in range(4):
    ax = axs[i]
    ax.plot(times, true_data[i], "-k", label="True data")
    ax.plot(truncated_times, means[i], "--r", label="Posterior mean")
    ax.fill_between(truncated_times, lower_bounds[i], upper_bounds[i], alpha=0.5,
                    label="95% credible interval", color="blue")
    if i == 0:
        ax.legend(loc='upper right')
    elif i == 3:
        ax.set_xlabel("Time [days]")
    ax.set_ylabel(ylabels[i])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6), useMathText=True)
    # ax.set_ylim(0, 2e6)
# fig.suptitle("Inferred SEIR curves: Case 1")
fig.savefig(f"inference_outputs/{radius}/{trial}/inferred_SEIR_CI.png")

# 2.9 Credible interval plot
R_t_list = []
num_samples = 1000
for j in range(num_samples):
    # R_t_list.append(beta_samples[j] / (gamma_samples[j] * pop_size) * susceptible[:])
    R_t_list.append(beta_samples[j] / gamma_samples[j] * SEIR_lists[0][j])
# mean_R_t = np.mean(np.array(R_t_list), axis=0)
# std_dev_R_t = np.std(np.array(R_t_list), axis=0)
# upper_R_t = np.percentile(np.array(R_t_list), 97.5, axis=0)
# median_R_t = np.percentile(np.array(R_t_list), 50, axis=0)
# lower_R_t = np.percentile(np.array(R_t_list), 2.5, axis=0)

# 2.10 Convert Rt_inst to Rt_case
fig, ax = plt.subplots()
serial_interval_samples = []
alternative_samples = []
for j in range(num_samples):
    kap, gam = kappa_samples[j], gamma_samples[j]
    serial_interval_samples.append(kap * gam / (kap - gam) * (np.exp(-gam * times) - np.exp(-kap * times)))
    alternative_samples.append(gam * np.exp(-gam * times))

import scipy.integrate as si
def Rt_inst_to_Rt_case(Rt_inst, f, t_start, t_end):
    """Converts the instantaneous reproduction number to the case reproduction number
    at time t, given a generation time/serial interval distribution, f.
    """
    Rt_case = []
    dx = 1
    for t in range(t_end - t_start):
        Rt_case_t = si.simpson(Rt_inst[t:] * (f[:t_end-t_start-t]), x=np.arange(t + t_start, t_end, 1.0))
        Rt_case.append(Rt_case_t)
    return Rt_case

case_Rt_list = []
for j in range(num_samples):
    case_Rt_sample = []
    if sampling_intervals:
        inst_Rt = [0] * (upper_t - lower_t)
        # Assigning the different Rt values to their correct timepoints
        i_start = 0
        for pair in sampling_intervals:
            inst_Rt[pair[0]:pair[1]] = R_t_list[j][i_start:i_start + pair[1] - pair[0]]
            i_start += pair[1] - pair[0]
        # Filling the gaps using linear interpolation
        for gap_num in range(len(sampling_intervals) - 1):
            left, right = sampling_intervals[gap_num][1], sampling_intervals[gap_num + 1][0]
            gap = right - left
            linear_spline = [inst_Rt[left - 1] + (inst_Rt[right] - inst_Rt[left - 1]) * (i + 1) / (gap + 1)
                             for i in range(gap)]
            inst_Rt[left:right] = linear_spline
        inst_Rt = np.array(inst_Rt)
    else:
        inst_Rt = R_t_list[j]
    case_Rt_sample = Rt_inst_to_Rt_case(inst_Rt, serial_interval_samples[j], lower_t, upper_t)
    case_Rt_list.append(case_Rt_sample)
mean_R_t = np.mean(np.array(case_Rt_list), axis=0)
std_dev_R_t = np.std(np.array(case_Rt_list), axis=0)
upper_R_t = np.percentile(np.array(case_Rt_list), 97.5, axis=0)
median_R_t = np.percentile(np.array(case_Rt_list), 50, axis=0)
lower_R_t = np.percentile(np.array(case_Rt_list), 2.5, axis=0)
# plt.plot(truncated_times[0:], secondary_infections_data["R_t"][20:61], 'k', 
#          label="True $R_t$")
plt.figure(figsize=(8, 2.5))
plt.plot(times, true_Rt[:], 'k', 
         label="True $R_t$")
plt.plot(times, mean_R_t[:], '--r', label="Posterior mean")
plt.fill_between(times, lower_R_t[:], upper_R_t[:], alpha=0.5,
                 label="95% credible interval", color="blue")
plt.xlabel("Time [days]")
plt.ylabel("$R_t$")
# plt.title("Inferred $R_t$ for Northern Ireland simulation: Case 2")
# plt.ylim(0, 16)
plt.ylim(0, 20)
plt.legend(loc='upper center')
plt.savefig(f"inference_outputs/{radius}/{trial}/inferred_case_R_t_CI_sampled.png", bbox_inches="tight")