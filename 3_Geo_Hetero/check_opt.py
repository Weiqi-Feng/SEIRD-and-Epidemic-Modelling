import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pints
from scipy.interpolate import interp1d
import seirmo

radius = "r_0_2"  # Heterogeneity
pop_size = 662854  # Total population of Luxembourg

small_seir_df = pd.read_csv(f"NI_outputs/{radius}/seir.csv", index_col=0)
susceptible = small_seir_df["Susceptible"]
exposed = small_seir_df["Exposed"]
infected = small_seir_df["Infected"]
recovered = small_seir_df["Recovered"]
times = np.arange(0, len(susceptible), 1)

all_data = np.array([susceptible, exposed, infected, recovered]).transpose()
# initial_infected = 100
initial_infected = all_data[0, 2]
I0 = 100
# S0, E0, I0, R0 = all_data[5, 0], all_data[5, 1], all_data[5, 2], all_data[5, 3]
all_data = all_data / pop_size

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
        # self._model.set_outputs(["S", "E", "I", "R"])
        # We want to ensure that we are simulating for all times and then afterwards take the sample
        times = np.arange(sampled_times[0], sampled_times[-1] + 1, 1)
        compartmental_results = self._model.simulate(parameters=parameters, times=times)
        if self._sampling_intervals:
            compartmental_lists = [compartmental_results[pair[0]:pair[1]] for pair in self._sampling_intervals]
            compartmental_results = np.array([result for result_list in compartmental_lists for result in result_list])
        return compartmental_results

trial = "bgk_AR1_prev_10"
prevalence = True # Use all four compartments for optimisation
sampling_intervals = None
fixed_hyperparams = {}

all_data = all_data if not prevalence else all_data[:, 2]
pints_model = SEIRModel(pop_size=pop_size, initial_infected=initial_infected,
                        fixed_hyperparams=fixed_hyperparams, prevalence=prevalence,
                        sampling_intervals=sampling_intervals)
# pints_model = SEIRModel(pop_size=pop_size, initial_infected=I0,
#                         fixed_hyperparams=fixed_hyperparams, prevalence=prevalence,
#                         S0=S0, E0=E0, R0=R0)
if sampling_intervals:
    time_lists = [times[pair[0]:pair[1]] for pair in sampling_intervals]
    truncated_times = np.array([time for time_list in time_lists for time in time_list])
    data_lists = [all_data[pair[0]:pair[1]] for pair in sampling_intervals]
    truncated_data = np.array([data for data_list in data_lists for data in data_list])
else:
    truncated_times = times[:]
    truncated_data = all_data[:]
problem = pints.MultiOutputProblem(pints_model, truncated_times, truncated_data)

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
# composed_boundaries = pints.RectangularBoundaries([0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [10, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4])
num_opts = 10
# Create log-likelihood
log_likelihood = pints.AR1LogLikelihood(problem)
param_names = ["beta", "kappa", "gamma",
               "rho_I", "sigma_I"]
# param_names = ["beta", "kappa", "gamma",
#                "rho_S", "sigma_S", "rho_E", "sigma_E", "rho_I", "sigma_I", "rho_R", "sigma_R"]

# Run 10 optimisations first
for j in range(num_opts):
    print(f"Iteration {j}")
    xs = composed_log_prior.sample(1)
    opt = pints.OptimisationController(log_likelihood, xs, boundaries=composed_boundaries, method=pints.CMAES)
    opt.set_max_unchanged_iterations(200, 0.00000001)
    opt.set_max_iterations(2500)
    opt_params, opt_values = opt.run()
    opt_param_df = pd.DataFrame({param_names[i]: [opt_params[i]] for i in range(len(param_names))})
    opt_param_df.to_csv(f"optimisation_outputs/{radius}/{trial}/opt_params_{j}.csv")

# Get the optimisation results
opt_params = []
for j in range(num_opts):
    opt_param_df = pd.read_csv(f"optimisation_outputs/{radius}/{trial}/opt_params_{j}.csv", index_col=0)
    opt_params.append(opt_param_df.to_numpy()[0])

chosen_opt = 2 # Use optimisation
pints_model._n_outputs = 4 # Want all four compartment results
# Draw Infected compartment for optimised parameters
num_params = pints_model.n_parameters()
opt_soln = pints_model.simulate(opt_params[chosen_opt][:num_params], truncated_times)
# opt_soln = model.simulate(parameters, times)

opt_S, opt_E, opt_I, opt_R = opt_soln[:, 0], opt_soln[:, 1], opt_soln[:, 2], opt_soln[:, 3]
# opt_I = opt_soln
# opt_V = opt_soln[:, 0]
plt.plot(truncated_times, opt_S, "-r", label="Optimised")
plt.plot(truncated_times, susceptible / pop_size, "--r", alpha=0.5, label="True data")
plt.xlabel("Time [days]")
plt.ylabel("Population")
plt.legend()

plt.plot(times, opt_E, "-b", label="Optimised")
plt.plot(times, exposed / pop_size, "--b", alpha=0.5, label="True data")
plt.xlabel("Time [days]")
plt.ylabel("Population")
plt.legend()

plt.plot(times, opt_I, "-g", label="Optimised")
plt.plot(times, infected / pop_size, "--g", alpha=0.5, label="True data")
plt.xlabel("Time [days]")
plt.ylabel("Population")
plt.legend()

plt.plot(times, opt_R, "-y", label="Optimised")
plt.plot(times, recovered / pop_size, "--y", alpha=0.5, label="True data")
plt.xlabel("Time [days]")
plt.ylabel("Population")
plt.legend()
plt.title(f"SEIR for {radius} optimisation {chosen_opt}")
plt.savefig(f"optimisation_outputs/{radius}/{trial}/seir_{radius}_{chosen_opt}.png")