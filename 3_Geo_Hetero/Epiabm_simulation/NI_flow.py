#
# Example simulation script running with Northern Ireland parameters
# Incorporates both age and spatial stratification.
# We also record secondary infections, serial intervals and generation times.
#

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

import pyEpiabm as pe

# Setup output for logging file
logging.basicConfig(filename='sim.log', filemode='w+', level=logging.DEBUG,
                    format=('%(asctime)s - %(name)s'
                            + '- %(levelname)s - %(message)s'))

# Set config file for Parameters
pe.Parameters.set_file(os.path.join(os.path.dirname(__file__),
                                    "NI_parameters.json"))

# Generate population from input file
# (Input converted from CovidSim with `microcell_conversion.py`)
file_loc = os.path.join(os.path.dirname(__file__),
                        "NI_inputs", "homogeneous_microcells.csv")
population = pe.routine.FilePopulationFactory.make_pop(file_loc,
                                                       random_seed=42)


# sim_ and file_params give details for the running of the simulations and
# where output should be written to.
sim_params = {"simulation_start_time": 0, "simulation_end_time": 90,
              "initial_infected_number": 100, "initial_infect_cell": True,
              "simulation_seed": 42}

file_params = {"output_file": "output_NI.csv",
               "output_dir": os.path.join(os.path.dirname(__file__),
                                          "simulation_outputs"),
               "spatial_output": True,
               "age_stratified": True}

dem_file_params = {"output_dir": os.path.join(os.path.dirname(__file__),
                                              "simulation_outputs"),
                   "spatial_output": True,
                   "age_output": True}

inf_history_params = {"output_dir": os.path.join(os.path.dirname(__file__),
                                                 "simulation_outputs"),
                      "status_output": True,
                      "infectiousness_output": True,
                      "secondary_infections_output": True,
                      "serial_interval_output": True,
                      "generation_time_output": True,
                      "compress": False}

# Create a simulation object, configure it with the parameters given, then
# run the simulation.
sim = pe.routine.Simulation()
sim.configure(
    population,
    [pe.sweep.InitialHouseholdSweep(),
     pe.sweep.InitialInfectedSweep(),
     pe.sweep.InitialisePlaceSweep(),
     pe.sweep.InitialDemographicsSweep(dem_file_params)],
    [
        pe.sweep.UpdatePlaceSweep(),
        pe.sweep.HouseholdSweep(),
        pe.sweep.PlaceSweep(),
        pe.sweep.SpatialSweep(),
        pe.sweep.QueueSweep(),
        pe.sweep.HostProgressionSweep(),
    ],
    sim_params,
    file_params,
    inf_history_params
)
sim.run_sweeps()
sim.compress_csv()

# Need to close the writer object at the end of each simulation.
del (sim.writer)
del (sim)

# Creation of a plot of results (plotter from spatial_simulation_flow)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
filename = os.path.join(os.path.dirname(__file__), "simulation_outputs",
                        "output_NI.csv")
SIRdf = pd.read_csv(filename)
total = SIRdf[list(SIRdf.filter(regex='InfectionStatus.Infect'))]
SIRdf["Infected"] = total.sum(axis=1)
SIRdf = SIRdf.groupby(["time"]).agg(
                                {"InfectionStatus.Susceptible": 'sum',
                                 "InfectionStatus.Exposed": 'sum',
                                 "Infected": 'sum',
                                 "InfectionStatus.Recovered": 'sum',
                                 "InfectionStatus.Dead": 'sum'})
SIRdf.rename(columns={"InfectionStatus.Susceptible": "Susceptible",
                      "InfectionStatus.Exposed": "Exposed",
                      "InfectionStatus.Recovered": "Recovered"},
             inplace=True)

# Create plot to show SIR curves against time
SIRdf.plot(y=["Susceptible", "Exposed", "Infected", "Recovered"])
plt.xlabel("Time [days]")
plt.ylabel("Number of people")
plt.title("Epiabm simulation")
plt.savefig(os.path.join(os.path.dirname(__file__),
            "simulation_outputs/simulation_flow_SEIR_plot.png"))

# Add Dead column (already exists as "InfectionStatus.Dead", just rename for clarity)
SIRdf.rename(columns={"InfectionStatus.Dead": "Dead"}, inplace=True)

# Reorder and save to CSV for SEIRD inference
SEIRD_df = SIRdf[["Susceptible", "Exposed", "Infected", "Recovered", "Dead"]]
SEIRD_df.to_csv(os.path.join(os.path.dirname(__file__),
                   "simulation_outputs", "seir.csv"))