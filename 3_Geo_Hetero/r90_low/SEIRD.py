import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
from scipy import integrate, optimize
import pints

data = pd.read_csv("./simulation_outputs/SEIRD_timeseries.csv", header=None)
data = data.iloc[1:, 3]

# SEIRD ODE system
def seird(y, t, beta, kappa, gamma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - kappa * E
    dIdt = kappa * E - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def simulate(func, parameters, y0, times):
    sol = scipy.integrate.odeint(func, y0, times, args=tuple(parameters))
    return sol

y0 = [662754/662854, 0, 1-662754/662854, 0, 0]  # S, E, I, R, D

class PintsSEIRD(pints.ForwardModel):
    def n_parameters(self):
        return 4
    def n_outputs(self):
        return 1
    def simulate(self, parameters, times):
        return simulate(seird, parameters, y0, times)
