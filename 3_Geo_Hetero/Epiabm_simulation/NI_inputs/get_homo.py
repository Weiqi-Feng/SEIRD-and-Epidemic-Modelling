import pandas as pd
import numpy as np

# Load original microcell data
df = pd.read_csv("NI_microcells.csv")
num_cells = len(df)

def distribute_evenly(total, size):
    base = total // size
    remainder = total % size
    values = np.full(size, base, dtype=int)
    extra_indices = np.random.choice(size, remainder, replace=False)
    values[extra_indices] += 1
    return values

# Create new DataFrame
df_homogeneous = df.copy()

# 1. Susceptible
total_pop = df["Susceptible"].sum()
df_homogeneous["Susceptible"] = distribute_evenly(total_pop, num_cells)

# 2. Household number
total_households = df["household_number"].sum()
df_homogeneous["household_number"] = distribute_evenly(total_households, num_cells)

# 3. Place number
total_places = df["place_number"].sum()
df_homogeneous["place_number"] = distribute_evenly(total_places, num_cells)

# 4. Save the new homogeneous file
df_homogeneous.to_csv("homogeneous_microcells.csv", index=False)
