# %%
# from Users.henry.Python Files.Electrical Conductivity SIMS Data.NS_ConductivityOlivines import Sample_Interpolate
import Fe_Mg_Diffusion_Convolution
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp


#%%

excel_path = "/Users/henry/Documents/Research/Mantle Xenoliths/AZ18 Samples/Microprobe July 2020/Final Microprobe Data/AB+HT_EMPA_July_2020_SIMS-Mounts_HenrysSamples.xlsx"

Ol_Profiles = pd.read_excel(
    excel_path,
    sheet_name="Olivine Profiles_WDS+EDS",
    header=52,
    index_col="DataSet/Point",
    engine="openpyxl",
)
# %%
Names = Ol_Profiles.Name.unique()
# %%


def get_C_prof(prof_name, DF, Element="Fo#"):
    prof = DF.loc[DF.Name == prof_name]
    distance_um = prof["Distance µm"]
    concentration = prof[Element]
    return distance_um.to_numpy(), concentration.to_numpy()


# %%
x, y = get_C_prof("AZ18_WHT06_ol41newname_prof_", Ol_Profiles)
# %%
for n in Names:
    fig, ax = plt.subplots()
    x, y = get_C_prof(n, Ol_Profiles)
    plt.plot(x, y)
    plt.title(n)
# %%

# Diffusion model constants # change eventually

# Give better pressures and fO2

fO2 = 1e-7  # 2.006191e-05 # Pa
EFo = 201000.0  # J/mol
P = 100000  # 200000000. # Pa
R = 8.3145  # J/molK
T = 1220 + 273.15  # T in kelvin

# %%
# Write a single function that takes all of these inputs and
sample = "AZ_WHT06_ol48_lasermount_xenocryst2_prof"

x, Fo = get_C_prof(sample, Ol_Profiles)
# Select these Parameters
D_FO_Func = D_Fo(
    T=T, P=P, fO2=fO2, alpha=90, beta=90, gamma=0, XFo=None, EFo=201000, R=8.3145
)

dx_micron = 1
dx = dx_micron * 1e-6  # m
dt = 60  # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt * Di) / (dx ** 2)
print(CFL)

Total_time = 30 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

max_length = 60  # Profile length Microns
inflect_x = 25  # Microns to the inflection point

step_x, step_c = step_condition(
    ((0, inflect_x), (inflect_x, max_length)), (0.849, 0.882), dx_micron
)
bounds_c = (step_c[0], step_c[-1])
vector_c_in = step_c
vector_Fo_in = vector_c_in


Prof_length = np.max(step_x)  # µm
x_num = Prof_length / dx_micron

num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
# %%

Fo_diffusion_results = timestepper(
    vector_c_in=vector_c_in,
    vector_Fo_in=vector_Fo_in,
    diffusivity_function=D_FO_Func,
    bounds_c=bounds_c,
    timesteps=timesteps,
)

Fo_interp = interp.interp1d(x, Fo)
data_interp = Fo_interp(distance)

best_time, idx_min, sum_r2 = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
best_time_days = seconds_to_days(best_time)

# %%
num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(distance, Fo_diffusion_results[idx_min], label="Best_fit_model")
plt.xlabel("Micron")
plt.ylabel("Fo")

plt.plot(x, Fo, label="Data")
plt.plot(step_x, step_c, Label="Initial Condition")
Title = sample
plt.title(Title)
model_time = "Best fit time: " + str(round(best_time_days, 2)) + " days"
plt.annotate(s=model_time, xy=(0.8, 0.05), xycoords="axes fraction")
plt.legend()
# %%


# %%
# Write a single function that takes all of these inputs and
sample = "AZ18_WHT06_ol43_xenocryst_Lasermount_prof_"

x, Fo = get_C_prof(sample, Ol_Profiles)
x = x[8:-1]
Fo = Fo[8:-1]
# Select these Parameters
D_FO_Func = D_Fo(
    T=T, P=P, fO2=fO2, alpha=90, beta=90, gamma=0, XFo=None, EFo=201000, R=8.3145
)

dx_micron = 1
dx = dx_micron * 1e-6  # m
dt = 60  # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt * Di) / (dx ** 2)
print(CFL)

Total_time = 30 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

max_length = 120  # Profile length Microns
inflect_x = 37  # Microns to the inflection point

step_x, step_c = step_condition(
    ((0, inflect_x), (inflect_x, max_length)), (0.852, 0.895), dx_micron
)
bounds_c = (step_c[0], step_c[-1])
vector_c_in = step_c
vector_Fo_in = vector_c_in


Prof_length = np.max(step_x)  # µm
x_num = Prof_length / dx_micron

num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
# %%

Fo_diffusion_results = timestepper(
    vector_c_in=vector_c_in,
    vector_Fo_in=vector_Fo_in,
    diffusivity_function=D_FO_Func,
    bounds_c=bounds_c,
    timesteps=timesteps,
)

Fo_interp = interp.interp1d(x, Fo)
data_interp = Fo_interp(distance)

best_time, idx_min, sum_r2 = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
best_time_days = seconds_to_days(best_time)

# %%
num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(distance, Fo_diffusion_results[idx_min], label="Best_fit_model")
plt.xlabel("Micron")
plt.ylabel("Fo")

plt.plot(x, Fo, label="Data")
plt.plot(step_x, step_c, Label="Initial Condition")
Title = sample
plt.title(Title)
model_time = "Best fit time: " + str(round(best_time_days, 2)) + " days"
plt.annotate(s=model_time, xy=(0.8, 0.05), xycoords="axes fraction")
plt.legend()
# %%
