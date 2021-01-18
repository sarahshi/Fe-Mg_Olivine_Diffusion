# %%
# from scipy.optimize import fsolve, root
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from numba import jit  #


# %%


def DH2O_Ol(T_C):
    """
    This function returns the diffusivity of H+ cations along the a [100] axis in
    olivine at a given temperature in Celcius.
    Ferriss et al. 2018 diffusivity in um2/s

    Parameters:
    T_C: Temperature in degrees C

    Returns:
    The diffusivity of H+ cations along the a [100] axis of olivine in um^2/S
    """
    T_K = T_C + 273
    DH2O = 1e12 * (10 ** (-5.4)) * np.exp(-130000 / (8.314 * T_K))
    return DH2O


# %%


def diffusion_kernel(dt, dx):

    delta = (dt) / ((dx) ** 2)

    # Diffusion Term
    kernel_1 = np.zeros(3)
    kernel_1[0] = 1
    kernel_1[1] = -2
    kernel_1[2] = 1

    # Central Difference Derivative. This is different than what is used in DIPRA which is a forward difference approx.
    # Remember to divide by 2 in diffusion step
    kernel_2 = np.zeros(3)
    kernel_2[0] = 1
    kernel_2[1] = 0
    kernel_2[2] = -1

    return kernel_1, kernel_2, delta


def VectorMaker(init_Concentration, N_points):
    """
    Makes flat inital profile
    """
    return init_Concentration * np.ones(N_points)


def boundary_cond(bounds_c):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions.
    # This can probably be accomplished with a list that pops next value or a generator
    pad = np.ones(3) * bounds_c

    if len(bounds_c) > 1:
        pad = (np.ones(3) * bounds_c[0], np.ones(3) * bounds_c[1])

    return pad


def step_condition(X_Intervals, Interval_concentrations, dx):
    """
    Makes Initial step conditions for diffusion models.
    Parameters:
    X_intervals - List of tuples which contain the starting and ending positions
                  for each interval of the initial diffusion model.

    Interval_concentrations - List of concentrations for each interval.

    dx - x spacing for array.

    output:
        step_x, step_c -- arrays of x positions and concentrations for step conditions
    """

    length = abs(np.max(X_Intervals) - np.min(X_Intervals))
    num_x = int(length / dx)

    segments_x = []
    segments_c = []

    # Define concentration steps.
    for idx, interval in enumerate(X_Intervals):
        interval_num = int((interval[1] - interval[0]) / length * num_x)
        int_x = np.linspace(start=interval[0], stop=interval[1] - dx, num=interval_num)
        int_c = np.ones_like(int_x) * Interval_concentrations[idx]
        segments_x.append(int_x)
        segments_c.append(int_c)
    step_x = np.concatenate(segments_x)
    step_c = np.concatenate(segments_c)
    return step_x, step_c


# %%

step = step_condition(((0, 75), (75, 250)), (0.859, 0.882), dx=2.5)

# %%


# TODO this function needs to be written to take an input that is either a Mn or Ni concentration or an XFo vector
# If XFo isnt supplied assume that concentration is in Fo units. If XFo is supplied I should make it an array that is covers the full concentration evolution.
# I will need a separate Diffusion function for Calcium which will be easier.

# I want to make this function as simple/fast as possible so I should ofload any logic
# I think the best way to do this is to have a wrapper function.  That runs the diffusion step function
# This wrapper funtion can be a good way to select the different elements. This could easily be written into a GUI


def diffusion_step(
    vector_c_in,
    vector_Fo_in,
    diffusivity_function,
    diff_kernel_1,
    der_kernel_2,
    delta,
    bounds_c,
    bounds_Fo,
):
    """
    Function that takes one step forward for Forsterite dependent diffusion.
    Parameters:
    bounds_c = tuple of left and right boundary conditions for diffusing species (Fixed bounds at the moment)
    bounds_c = tuple of left and right boundary conditions for Fo
    Output:

    """
    pad = np.ones(3)
    pad_c = (bounds_c[0] * pad, bounds_c[1] * pad)
    pad_Fo = (bounds_Fo[0] * pad, bounds_Fo[1] * pad)
    # pad generation can probably be taken out of the loop

    vector_c = np.concatenate([pad_c[0], vector_c_in, pad_c[1]])

    vector_Fo = np.concatenate(
        [pad_Fo[0], vector_Fo_in, pad_Fo[1]]
    )  # This might need to step through a larger matrix of values

    vector_D = diffusivity_function(vector_Fo)

    Diffusion = (np.convolve(vector_c, diff_kernel_1, mode="same") * vector_D)[3:-3]

    Diff_C = np.convolve(vector_c, der_kernel_2, mode="same")[3:-3]

    Diff_D = np.convolve(vector_D, der_kernel_2, mode="same")[3:-3]

    vector_out = vector_c_in + delta * (Diffusion + (Diff_C * Diff_D) / 2)

    # vector_c_in + delta*(Diffusion + (Diff_C* Diff_D))
    out = (Diff_C * Diff_D) * delta
    return vector_out


def interp_data():
    # Interpolate Data to fit dx
    # Kriging Function is probably good for this but maybe a bit overkill
    # .

    return None


"""
3 Vectors
1) Diffusing Concentration 
2) Diffusing or previously diffused Fo array
3) Diffusivities  Column 
"""
# %%

# %%
"""
One Idea is to do a refining grid search. Do a really sparse dt model with tight dX and then refine. based on when the best fit is bracketed 
"""
# %%
# Elemental Diffusivities


def FO2(T, P, Buffer):
    " this function should be able to calculate FO2 for a given buffer at various pressures and temps"
    return None


def D_Fo(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=201000, R=8.3145):
    """
    Function that calculates the diffusivity for Forsterite (and Mn) in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO  Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a -- degrees
        beta, - minimum angle to [010] axis b -- degrees
        gamma - minimum angle to [001] axis c -- degrees

    Returns: Diffusivity function That's only input it is:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s

    """

    def D_Func_Fo(XFo):
        """Returns diffusivity and derivative of diffusivity at each point in an olivine for a given oxygen fugacity, proportion of forsterite, activation energy, pressure, gas constant, temperature, and crystallographic orientation. """
        tenterm = 10 ** -9.21
        fugacityterm = (fO2 / (1e-7)) ** (1.0 / 6.0)
        forsteriteterm = 10 ** (3.0 * (0.9 - XFo))
        D = (
            tenterm
            * fugacityterm
            * forsteriteterm
            * np.exp(-(EFo + 7 * (10 ** -6 * (P - 10 ** 5))) / (R * T))
        )
        # This next term should be calculated with angle in degrees.
        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di = (
            ((1 / 6) * D * (np.cos(alpha_rad) ** 2))
            + ((1 / 6) * D * (np.cos(beta_rad) ** 2))
            + (D * (np.cos(gamma_rad) ** 2))
        )  # Use this term for crystallographic orientation constraints from EBSD.

        return Di  # units of m2/s

    if XFo is not None:
        return D_Func_Fo(XFo)

    return D_Func_Fo


def D_Ni(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=220000, R=8.3145):
    """
    Function that calculates the diffusivity for Mn in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO Pa
        E, - Activation Energy 220000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a
        beta, - minimum angle to [010] axis b
        gamma - minimum angle to [001] axis c

    Returns: Diffusivity function That's only input it is:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s

    """

    def D_Func_Ni(XFo):
        """Returns diffusivity and derivative of diffusivity at each point in an olivine for a given oxygen fugacity, proportion of forsterite, activation energy, pressure, gas constant, temperature, and crystallographic orientation. """
        tenterm = 3.84 * 10 ** -9
        fugacityterm = (fO2 / (1e-6)) ** (1.0 / 4.25)
        forsteriteterm = 10 ** (1.5 * (0.9 - XFo))
        D = (
            tenterm
            * fugacityterm
            * forsteriteterm
            * np.exp(-(EFo + 7 * (10 ** -6 * (P - 10 ** 5))) / (R * T))
        )
        # This next term should be calculated with angle in degrees.

        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di = (
            ((1 / 6) * D * (np.cos(alpha_rad) ** 2))
            + ((1 / 6) * D * (np.cos(beta_rad) ** 2))
            + (D * (np.cos(gamma_rad) ** 2))
        )  # Use this term for crystallographic orientation constraints from EBSD.

        return Di  # units of m2/s

    if XFo is not None:
        return D_Func_Fo(XFo)

    return D_Func_Ni


def D_Func_Ca(T, fO2, alpha, beta, gamma, R=8.3145):
    """
    Function that calculates the diffusivity for Mn in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a
        beta, - minimum angle to [010] axis b
        gamma - minimum angle to [001] axis c

    Returns: Diffusivity function That's only input it is:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s):
    """

    fugacityterm = (fO2 / (1e-7)) ** (0.3)

    Da = 16.59 * 10 ** -12 * fugacityterm * np.exp(-(193000) / (R * T))
    Db = 34.67 * 10 ** -12 * fugacityterm * np.exp(-(201000) / (R * T))
    Dc = 95.49 * 10 ** -12 * fugacityterm * np.exp(-(207000) / (R * T))
    # This next term should be calculated with angle in degrees.

    alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))

    Di = (
        Da * (np.cos(alpha_rad) ** 2)
        + Db * (np.cos(beta_rad) ** 2)
        + (Dc * (np.cos(gamma_rad) ** 2))
    )
    # Use this term for crystallographic orientation constraints from EBSD.

    return Di  # units of m2/s


# %%


# %%

# %%

# %%

# Enumerate for loop or iterate index number this allows us to input an array of Fo which we can use to compare.
# Pad should also be able to be iterated through time


def timestepper(vector_c_in, vector_Fo_in, diffusivity_function, bounds_c, timesteps):
    kernel_1, kernel_2, delta = diffusion_kernel(dt=dt, dx=dx)
    # At the moment only handles Fo but should diffuse other elements too with a little modification
    results = np.zeros((timesteps, len(vector_c_in)))
    n = 0
    for time in range(timesteps):
        vector_c_in = diffusion_step(
            vector_c_in=vector_c_in,
            vector_Fo_in=vector_Fo_in,
            diffusivity_function=diffusivity_function,
            diff_kernel_1=kernel_1,
            der_kernel_2=kernel_2,
            delta=delta,
            bounds_c=bounds_c,
            bounds_Fo=bounds_c,
        )
        vector_Fo_in = vector_c_in
        results[n] = vector_Fo_in
        n = n + 1
    return results


# %%

fO2 = 1e-7  # 2.006191e-05 # Pa
EFo = 201000.0  # J/mol
P = 100000  # 200000000. # Pa
R = 8.3145  # J/molK
T = 1200 + 273.15  # T in kelvin


D_FO_Func = D_Fo(
    T=T, P=P, fO2=fO2, alpha=90, beta=90, gamma=0, XFo=None, EFo=201000, R=8.3145
)

dx_micron = 2.5
dx = dx_micron * 1e-6  # m
dt = 4000  # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt * Di) / (dx ** 2)
print(CFL)
# delta = (dt)/ ((dx) ** 2)

# (vector_c_in, vector_Fo_in, diffusivity_function, bounds_c, timesteps)
# vector_c_in, vector_Fo_in, diffusivity_function, diff_kernel_1, der_kernel_2, delta, bounds_c, bounds_Fo
#  Step funct - Maybe write to move inflection point X_Intervals, Interval_concentrations, dx
# Temperature is the biggest error in the Diffusivity. Maybe I can include the error in other Params too with Tau
# Maybe plot the trace of best fits?
def Diffusion_call():
    D_FO_Func = D_Fo(
        T=T, P=P, fO2=fO2, alpha=90, beta=90, gamma=0, XFo=None, EFo=201000, R=8.3145
    )
    inflection_x = 90  # Microns to the inflection point
    step_x, step_c = step_condition(
        ((0, inflect_x), (inflect_x, 250)), (0.859, 0.882), dx_micron
    )
    X_Intervals, Interval_concentrations, dx

    Fo_diffusion_results = timestepper(
        vector_c_in=vector_c_in,
        vector_Fo_in=vector_Fo_in,
        diffusivity_function=D_FO_Func,
        bounds_c=bounds_c,
        timesteps=timesteps,
    )
    return None


# %%
inflect_x = 90  # Microns to the inflection point

step_x, step_c = step_condition(
    ((0, inflect_x), (inflect_x, 250)), (0.859, 0.882), dx_micron
)
bounds_c = (step_c[0], step_c[-1])
vector_c_in = step_c
vector_Fo_in = vector_c_in

Total_time = 500 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)


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
Fo_diffusion_results[-1]
# %%
#%%
# In the AM write code to make a PDF for all data.


def Best_fit_R2(results, data_interp, dt):
    # This minimizes for sum of residuals^2
    # Maybe should be maximizing for likelihood

    residual = results - data_interp
    sum_r2 = np.sum(residual ** 2, axis=1)
    idx_min = np.argmin(sum_r2)

    sum_r2[idx_min] * 1.05

    time = (idx_min + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)
    return time, idx_min, sum_r2


def Best_fit_Liklihood(results, data_interp, sigma, dt):
    # This minimizes for sum of residuals^2
    # Maybe should be maximizing for likelihood

    residual = results - data_interp
    sum_r2 = np.sum(residual ** 2, axis=1)
    idx_max = np.argmin(sum_r2)

    sum_r2[idx_min] * 1.05

    time = (idx_min + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)
    return time, idx_min, sum_r2


def seconds_to_days(time_seconds):
    return time_seconds / (60 * 60 * 24)


# Find a way to return time steps ranges for all data within 5+/-5% of the minimum
# find roots near to a value then report those ranges.
#%%

# To fit data I should interpolate the data to the dx spacing. I might be able to use Kringing to get error bars.
# Weighted Residuals. Weights will be inverse of STD^2. If STdevs are the same I can use normal functions.
# Interpolate data to results dx (use Kringing to improve interp )
Fo_interp = interp.interp1d(ol40_x, ol40_Fo)
data_interp = Fo_interp(distance)

best_time, idx_min, sum_r2 = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
best_time_days = seconds_to_days(best_time)
plt.plot(sum_r2)
# %%
# This mentod for estimating time is pretty good but doesnt save enough gradient info.
# depending on how I set up MCMC I need to think how the gradient gets inputted.
# Maybe instead of best fit I return vectors of times and sum(reisuals^2)
time_range = np.where((sum_r2 < sum_r2[idx_min] * 1.05))
min_time = (time_range[0].min() + 1) * dt / (60 * 60 * 24)  # days
max_time = (time_range[0].max() + 1) * dt / (60 * 60 * 24)  # days

ol40 = np.loadtxt(
    "/Users/henry/Python Files/Fe-Mg Diffusion/AZ18_WHT06_ol40_C-Prof.txt"
)
ol40_Fo = ol40[:, 1]
ol40_x = ol40[:, 0]

num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
plt.plot(distance, Fo_diffusion_results[idx_min])
plt.xlabel("Micron")
plt.ylabel("Fo")
# plt.ylim(0.65, 0.81)


plt.plot(ol40_x, ol40_Fo)
plt.plot(step_x, step_c)

# %%

# %%

# Sum of the Residuals^2 Write it so that it evaluates it at certain intervals
# Maybe divide timesteps into even amounts and then evaluate every N timesteps.

# Also think about 2-D diffusion for profiles with no central plateau. A 2D convolution algorithm would be helpful.
# %%
# Write Pad function. Each Step needs a pad
# 1) Constant Boundary - Diffusion
# 2) Constant Boundary - No_Diffusion at edge
# 3) Changing Boundary - Ascent Path
# %%

# This is the Kriging Interpolation I wrote for another code but I should modify it for this to produce an array of interpreded value and errors.


def Krige_Interpolate(
    dataframe, sample_name, column_name, SIMS_session=None, label=None
):

    Sample_df = dataframe.loc[dataframe["sample"] == sample_name]
    if SIMS_session is not None:
        Sample_df = Sample_df.loc[Sample_df["sims_session"] == SIMS_session]
        dataframe = dataframe.loc[dataframe["sims_session"] == SIMS_session]

    X = Sample_df["time_stamp"].to_numpy(dtype=float)
    y = Sample_df[column_name].to_numpy(dtype=float)

    X = (X[1:] + X[:-1]) / 2
    y = (y[1:] + y[:-1]) / 2
    X = X[0::2]
    y = y[0::2]
    uk = OrdinaryKriging(
        X,
        np.zeros(X.shape),
        y,
        weight=False,
        nlags=6,
        # variogram_model="linear",)
        # variogram_model="linear", variogram_parameters={'slope': 5e-18, 'nugget': 6e-7})
        variogram_model="gaussian",
        variogram_parameters={"sill": 1e-4, "range": 9e12, "nugget": 0},
    )

    X_pred = dataframe["time_stamp"].to_numpy(dtype=float)
    # uk.display_variogram_model()
    # 480000000000.0
    # 300000000000.0
    # 4000000000000.0
    y_pred, y_std = uk.execute("grid", X_pred, np.array([0.0]))
    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)

    return dataframe["time_stamp"], y_pred, y_std


fig, ax = plt.subplots(figsize=(12, 6))

X_pred, y_pred, y_std = Krige_Interpolate(
    Samples_db, "Herasil", "17_30si_ratio", SIMS_session=1
)
ax.plot(X_pred, y_pred, label="Predicted values")

ax.fill_between(
    X_pred,
    y_pred - 2 * y_std,
    y_pred + 2 * y_std,
    alpha=0.3,
    label="Confidence interval",
)

Sample_plot(
    Samples_db, "Herasil", "17_30si_ratio", ax=ax, SIMS_session=1, label="Session 1"
)

#%% Single fit function for MCMC
# Inputs Params = Every parameter that gets varied
# Best fit modeled data.
# Maybe there is a way to write best fit times to a seperate time without returning it in the main function.

# I mean I care less about the error on the input parameters than about getting the times variablitiy in times.
# %%

# %%

# %%

# %%

# %%
