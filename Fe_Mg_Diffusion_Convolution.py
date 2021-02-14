# %%
# from scipy.optimize import fsolve, root
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from numba import jit  #
from pykrige import OrdinaryKriging
import mc3

%matplotlib inline
#%%
ol40 = np.loadtxt(
    "/Users/henry/Python Files/Fe-Mg Diffusion/AZ18_WHT06_ol40_C-Prof.txt"
)
ol40_Fo = ol40[:, 1]
ol40_x = ol40[:, 0]

plt.plot(ol40_x, ol40_Fo)
plt.plot(step_x, step_c)
# %%


def diffusion_kernel(dt, dx):
    """
    returns the relevant kernel for 1D diffusion and a defivative of the Fo#
    dt = time step Seconds
    dx = spatial step Meters

    """
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
    Creates a profile with a flat initial concentration
    """
    return init_Concentration * np.ones(N_points)


def boundary_cond(bounds_c):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions.
    # This can probably be accomplished with a list that pops next value
    pad = np.ones(3) * bounds_c

    if len(bounds_c) > 1:
        pad = (np.ones(3) * bounds_c[0], np.ones(3) * bounds_c[1])

    return pad


def step_condition(X_Intervals, Interval_concentrations, dx):
    """
    Creates a step function for diffusion models
    Parameters:
        X_Intervals - List of tuples - each tuple has start and end point of step
        Interval_concentrations - list of concentrations coresponding to each interval
        dx - spacing between x points.

    Returns
        step_x - array of x coordinates
        step_c - array of concentrations
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
    " this function shold be able to calculate FO2"
    return None


def D_Fo(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=201000):
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
        R = 8.3145
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


def D_Ni(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=220000):
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
        R = 8.3145
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
        return D_Func_Ni(XFo)

    return D_Func_Ni


def D_Func_Ca(
    T,
    fO2,
    alpha,
    beta,
    gamma,
):
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

    Returns: Diffusivity function That's only input is XFo:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s):
    """
    R = 8.3145
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


def timestepper(
    vector_c_in, vector_Fo_in, diffusivity_function, bounds_c, timesteps, **kwargs
):
    """
    Iterates multiple diffusion steps
    """
    kernel_1, kernel_2, delta = diffusion_kernel(dt=dt, dx=dx)

    # At the moment only handles Fo but should diffuse other elements too with a little modification
    results = np.zeros((timesteps, len(vector_c_in)))
    for n, _ in enumerate(range(timesteps)):
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
    return results


# %%


def Best_fit_R2(results, data_interp, dt):
    # Should be Chi2 but looks more like likelihood?

    residual = results - data_interp
    sum_r2 = np.sum(residual ** 2, axis=1)
    idx_min = np.argmin(sum_r2)

    # sum_r2[idx_min] * 1.05

    time = (idx_min + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)
    return time, idx_min, sum_r2


def Best_fit_Chi2(results, data_interp, sigma, dt, sigma_min=1e-4):
    # This minimizes for sum of residuals^2/sigma

    residual = results - data_interp
    sum_r2 = np.sum((residual ** 2) / (sigma + sigma_min) ** 2, axis=1)
    idx_min = np.argmin(sum_r2)

    time = (idx_min + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)
    return time, idx_min, sum_r2


# %%
# Write Pad function. Each Step needs a pad
# 1) Constant Boundary - Diffusion
# 2) Constant Boundary - No_Diffusion at edge
# 3) Changing Boundary - Ascent Path
# %%


# %%


def Krige_Interpolate(
    X, Y, new_X, variogram_parameters={"slope": 1e-4, "nugget": 1e-5}
):
    # X, Y, new_X, variogram_parameters={"sill": 1e3, "range": 1e2, "nugget": 0.0001}

    uk = OrdinaryKriging(
        X,
        np.zeros(X.shape),
        Y,
        pseudo_inv=True,
        # weight=True,
        # nlags=2,
        # variogram_model="gaussian",
        # exact_values = False,
        variogram_model="linear",
        variogram_parameters=variogram_parameters
        # variogram_model="gaussian",
        # variogram_parameters=variogram_parameters,
    )

    y_pred, y_std = uk.execute("grid", new_X, np.array([0.0]))
    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)

    return new_X, y_pred, y_std

#%%
X_interp, Y_Interp, Y_interp_std = Krige_Interpolate(ol40_x, ol40_Fo, step_x)

plt.plot(ol40_x, ol40_Fo)
plt.plot(step_x, Y_Interp)
plt.plot(step_x, Y_Interp + 2 * Y_interp_std)
plt.plot(step_x, Y_Interp - 2 * Y_interp_std)
# %%

"""
Find max time steps from 3 point diffusion model 
1   
"""
# TODO Sort Variables by whether they will be sampled with PyMC or not.
# Write subfunctions to handle general and specific Diffusion model set ups.


def Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calcualte the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt,
    output_full=False,
    **kwargs
):

    T, P, fO2, inflect_x, edge_x1, edge_x2, edge_c, center_c = p

    D_FO_Func = D_Fo(
        T=T,
        P=P,
        fO2=fO2,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        EFo=EFo,
    )
    dx = dx_micron * 1e-6  # m

    # sets up a single stairstep for diffusion models
    X_Intervals = ((edge_x1, inflect_x), (inflect_x, edge_x2))
    Interval_concentrations = (edge_c, center_c)
    step_x, step_c = step_condition(X_Intervals, Interval_concentrations, dx_micron)
    #  Only implmented for Fo# Zoning at the moment.

    Fo_diffusion_results = timestepper(
        vector_c_in=step_c,
        vector_Fo_in=step_c,
        diffusivity_function=D_FO_Func,
        bounds_c=(edge_c, center_c),
        timesteps=timesteps,
        dx=dx,
        dt=dt,
    )

    time, idx_min, sum_r2 = Best_fit_Chi2(
        Fo_diffusion_results, data_interp, std_interp, dt, **kwargs
    )

    # time, idx_min, sum_r2 = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
    if output_full:
        return time, idx_min, sum_r2, Fo_diffusion_results
    return Fo_diffusion_results[idx_min]


# %%

fO2 = 1e-7  # 2.006191e-05 # Pa
EFo = 201000.0  # J/mol
P = 100000  # 200000000. # Pa
R = 8.3145  # J/molK
T = 1200 + 273.15  # T in kelvin


# EFo=201000 should be +/- 8000

D_FO_Func = D_Fo(
    T=T,
    P=P,
    fO2=fO2,
    alpha=90,
    beta=90,
    gamma=0,
    XFo=None,
    EFo=201000,
)

dx_micron = 2.5
dx = dx_micron * 1e-6  # m
dt = 4000  # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt * Di) / (dx ** 2)
print(CFL)
# delta = (dt)/ ((dx) ** 2)

# %%

"""
Conditions to solve for:
Position of edge accounts for crystallization etc... Maybe not as important 
Inflection Point for step 

Initial concentration 
Edge Concentration

Is there some way to constrain simultaneous crystallization and diffusion by shape?

Kd for Olivine that have lost central concentration 
"""


inflect_x = 90  # Microns to the inflection point

step_x, step_c = step_condition(
    ((0, inflect_x), (inflect_x, 270)), (0.859, 0.882), dx_micron
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

# To fit data I should interpolate the data to the dx spacing. I might be able to use Kringing to get error bars.
# Weighted Residuals. Weights will be inverse of STD^2. If STdevs are the same I can use normal functions.
# Interpolate data to results dx (use Kringing to improve interp )
Fo_interp = interp.interp1d(ol40_x, ol40_Fo)
data_interp = Fo_interp(distance)

time, idx_min, sum_r2 = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
time2, idx_max, sum_chi2 = Best_fit_Chi2(
    Fo_diffusion_results, data_interp, std_interp, dt
)

plt.plot(sum_r2)

# %%
# This mentod for estimating time is pretty good but doesnt save enough gradient info.
# depending on how I set up MCMC I need to think how the gradient gets inputted.
# Maybe instead of best fit I return vectors of times and sum(reisuals^2)
time_range = np.where((sum_r2 < sum_r2[idx_min] * 1.05))
min_time = (time_range[0].min() + 1) * dt / (60 * 60 * 24)  # days
max_time = (time_range[0].max() + 1) * dt / (60 * 60 * 24)  # days
# %%

num = len(vector_c_in)
distance = np.linspace(0, dx_micron * (num), num)
plt.plot(distance, Fo_diffusion_results[4572])
plt.plot(distance, Fo_diffusion_results[idx_min])
plt.xlabel("Micron")
plt.ylabel("Fo")
# plt.ylim(0.65, 0.81)
#%%
ol40 = np.loadtxt(
    "/Users/henry/Python Files/Fe-Mg Diffusion/AZ18_WHT06_ol40_C-Prof.txt"
)
ol40_Fo = ol40[:, 1]
ol40_x = ol40[:, 0]

plt.plot(ol40_x, ol40_Fo)
plt.plot(step_x, step_c)
# %%
Fo_interp = interp.interp1d(ol40_x, ol40_Fo)
data_interp = Fo_interp(distance)

X_interp, Y_interp, Y_interp_std = Krige_Interpolate(
    ol40_x,
    ol40_Fo,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(ol40_x, ol40_Fo, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%
dx_micron = 2.5
dt = 4000


fO2 = 1e-7  # 2.006191e-05 # Pa
EFo = 201000.0  # J/mol
P = 100000  # 200000000. # Pa
R = 8.3145  # J/molK
T = 1250 + 273.15  # 1200 + 273.15  # T in kelvin


inflection_x = 90
edge_x1 = 0
edge_x2 = 270
edge_c = 0.8584
center_c = 0.8815

alpha = 90
beta = 90
gamma = 0

data_interp = Y_interp
std_interp = Y_interp_std

Total_time = 500 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt,
    output_full=True,
)

# %%
plt.plot(X_interp, data_interp)

plt.plot(X_interp, Fo_diffusion_results[idx_min])
plt.plot(X_interp, Fo_diffusion_results[0])
#plt.plot(X_interp, Fo_diffusion_results[1779])
#plt.plot(X_interp, Fo_diffusion_results[3934])
#plt.plot(X_interp, Fo_diffusion_results[3842])
# %%
Z = Best_fit_Chi2(Fo_diffusion_results, data_interp, Y_interp_std, dt, sigma_min=1e-4)
V = Best_fit_R2(Fo_diffusion_results, data_interp, dt)
# %%
# %%

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(3) == 2)[0]
# %%
time_range * dt / (60 * 60 * 24)
# %%
# sum_r2_1200 = sum_r2
sum_r2_1250 = sum_r2
# %%

#141 312 208

# %%


