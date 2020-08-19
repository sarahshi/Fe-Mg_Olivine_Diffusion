# %%
from scipy.optimize import fsolve, root
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

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

def VectorMaker(init_Concentration, N_points):
    return init_Concentration * np.ones(N_points)


def diffusion_kernel(dt, dx):

    delta = (dt) / ((dx) ** 2)

    # Diffusion Term
    kernel_1 = np.zeros(3)
    kernel_1[1] = -2 
    kernel_1[0] = 1
    kernel_1[2] = 1

    # Forward Difference Derivative. I might try a central differencing equation. 
    kernel_2 = np.zeros(3)
    kernel_2[1] = 1 
    kernel_2[0] = 0
    kernel_2[2] = -1 

    return kernel_1, kernel_2, delta



def boundary_cond(C=0):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions. 
    # This can probably be accomplished with a list that pops next value 
    pad = np.ones(3) * C
    return pad

def step_condition(X_Intervals, Interval_concentrations, dx):
    # Define concentration steps. 

    return None

# Function to 




# TODO this function needs to be written to take an input that is either a Mn or Ni concentration or an XFo vector
# If XFo isnt supplied assume that concentration is in Fo units. If XFo is supplied I should make it an array that is covers the full concentration evolution.  
# I will need a separate Diffusion function for Calcium which will be easier. 

# I want to make this function as simple/fast as possible so I should ofload any logic 
# I think the best way to do this is to have a wrapper function.  That runs the diffusion step function 
# This wrapper funtion can be a good way to select the different elements. This could easily be written into a GUI

def diffusion_step(vector_c_in, vector_Fo_in, diffusivity_function, diff_kernel_1, der_kernel_2, delta, pad_c, pad_Fo):
    """
    Function that takes one step forward for Forsterite dependent diffusion. 
    Parameters:

    Output:

    """
    # pad = np.ones(3) * Bound_Concentration put back in

    vector_c = np.concatenate([pad_c, vector_c_in, pad_c])

    vector_Fo = np.concatenate([pad_Fo, vector_Fo_in, pad_Fo])

    vector_D = diffusivity_function(vector_Fo)

    Diffusion = (np.convolve(vector_c, diff_kernel_1, mode="same")* vector_D)[3:-3]



    Diff_C = np.convolve(vector_c, der_kernel_2, mode="same")[3:-3]

    Diff_D = np.convolve(vector_D, der_kernel_2, mode="same")[3:-3]

    vector_out = vector_c_in + delta*(Diffusion + (Diff_C* Diff_D))

    #vector_c_in + delta*(Diffusion + (Diff_C* Diff_D))
    out = (Diff_C* Diff_D)*delta
    return vector_out


def interp_data():
    # Interpolate Data to fit dx

    return None

def residuals(vector_out_c, initial_interp):
    # Function that Keeps track of the residuals^2 and searches for a minimum. 
    #I think this is faster with a list 
    # Append residuals to a list but doesnt give obvious cutoff time... 
    return None


def Diffusion_Stepper(Element, dt, dx, time):
    # Runs diffusion through time.
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

def FO2(T, P , Buffer):
    " this function shold be able to calculate FO2"
    return None

def D_Fo(T, P, fO2, alpha, beta, gamma, XFo=None, EFo= 201000, R= 8.3145):
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
        tenterm = (10**-9.21)
        fugacityterm = (fO2/(1e-7))**(1./6.)
        forsteriteterm = 10**(3.*(0.9-XFo))
        D = tenterm * fugacityterm * forsteriteterm * np.exp(-(EFo + 7 * (10**-6 * (P-10**5)))/(R*T))
        # This next term should be calculated with angle in degrees. 
        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di =  ((1/6) * D * (np.cos(alpha_rad)**2)) + ((1/6) * D * (np.cos(beta_rad)**2)) + (D * (np.cos(gamma_rad)**2)) # Use this term for crystallographic orientation constraints from EBSD. 
        
        return Di # units of m2/s

    if XFo is not None:
        return D_Func_Fo(XFo)

    return D_Func_Fo

def D_Ni(T, P, fO2, alpha, beta, gamma, XFo=None, EFo= 220000, R= 8.3145):
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
        tenterm = (3.84 * 10**-9)
        fugacityterm = (fO2/(1e-6))**(1./4.25)
        forsteriteterm = 10**(1.5*(0.9-XFo))
        D = tenterm * fugacityterm * forsteriteterm * np.exp(-(EFo + 7 * (10**-6 * (P-10**5)))/(R*T))
        # This next term should be calculated with angle in degrees. 

        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di =  ((1/6) * D * (np.cos(alpha_rad)**2)) + ((1/6) * D * (np.cos(beta_rad)**2)) + (D * (np.cos(gamma_rad)**2)) # Use this term for crystallographic orientation constraints from EBSD. 
        
        return Di # units of m2/s

    if XFo is not None:
        return D_Func_Fo(XFo)

    return D_Func_Ni

def D_Func_Ca(T, fO2, alpha, beta, gamma, R= 8.3145):
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
    
    fugacityterm = (fO2/(1e-7))**(0.3)


    Da = 16.59 * 10**-12 * fugacityterm * np.exp(-(193000)/(R*T))
    Db =  34.67 * 10**-12 * fugacityterm * np.exp(-(201000)/(R*T))
    Dc =  95.49 * 10**-12 * fugacityterm * np.exp(-(207000)/(R*T))
    # This next term should be calculated with angle in degrees. 

    alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))

    Di =   Da * (np.cos(alpha_rad)**2) +  Db * (np.cos(beta_rad)**2) + (Dc * (np.cos(gamma_rad)**2)) 
    # Use this term for crystallographic orientation constraints from EBSD. 
    
    return Di# units of m2/s

# %%



# %%

fO2 = 1e-7 #2.006191e-05 # Pa
EFo = 201000. # J/mol
P = 100000 #200000000. # Pa
R = 8.3145 # J/molK
T = 1200 + 273.15 # T in kelvin

D_FO_Func = D_Fo(T=T, P=P, fO2=fO2, alpha = 90, beta = 90, gamma = 0, XFo=None, EFo= 201000, R= 8.3145)

# %%
D_FO_Func(XFo=np.array((0.9, 0.7, 0.6, 0.77)))

# %%

# To fit data I should interpolate the data to the dx spacing. I might be able to use Kringing to get error bars. 
# Weighted Residuals. Weights will be inverse of STD^2. If STdevs are the same I can use normal functions. 

# %%



dx_micron = 5
dx = dx_micron * 1e-6 # m
dt =4000 # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt*Di) / (dx**2)
print(CFL)
# delta = (dt)/ ((dx) ** 2)

Total_time =  5 * 24 * 60 * 60    # seconds 
timesteps = 100 #int(Total_time/ dt)

 # %%
pad  = np.ones(3)*0.7
vector_c_in = np.ones(50)*0.8
vector_Fo_in = vector_c_in
D_Fo = D_FO_Func(vector_Fo_in)
kernel_1, kernel_2, delta = diffusion_kernel(dt=dt, dx=dx)

for time in range(timesteps):

    vector_c_in = diffusion_step(vector_c_in = vector_c_in, vector_Fo_in = vector_Fo_in , 
        diffusivity_function = D_FO_Func, diff_kernel_1 = kernel_1, der_kernel_2 = kernel_2, 
        delta= delta, pad_c= pad, pad_Fo = pad) 
    vector_Fo_in = vector_c_in

vector_c_in
# %%

num = len(vector_c_in)
distance =  np.linspace(0, dx_micron*(num), num)
plt.plot(distance, vector_c_in)
plt.xlabel("Micron")
plt.ylabel("Fo")


# Sum of the Residuals^2 Write it so that it evaluates it at certain intervals 
# Maybe divide timesteps into even amounts and then evaluate every N timesteps.

# Also think about 2-D diffusion for profiles with no central plateau. A 2D convolution algorithm would be helpful. 
# %%
# Write Pad function. Each Step needs a pad
# 1) Constant Boundary - Diffusion
# 2) Constant Boundary - No_Diffusion at edge
# 3) Changing Boundary - Ascent Path
# %%

# %%

# %%
vector_c = np.concatenate([pad, vector_c_in, pad])

pad_d = np.ones(3)*1.82955e-16
d = np.ones(50)*9.16947e-17

#vector_D = diffusivity_function(vector_Fo)

vector_D = np.concatenate([pad_d, d,pad_d])

vector_cxD = vector_D * vector_c


Diffusion = np.convolve(kernel_1,vector_c, mode="same")#[3:-3]

delta = (dt) / ((dx) ** 2)
delta*(Diffusion*vector_D)+vector_c


# %%

    vector_c = np.concatenate([pad_c, vector_c_in, pad_c])

    vector_Fo = np.concatenate([pad_Fo, vector_Fo_in, pad_Fo])

    pad_d = np.ones(3)*1.82955e-04
    d = np.ones(50)*9.16947e-05

    vector_D = diffusivity_function(vector_Fo)
    vector_D = np.concatenate([pad_d, d, pad_d])
    vector_cxD = vector_D * vector_c # what is this g

    Diffusion = np.convolve( diff_kernel_1,vector_cxD, mode="same")[3:-3]


    vector_out = vector_c_in + delta*(Diffusion) #+ (Diff_C* Diff_D))

    #vector_c_in + delta*(Diffusion + (Diff_C* Diff_D))

    return vector_out
# %%
