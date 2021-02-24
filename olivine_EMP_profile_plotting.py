# %%
# from Users.henry.Python Files.Electrical Conductivity SIMS Data.NS_ConductivityOlivines import Sample_Interpolate
# import Fe_Mg_Diffusion_Convolution
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp
from matplotlib.backends.backend_pdf import PdfPages

import Fe_Mg_Diffusion_Convolution_Streamlined as Ol_Diff

#%%

Feb_excel_path = "Feb 2021_EMP_GCB_version_2_16_20.xlsx"

Ol_Profiles_Feb = pd.read_excel(
    Feb_excel_path,
    sheet_name="Sorted",
    #header=60,
    index_col="DataSet/Point",
    engine="openpyxl",
)
#%%
July_excel_path = "July_2020_EMP_olivine_profiles.xlsx"

Ol_Profiles_July = pd.read_excel(
    July_excel_path,
    sheet_name="EDS Profiles",
    header=1,
    index_col="DataSet/Point",
    engine="openpyxl",
)
# %%
Names_Feb = Ol_Profiles_Feb.Name.unique()
Names_July = Ol_Profiles_July.Name.unique()
# %%


def get_C_prof(prof_name, DF, Element="Fo#", X="Distance µm"):
    prof = DF.loc[(DF.Name == prof_name) & (DF.Bad != "bad")]
    distance_um = prof[X]
    concentration = prof[Element]
    return distance_um.to_numpy(), concentration.to_numpy()


# %%
def plot_prof_trace(
    prof_name,
    DF,
    Element_x="Fo#",
    Element_y="Al2O3",
    ax=None,
    Category=None,
    Distance_Color=True,
):
    if ax is None:
        ax = plt.gca()
    prof = DF.loc[(DF.Name == prof_name) & (DF.Bad != "bad")]

    plt.xlabel(Element_x)
    plt.ylabel(Element_y)

    if Distance_Color is False:
        plt.scatter(
            x=prof[Element_x],
            y=prof[Element_y],
        )
        return ax

    plt.scatter(x=prof[Element_x], y=prof[Element_y], c=prof["Distance µm"])
    return ax


# %%
MP_Overgrowth = Ol_Profiles_Feb.loc[(Ol_Profiles_Feb.Category == "Melt Pocket Overgrowth")]
MP_Names_Feb = MP_Overgrowth.Name.unique()
fig, ax = plt.subplots()
for n in MP_Names_Feb:
    # fig, ax = plt.subplots()
    # plot_prof_trace(n, Ol_Profiles, ax=ax, Element_y="NiO", Distance_Color=False)
    plot_prof_trace(
        n,
        Ol_Profiles_Feb,
        ax=ax,
        Element_y="Fo#",
        Element_x="Distance µm",
        Distance_Color=False,
    )
# plt.ylim(0,0.05)

# %%
Xenocrysts = Ol_Profiles_Feb.loc[(Ol_Profiles_Feb.Category == "Xenocryst")]
Xenocryst_Names_Feb = Xenocrysts.Name.unique()
fig, ax = plt.subplots()
for n in Xenocryst_Names_Feb:
    # fig, ax = plt.subplots()
    # plot_prof_trace(n, Ol_Profiles, ax=ax, Element_y="NiO", Distance_Color=False)
    plot_prof_trace(
        n,
        Ol_Profiles_Feb,
        ax=ax,
        Element_y="NiO",
        Element_x="Distance µm",
        Distance_Color=False,
    )


# %%

# %%
def plot_2_elements(Ol_Profiles, Sample_name, element_1="Fo#", element_2="CaO", ax = None):
    if ax == None:
        ax = plt.gca()
    #fig, ax1 = plt.subplots()
    ax1 = ax
    #plt.title(Sample_name)
    color = "tab:red"
    ax1.set_xlabel("Micron (µm)")
    ax1.set_ylabel(element_1, color=color)

    x_1, y_1 = get_C_prof(Sample_name, Ol_Profiles, Element=element_1)
    ax1.plot(x_1, y_1, color=color, marker="o", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(element_2, color=color)
    x_2, y_2 = get_C_prof(Sample_name, Ol_Profiles, Element=element_2)
    ax2.plot(x_2, y_2, color=color, marker="s", linestyle="dashed")
    ax2.tick_params(axis="y", labelcolor=color)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped

    #return fig, ax1, ax2


# %%
def profile_pdf_plot(Ol_Profiles, Names):
    for n in Names:
        x, y = get_C_prof(n, Ol_Profiles)
        if len(x) > 1:
            with PdfPages(n + " multielement_pdf.pdf") as pdf:
                
                fig = plt.figure(figsize=(8, 11.5), frameon=False,)
                
                Catergory = Ol_Profiles.loc[(Ol_Profiles.Name == n)]['Category'][0]
              
                fig.add_subplot(4,1,1)
                plt.title(n, fontsize = 12,)
                plt.suptitle(Catergory,fontsize = 20, y=.93)

                plot_2_elements(
                    Ol_Profiles,
                    Sample_name=n, element_1="Fo#", element_2="CaO",
                    # ax=ax_a
                )
  
                #ax_b = fig.add_subplot(4,1,2, sharex = ax_a)
                fig.add_subplot(4,1,2)
                plot_2_elements(
                    Ol_Profiles,
                    Sample_name=n, element_1="Al2O3", element_2="P2O5",
                    # ax=ax_b
                )
               

                #ax_c = fig.add_subplot(4,1,3, sharex = ax_a)
                fig.add_subplot(4,1,3)
                plot_2_elements(
                    Ol_Profiles,
                    Sample_name=n, element_1="NiO", element_2="MnO",
                    #ax=ax_c
                )

                # #ax_d = fig.add_subplot(4,1,4, sharex = ax_a)
                # fig.add_subplot(4,1,4)
                # plot_2_elements(
                #     Ol_Profiles,
                #     Sample_name=n, element_1="TiO2", element_2="Cr2O3",
                #     # ax=ax_d
                #)
                fig.subplots_adjust(hspace=0.05)
                #ax_a.get_shared_x_axes().join(ax_a, ax_b, ax_c, ax_d)
                
                pdf.savefig()
                plt.close()
            
        

# %%
profile_pdf_plot(Ol_Profiles_Feb,
    Names_Feb
)
# %%
profile_pdf_plot(Ol_Profiles_July,
    Names_July
)

# %%
"""
Fo# Diffusion Modeling
"""
# %%


x, y = get_C_prof('AZ18 WHT01_bigol1_overgrowth_prof_1', Ol_Profiles_Feb)


dx_micron = 1
dt = 600
step_x = np.arange(0, x.max(), dx_micron)




X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%

#Convert Fo2 from log units with 10 ^(fo2) * 10^5 *1.02

EFo = 201000.0  # J/mol
P = 200000000. # Pa  100000  
R = 8.3145  # J/molK
T_Celsius = 1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin

alpha = 90
beta = 90
gamma = 0

fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, 'FMQ')

Di = Ol_Diff.D_Fo(T, P, fO2, alpha, beta, gamma, XFo=.90, EFo=201000)
# # Check for obeying the CFL Condition


CFL = (dt * Di) / ((dx_micron/1e6) ** 2)
print('Stability Condition: ' + str(CFL))
print('Keep Below 0.5')

#%%
inflection_x = 35
edge_x1 = 20
edge_x2 = 80

edge_c = 0.9215
center_c = 0.8965


# This indexing only works when dx is 1 µm. make more universal 
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 20 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

Z = Ol_Diff.Best_fit_Chi2(Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(2) == 2)[0]
time_days = time/(60*60*24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize = (8,6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min], linewidth = 6 )
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth = 3 )
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
#plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle = 'dashed')
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle = 'dashed')
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle = 'dashed')

plt.xlabel('Distance from Rim µm')
plt.ylabel('Fo#')

plt.title('AZ18 WHT01_bigol1_overgrowth_prof_1')

plt.annotate(f'Temperature: {T_Celsius} ˚C', xy = (.74, 0.95), xycoords = 'axes fraction')
plt.annotate(f'Bestfit time: {round(time_days,1)} days', xy = (.74, 0.9), xycoords = 'axes fraction')




# %%
# %%










# %%
"""
New Model!
"""


x, y = get_C_prof('AZ18 WHT01_bigol2_overgrowth_prof_1', Ol_Profiles_Feb)


dx_micron = 1
dt = 400
step_x = np.arange(0, x.max(), dx_micron)




X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%


# Di = D_FO_Func(0.8)
# # Check for obeying the CFL Condition
# CFL = (dt * Di) / (dx ** 2)
# print(CFL)


EFo = 201000.0  # J/mol
P = 200000000. # Pa
R = 8.3145  # J/molK
T_Celsius = 1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin
fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, 'FMQ')

inflection_x = 15
edge_x1 = 0
edge_x2 = 45

edge_c = 0.9150
center_c = 0.897

alpha = 90
beta = 90
gamma = 0

# This indexing only works when dx is 1 µm. make more universal 
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 100 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

Z = Ol_Diff.Best_fit_Chi2(Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(1) == 2)[0]
time_days = time/(60*60*24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize = (8,6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min], linewidth = 6 )
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth = 3 )
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
#plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle = 'dashed')
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle = 'dashed')
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle = 'dashed')

plt.xlabel('Distance from Rim µm')
plt.ylabel('Fo#')

plt.title('AZ18 WHT01_bigol1_overgrowth_prof_1')

plt.annotate(f'Temperature: {T_Celsius} ˚C', xy = (.74, 0.95), xycoords = 'axes fraction')
plt.annotate(f'Bestfit time: {round(time_days,1)} days', xy = (.74, 0.9), xycoords = 'axes fraction')


# %%


# %%
"""
Melt Pocket Adjacent Olivine Overgrowth WHT06. 
"""

x, y = get_C_prof('AZ18 WHT06 ol_overgrowth1 prof 1', Ol_Profiles_July)


dx_micron = 1
dt = 600
step_x = np.arange(0, x.max(), dx_micron)




X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%

#Convert Fo2 from log units with 10 ^(fo2) * 10^5 *1.02

EFo = 201000.0  # J/mol
P = 2000000000. # Pa  100000  
R = 8.3145  # J/molK
T_Celsius = 1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin

alpha = 0
beta = 90
gamma = 90

fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, 'FMQ')

Di = Ol_Diff.D_Fo(T, P, fO2, alpha, beta, gamma, XFo=.90, EFo=201000)
# # Check for obeying the CFL Condition


CFL = (dt * Di) / ((dx_micron/1e6) ** 2)
print('Stability Condition: ' + str(CFL))
print('Keep Below 0.5')

#%%
inflection_x = 18
edge_x1 = 0
edge_x2 = 35

edge_c = 0.9325
center_c = 0.907


# This indexing only works when dx is 1 µm. make more universal 
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 20 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

Z = Ol_Diff.Best_fit_Chi2(Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(2) == 2)[0]
time_days = time/(60*60*24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize = (8,6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min], linewidth = 6 )
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth = 3 )
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
#plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle = 'dashed')
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle = 'dashed')
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle = 'dashed')

plt.xlabel('Distance from Rim µm')
plt.ylabel('Fo#')

plt.title('AZ18 WHT06 MP Adjacent Ol overgrowth')

plt.annotate(f'Temperature: {T_Celsius} ˚C', xy = (.74, 0.95), xycoords = 'axes fraction')
plt.annotate(f'Bestfit time: {round(time_days,1)} days', xy = (.74, 0.9), xycoords = 'axes fraction')
# %%



# %%


"""
GCB Xenocryst Scoria Ol_6 Overgrowth  
"""

x, y = get_C_prof('Xenocryst_ol6_prof_2', Ol_Profiles_Feb)


dx_micron = 1
dt = 600
step_x = np.arange(0, x.max(), dx_micron)




X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%

#Convert Fo2 from log units with 10 ^(fo2) * 10^5 *1.02

EFo = 201000.0  # J/mol
P = 200000000. # Pa  100000  
R = 8.3145  # J/molK
T_Celsius = 1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin

alpha = 90
beta = 90
gamma = 0

fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, 'FMQ')

Di = Ol_Diff.D_Fo(T, P, fO2, alpha, beta, gamma, XFo=.90, EFo=201000)
# # Check for obeying the CFL Condition


CFL = (dt * Di) / ((dx_micron/1e6) ** 2)
print('Stability Condition: ' + str(CFL))
print('Keep Below 0.5')

#%%
inflection_x = 25
edge_x1 = 12
edge_x2 = 54

edge_c = 0.836
center_c = 0.905


# This indexing only works when dx is 1 µm. make more universal 
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 20 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

Z = Ol_Diff.Best_fit_Chi2(Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(2) == 2)[0]
time_days = time/(60*60*24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize = (8,6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min], linewidth = 6 )
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth = 3 )
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
#plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle = 'dashed')
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle = 'dashed')
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle = 'dashed')

plt.xlabel('Distance from Rim µm')
plt.ylabel('Fo#')

plt.title('Xenocryst S_Ol_6 Overgrowth Profile 2')

plt.annotate(f'Temperature: {T_Celsius} ˚C', xy = (.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f'Bestfit time: {round(time_days,1)} days', xy = (.01, 0.9), xycoords = 'axes fraction')
# %%


"""
GCB Xenocryst Scoria ol43 AZ18 WHT01  
"""

x, y = get_C_prof('AZ18 WHT06 ol43 prof 1', Ol_Profiles_July)
x = x[1:-1]
y = y[1:-1]

dx_micron = 1
dt = 600
step_x = np.arange(0, x.max(), dx_micron)



#%%
X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%

#Convert Fo2 from log units with 10 ^(fo2) * 10^5 *1.02

EFo = 201000.0  # J/mol
P = 200000000. # Pa  100000  
R = 8.3145  # J/molK
T_Celsius = 1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin

alpha = 0
beta = 90
gamma = 90

fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, 'FMQ')

Di = Ol_Diff.D_Fo(T, P, fO2, alpha, beta, gamma, XFo=.90, EFo=201000)
# # Check for obeying the CFL Condition


CFL = (dt * Di) / ((dx_micron/1e6) ** 2)
print('Stability Condition: ' + str(CFL))
print('Keep Below 0.5')

#%%
inflection_x = 37
edge_x1 = 20
edge_x2 = 120

edge_c = 0.851
center_c = 0.895


# This indexing only works when dx is 1 µm. make more universal 
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 20 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

Z = Ol_Diff.Best_fit_Chi2(Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(2) == 2)[0]
time_days = time/(60*60*24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize = (8,6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min], linewidth = 6 )
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth = 3 )
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
#plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle = 'dashed')
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle = 'dashed')
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle = 'dashed')

plt.xlabel('Distance from Rim µm')
plt.ylabel('Fo#')

plt.title('Xenocryst AZ18_WHT06 Ol_43 Overgrowth')

plt.annotate(f'Temperature: {T_Celsius} ˚C', xy = (.01, 0.95), xycoords = 'axes fraction')
plt.annotate(f'Bestfit time: {round(time_days,1)} days', xy = (.01, 0.9), xycoords = 'axes fraction')

# %%

# %%
