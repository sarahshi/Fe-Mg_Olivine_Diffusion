# %%
# from Users.henry.Python Files.Electrical Conductivity SIMS Data.NS_ConductivityOlivines import Sample_Interpolate
# import Fe_Mg_Diffusion_Convolution
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp
from matplotlib.backends.backend_pdf import PdfPages


from pykrige import OrdinaryKriging

# %%
import Fe_Mg_Diffusion_Convolution as Ol_Diff

#%%

excel_path = "Feb 2021_EMP_GCB_version_2_16_20.xlsx"

Ol_Profiles = pd.read_excel(
    excel_path,
    sheet_name="Sorted",
    #header=60,
    index_col="DataSet/Point",
    engine="openpyxl",
)
# %%
Names = Ol_Profiles.Name.unique()
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
MP_Overgrowth = Ol_Profiles.loc[(Ol_Profiles.Category == "Melt Pocket Overgrowth")]
MP_Names = MP_Overgrowth.Name.unique()
fig, ax = plt.subplots()
for n in MP_Names:
    # fig, ax = plt.subplots()
    # plot_prof_trace(n, Ol_Profiles, ax=ax, Element_y="NiO", Distance_Color=False)
    plot_prof_trace(
        n,
        Ol_Profiles,
        ax=ax,
        Element_y="Fo#",
        Element_x="Distance µm",
        Distance_Color=False,
    )
# plt.ylim(0,0.05)

# %%
Xenocrysts = Ol_Profiles.loc[(Ol_Profiles.Category == "Xenocryst")]
Xenocryst_Names = Xenocrysts.Name.unique()
fig, ax = plt.subplots()
for n in Xenocryst_Names:
    # fig, ax = plt.subplots()
    # plot_prof_trace(n, Ol_Profiles, ax=ax, Element_y="NiO", Distance_Color=False)
    plot_prof_trace(
        n,
        Ol_Profiles,
        ax=ax,
        Element_y="NiO",
        Element_x="Distance µm",
        Distance_Color=False,
    )
# plt.ylim(0,0.05)

# %%
x, y = get_C_prof("GCB1A_S_ol2_prof1_1_", Ol_Profiles)
# %%
for n in Names:
    x, y = get_C_prof(n, Ol_Profiles)
    if len(x) > 1:
        fig, ax = plt.subplots()
        plt.plot(x, y, marker="o", linestyle="dashed")
        plt.title(n)


# %%
for n in Names:
    x, y = get_C_prof(n, Ol_Profiles, Element="CaO")
    if len(x) > 1:
        fig, ax = plt.subplots()
        plt.plot(x, y, marker="o", linestyle="dashed")
        plt.title(n)

# %%
for n in Names:
    x, y = get_C_prof(n, Ol_Profiles, Element="Al2O3", X="Fo#")
    if len(x) > 1:
        fig, ax = plt.subplots()
        plt.plot(x, y, marker="o", linestyle="dashed")
        plt.title(n)
# %%
def plot_2_elements(Sample_name, element_1="Fo#", element_2="CaO"):

    fig, ax1 = plt.subplots()
    plt.title(Sample_name)
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

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig, ax1, ax2


# %%

for n in Names:
    x, y = get_C_prof(n, Ol_Profiles)
    if len(x) > 1:
        with PdfPages(n + " multielement_pdf.pdf") as pdf:
            fig, ax1, ax2 = plot_2_elements(
                Sample_name=n, element_1="Fo#", element_2="CaO"
            )
            pdf.savefig()
            plt.close()

            fig, ax1, ax2 = plot_2_elements(
                Sample_name=n, element_1="Al2O3", element_2="P2O5"
            )
            pdf.savefig()
            plt.close()

            fig, ax1, ax2 = plot_2_elements(
                Sample_name=n, element_1="NiO", element_2="MnO"
            )
            pdf.savefig()
            plt.close()

# %%


# %%
"""
Fo# Diffusion Modeling
"""
# %%


x, y = get_C_prof('AZ18 WHT01_bigol1_overgrowth_prof_1', Ol_Profiles)
dx = 1
step_x = np.arange(0, x.max(), dx)

# %%


X_interp, Y_interp, Y_interp_std = Krige_Interpolate(
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

#%%
dx_micron = 2.5
dx = dx_micron * 1e-6  # m
dt = 4000  # 100000
Di = D_FO_Func(0.8)
# Check for obeying the CFL Condition
CFL = (dt * Di) / (dx ** 2)
print(CFL)
# delta = (dt)/ ((dx) ** 2)

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