#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% QUESTION 1
# preliminaries:
y_l = 110 * 1000  # m
y_c = 31 * 1000  # m
Tm = 1333  # °C
K = 10 ** (-6)  # m2s-1
alpha = 3.3e-5  # (°C-1)
rho_m = 3300  # kg/m3
rho_s = 2100  # kg/m3
rho_w = 1035  # kg/m3

times = -np.array([100, 65, 55, 20, 0])  # Myr ago?
elevations = -np.array([0.217, -1.031, -1.251, -1.704, -1.854]) * 1000  # m
# %% a)
fig = plt.figure(figsize=(14, 7))
plt.plot(times, elevations, linewidth=3)
plt.grid()
plt.ylabel("Subsidence [m]")
plt.xlabel("Myr from present")
plt.title("Table 1 Elevation vs. Time")
# %% b)
# calculate the thermal time constant:
tau = (y_l ** 2) / (np.pi ** 2 * K)  # seconds...
tau_myr = tau / 60 / 60 / 24 / 365 / 1e6
print(f"Thermal time constant: {tau_myr:.1f} Myr")

# %% c)
#  plot subsidence against scaled time constant:
etime = 1 - np.exp(-times / tau_myr)  # dimensionless [Myr/Myr]

# make the plot:
plt.figure(figsize=(14, 7))
plt.plot(etime, elevations, linewidth=3)
plt.ylabel("Subsidence [m]")
plt.xlabel(r"$1-e^{-\frac{t}{\tau}}$")
plt.title("Subsidence vs. scaled time")
plt.grid()


# %%
# fit a line with linear least squares (simple)
A = np.vstack([etime, np.ones(len(etime))]).T
m, c = np.linalg.lstsq(A, elevations, rcond=None)[0]
print(f"The slope of the linear regression is {m}")

# make the plot:
plt.figure(figsize=(14, 7))
plt.plot(
    etime, elevations, label="Data", linewidth=3, markersize=20, marker=".", alpha=0.7
)
plt.plot(etime, m * etime + c, "r", label="Fit", linestyle="--", alpha=0.9)
plt.legend()
plt.ylabel("Subsidence [m]")
plt.xlabel(r"$1-e^{-\frac{t}{\tau}}$")
plt.title("Subsidence vs. scaled time")
plt.grid()


# %%
# #find E0 and beta...
E0_w = (4 * y_l * rho_m * alpha * Tm) / (np.pi ** 2 * (rho_m - rho_w))  # for water
E0_s = (4 * y_l * rho_m * alpha * Tm) / (np.pi ** 2 * (rho_m - rho_s))  # for seds


dummy_betas = np.arange(1, 2, 0.001)


def get_slopes(E0, betas):
    """slope is equal to E0*(beta/pi)*sin(pi/beta) (from Allen&Allen pg. 511)
    we know slope and beta so shotgun some slopes for different betas"""
    slopes = [E0 * (beta / np.pi) * np.sin(np.pi / beta) for beta in betas]
    return slopes


slopes_w = get_slopes(E0_w, dummy_betas)
slopes_s = get_slopes(E0_s, dummy_betas)
beta_w = dummy_betas[np.isclose(slopes_w, m, rtol=0.01)]
beta_s = dummy_betas[np.isclose(slopes_s, m, rtol=0.016)]
print(f"Water filled basin:\n \tBeta = {beta_w[0]:.3}\n \tE0 = {E0_w:.2}")
print(f"Seds filled basin:\n \tBeta = {beta_s[0]:.3}\n \tE0 = {E0_s:.2}")


fig = plt.figure(figsize=(14, 7))
plt.plot(dummy_betas, slopes_w, color="brown", label="water fill")
plt.plot(dummy_betas, slopes_s, color="g", label="seds fill")
plt.xlabel(r"$\beta$")
plt.ylabel("Slope")
xl = plt.xlim()
plt.hlines(m, xl[0], xl[1], color="k", linestyle="--", alpha=0.5)
yl = plt.ylim()
plt.vlines(beta_w, yl[0], yl[1], color="k", linestyle="--", alpha=0.5)
plt.vlines(beta_s, yl[0], yl[1], color="k", linestyle="--", alpha=0.5)
plt.title("Slope vs beta values")
plt.grid()
plt.plot(beta_w, m, marker="x", color="brown", markersize=10, label="beta_w")
plt.plot(beta_s, m, marker="x", color="g", markersize=10, label="beta_s")
plt.legend()
plt.text(1.1, 1400, r"$\beta_{w}$ = " + f"{beta_w[0]:.3}", fontsize=12)
plt.text(1.1, 1200, r"$\beta_{s}$ = " + f"{beta_s[0]:.3}", fontsize=12)


# %% QUESTION 2:
# preliminaries:
# set the constants for each lithology:
shale = {}
shale["c"] = 0.0005  # constant
shale["phi"] = 0.5  # porosity
shale["rho"] = 2720  # kgm-3

sand = {}
sand["c"] = 0.0003  # constant
sand["phi"] = 0.4  # porosity
sand["rho"] = 2650  # kgm-3

lime = {}
lime["c"] = 0.0007  # constant
lime["phi"] = 0.5  # porosity
lime["rho"] = 2710  # kgm-3

# pack all the lithologies in a dictionary:
vars = {}
vars["shale"] = shale
vars["sand"] = sand
vars["lime"] = lime
vars["none"] = None


# set up the problem:
depths = np.array(
    [0, 1000, 1800, 1900, 1900, 2800, 2900, 5100]
)  # top down depths to edges
heights = np.array([1000, 800, 100, 0, 900, 100, 2200])  # top down thickneses
midpoints = depths[1:] - heights / 2
liths = [
    "lime",
    "shale",
    "sand",
    "none",
    "lime",
    "shale",
    "sand",
]  # top down lithologies

#%% decompaction:

# 1) calculate in situ compacted porosities:
phis = np.zeros_like(heights, dtype=object)
for ii in range(len(liths)):
    if heights[ii] > 0:
        phi_0 = vars[liths[ii]]["phi"] * np.exp(-vars[liths[ii]]["c"] * midpoints[ii])
    else:
        phi_0 = 0
    phis[ii] = phi_0


def update_phi(consts, midpoint):
    """Updates porosity given a change in burial midpoint"""
    phi_new = consts["phi"] * np.exp(-consts["c"] * midpoint)
    return phi_new


def update_T(height, phi_old, phi_new):
    """Updates the thickness of a layer given a change in porosity"""
    T_new = ((1 - phi_old) * height) / (1 - phi_new)
    return T_new


#%%
# # want to get 41...
# phi_old = phis[1]
# phi_new = update_phi(vars[liths[1]], heights[1] / 2)
# T_new = update_T(heights[1], phi_old, phi_new)

# # now find the porosity of the block underneath...
# phi_old = phis[2]
# phi_new = update_phi(vars[liths[2]], T_new + heights[2] / 2)
# T_new = update_T(heights[2], phi_old, phi_new)

phi_array = np.zeros(shape=(7, 7))
T_array = np.zeros(shape=(7, 7))
phi_array[:, 0] = phis.T
T_array[:, 0] = heights.T

# loop over columns
for ii in range(1, 7):
    # then loop over rows:
    for jj in range(1, 7):
        # default to zero porosity and height at the unconfomity:
        if np.all(T_array[jj, :] == 0):
            phi_array[jj, ii] = 0
            T_array[jj, ii] = 0
        # skip the blocks we've removed:
        elif jj < ii:
            phi_array[jj, ii] = 0
            T_array[jj, ii] = 0
        # remove the top block and use midpoint for porosity calculation:
        elif jj == ii:
            phi_array[jj, ii] = update_phi(vars[liths[jj]], T_array[jj, ii - 1] / 2)
            T_array[jj, ii] = update_T(
                T_array[jj, ii - 1], phi_array[jj, ii - 1], phi_array[jj, ii]
            )
        # solve for underlying blocks, using adjusted midpoints:
        else:
            phi_array[jj, ii] = update_phi(
                vars[liths[jj]],
                np.sum(T_array[:, ii][ii - 1 : jj]) + T_array[jj, ii - 1] / 2,
            )
            T_array[jj, ii] = update_T(
                T_array[jj, ii - 1], phi_array[jj, ii - 1], phi_array[jj, ii]
            )


decompacted_depths = np.append(np.sum(T_array, axis=0), 0)[::-1]

T_table = pd.DataFrame(T_array)
phi_table = pd.DataFrame(phi_array)
# %%
# make a subsidence curve...
ages = [0, 10, 20, 30, 50, 60, 70, 75][::-1]  # Ma

fig = plt.figure(figsize=(14, 7))
ax = plt.subplot()
plt.plot(ages, depths, label="Compacted")
plt.plot(ages, decompacted_depths, label="Decompacted")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.grid()
plt.ylabel("Thickness [m]")
ax.xaxis.set_ticks_position("top")
ax.set_xlabel("Time [Ma]")
ax.xaxis.set_label_position("top")
plt.title("Subsidence curve")
plt.legend()
# %%
