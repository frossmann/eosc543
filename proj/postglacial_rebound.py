#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy import core
import pandas as pd


def init_deflection(q0, x_max, D, delta_rho, k):
    """for a an axisymmetric ice sheet blob"""
    g = -9.81  # ms-2
    x = np.arange(0, x_max, 1000)  # 0 to x_max in steps of 1 km in m
    w0 = q0 * np.cos(k * x) / (D * (k ** 4) + delta_rho * g)
    return x, w0


def update_deflection(time, w_old, tau):
    w_new = w_old * np.exp(-time / tau)
    return w_new


def ts_tau(mu, llambda):
    g = 9.81  # ms-2
    rho_m = 3300  # kgm-3
    tau = (4 * np.pi * mu) / (rho_m * g * llambda)
    return tau


def rebound_relax():
    pass


#%% Plot an example rebound problem for a 4km high periodic load:
# preliminaries:
q0 = 4000 * 9.81 * 900  # 4km high in m
x_max = 1400 * 1000  # 1000km halfspace in m
D = 5e23  # flexural rigidity
mu = 5e21  # viscosity
delta_rho = 3300 - 900  # ...
llambda = 4000 * 1000
k = 2 * np.pi / (llambda)  #  llambda
y2s = 365 * 24 * 60 * 60  # years to seconds

# get initial deflection profile:
x, w0 = init_deflection(q0, x_max, D, delta_rho, k)

# find relaxation time:
tau = ts_tau(mu, llambda)
half_life = np.log(2) * tau / 60 / 60 / 24 / 365.25
print(f"half life: {half_life} years")

# initialize some times
times = np.array([0, 1000, 2000, 5000, 10000]) * 365 * 24 * 60 * 60

# make a figure
fig = plt.figure(figsize=(12, 8))
plt.plot(x / 1000, w0)
for time in times:
    w_new = update_deflection(time, w0, tau)
    plt.plot(x / 1000, w_new, label=str(round(time / y2s / 1000)) + " kyr")

plt.xlabel("Distance [km]")
plt.ylabel("Elevation [m]")
plt.grid()
plt.legend()
plt.title("")


#%%
filename = "/Users/francis/Desktop/e543_proj/1-s2.0-S0277379114002030-mmc1.xlsx"
df = pd.read_excel(filename, index_col=24)  # index by regionls
mainland = df[df.index.str.startswith("Lower")]
ages = mainland["Median calibrated age"].to_numpy()
msl = mainland["MSL (m)"].to_numpy()

# %% MAINLAND VANCOUVER

llambda = 900 * 1000
k = 2 * np.pi / (llambda)
w0 = 200  # initial height guess
mus = [1e19, 5e19, 1e20, 2e20]
taus = [ts_tau(mu, llambda) for mu in mus]
t0 = 14000  # time since uplift
ages_since = t0 - ages

times = np.arange(0, t0, 100) * y2s
fig, ax = plt.subplots(figsize=(14, 7))
plt.scatter(ages_since, msl, 5)
# plt.scatter(times_since, elevations)

for ii, tau in enumerate(taus):
    half_life = np.log(2) * tau / 60 / 60 / 24 / 365.25
    print(f"half life: {half_life} years")
    plt.plot(
        times / y2s, w0 * np.exp(-times / tau), label=rf"$\mu$: {mus[ii]}", alpha=0.7
    )
plt.xlabel("Time since uplift [yr]")
plt.ylabel("Elevation")
plt.title("Lower Fraser Valley GIA")
plt.legend()
plt.grid()


#%% NORTH STRAIGHT OF GEORGIA
nsog = df[df.index.str.startswith("North Strait of Georgia")]

# drop Cortez bay measurements (outliers here)
# nsog = nsog_full[nsog_full["Site"].str.startswith("Cortes Bay")==False]

ages = nsog["Median calibrated age"].to_numpy()
msl = nsog["MSL (m)"].to_numpy()

# Add core observations from Fedje (2018)
core_times = np.array([14200, 14000, 13900, 13300, 12650, 12600, 12300, 1500])
t_0 = 14200  # start of uplift
core_times_since = t_0 - core_times
core_elevations = np.array([197, 175, 144, 75, 26, 14, 7, 0.75])


llambda = 900 * 1000
k = 2 * np.pi / (llambda)
w0 = 200  # initial height guess
mus = [1e19, 5e19, 1e20, 2e20]
taus = [ts_tau(mu, llambda) for mu in mus]
t0 = 14000  # time since uplift
ages_since = t0 - ages

times = np.arange(0, t0, 100) * y2s
fig, ax = plt.subplots(figsize=(14, 7))
plt.scatter(ages_since[np.where(ages_since > 0)], msl[np.where(ages_since > 0)], 8)
plt.scatter(core_times_since, core_elevations, 8)
# plt.scatter(times_since, elevations)

for ii, tau in enumerate(taus):
    half_life = np.log(2) * tau / 60 / 60 / 24 / 365.25
    print(f"half life: {half_life} years")
    plt.plot(
        times / y2s, w0 * np.exp(-times / tau), label=rf"$\mu$: {mus[ii]}", alpha=0.7
    )
plt.xlabel("Time since uplift [yr]")
plt.ylabel("Elevation")
plt.title("Northern Strait of Georgia GIA")
plt.legend()
plt.grid()
# %% NORTHERN STRAIGHT OF GEORGIA (cores only)

# fig, ax = plt.subplots(figsize=(12, 6))
# plt.scatter(times_since, elevations)
# for ii, tau in enumerate(taus):
#     plt.plot(times / y2s, w0 * np.exp(-times / tau), label=rf"$\mu$: {mus[ii]}")
# plt.title("N. Georgia Strait (cores only) GIA")
# plt.legend()
# plt.grid()
# plt.xlabel("Time since uplift [yr]")
# plt.ylabel("Elevation")

# half_life = np.log(2) * taus[1] / 60 / 60 / 24 / 365.25
# print(f"hl: {half_life}")

#%%


# %%
from scipy import interpolate

barbados = pd.read_excel(
    "/Users/francis/repos/eosc543/proj/extended_Barbados_sea_level_record.xlsx"
)
rsl = np.array([depth for depth in -barbados["Depth(m) uplift corrected"]])
rsl_time = np.array([age for age in barbados["Mean Th/U Age yr BP"]])

iloc = np.where(rsl_time < 16000)

rsl_crop = rsl[iloc]
rsl_time_crop = rsl_time[iloc]


# spl = interpolate.Rbf(rsl_time_crop[np.argsort(rsl_time_crop)], rsl_crop[np.argsort(rsl_time_crop)])
# spl.set_smoothing_factor(0.1)
# xnew = np.linspace(rsl_time_crop.min(), rsl_time_crop.max(), num=100, endpoint=True)

plt.figure()
plt.scatter(rsl_time_crop, rsl_crop)
# plt.plot(xnew, spl(xnew))


# %%
