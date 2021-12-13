#%%
import numpy as np
import matplotlib.pyplot as plt
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


def get_tau(nu, D, k, delta_rho):
    # time in seconds
    g = -9.81  # ms-2
    tau = nu / (D * k ** 4 + delta_rho * g)
    return np.abs(tau)


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
L_mantle = 10 ** 6  # mantle length (?)
delta_rho = 3300 - 900  # ...
llambda = 4000 * 1000
k = 2 * np.pi / (llambda)  #  llambda
y2s = 365 * 24 * 60 * 60  # years to seconds

# get initial deflection profile:
x, w0 = init_deflection(q0, x_max, D, delta_rho, k)

# find relaxation time:
nu = mu / L_mantle
tau = get_tau(nu, D, k, delta_rho)
half_life = np.log(2) * tau / 60 / 60 / 24 / 365.25
print(f"{half_life=}")

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

# %%
youngs_mod = 7e10  # youngs modulus [Pa]
poissons_ratio = 0.25  # poisson's ratio
elastic_thickness = 30 * 1000  # [m]
flexural_rigidity = (youngs_mod * (elastic_thickness ** 3)) / (
    12 * (1 - (poissons_ratio ** 2))
)


llambda = 900 * 1000
k = 2 * np.pi / (llambda)
w0 = 200  # initial height guess
mus = [1e19, 5e19, 1e20, 2e20]
L_mantle = 1e2 * 1000  # mantle length (?)
nus = [mu / L_mantle for mu in mus]
taus = [get_tau(nu, flexural_rigidity, k, delta_rho) for nu in nus]
# taus = [ts_tau(nu, llambda) for nu in nus]
t0 = 14000  # time since uplift
ages_since = t0 - ages

times = np.arange(0, t0, 100) * y2s
fig, ax = plt.subplots(figsize=(14, 7))
plt.scatter(ages_since, msl, 5)
# plt.scatter(times_since, elevations)

for ii, tau in enumerate(taus):
    half_life = np.log(2) * tau / 60 / 60 / 24 / 365.25
    print(f"{half_life=}")
    plt.plot(
        times / y2s, w0 * np.exp(-times / tau), label=rf"$\mu$: {mus[ii]}", alpha=0.7
    )
plt.xlabel("Time since uplift [yr]")
plt.ylabel("Elevation")
plt.title("Mainland Vancouver paleo_RSL")
plt.legend()
plt.grid()
# %%
# %% data entry
times_ago = np.array([14200, 14000, 13900, 13300, 12650, 12600, 12300, 1500])
t_0 = 14200  # start of uplift
times_since = t_0 - times_ago
elevations = np.array([197, 175, 144, 75, 26, 14, 7, 0.75])

fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(times_since, elevations)
for ii, tau in enumerate(taus):
    plt.plot(times / y2s, w0 * np.exp(-times / tau), label=rf"$\mu$: {mus[ii]}")
plt.title("N. Georgia Strait (cores only) paleo-RSL")
plt.legend()
plt.grid()
plt.xlabel("Time since uplift [yr]")
plt.ylabel("Elevation")

half_life = np.log(2) * taus[1] / 60 / 60 / 24 / 365.25
print(f"{half_life=}")


# %%
