import math, statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

## Functions
def ProductQuotient_Error(F, x, y, x_err, y_err):
    F_err = F * np.sqrt((x_err / x)**2 + (y_err / y)**2)
    return F_err

def SlopeError(x, y, y_pred):
    x_mean = np.average(x)
    error = np.sqrt((np.sum((1/(len(x)-2)) * (y - y_pred)**2) / np.sum((x - x_mean)**2)))
    return error

def Chi2(Observed, Expected):
    return (Observed - Expected)**2 / Expected

def Energy(mean_range):
    return np.exp(6.63 - 3.2376*np.sqrt(10.2146-np.log(mean_range))) #[eV]

#%% Part 1: Plateau Experiment

print("\nPart 1: Plateau Experiment...")

with open('Data/plateau_talium.tsv') as file:
    plat_voltages = []; plat_counts = []; i=0
    for line in file:
        l=line.split('\n')[0]
        l=l.split('\t')
        if i>10:
            plat_voltages.append(int(l[1]))
            plat_counts.append(int(l[2]))
        i += 1

plt.figure()
plt.plot(plat_voltages, plat_counts)
plt.plot(1020, plat_counts[104], marker='.', ms=10)
plt.xlabel("Voltage [V]")
plt.ylabel("Counts")
plt.grid()
#plt.show()

#%% Part 2: Background Measurement & Statistics

print("\nPart 2: Background Measurement & Statistics...")

BG_Voltage = 1020 #[V]
BG_Time = 100 #[sec]
BG_Counts = 46 #[counts]

R_b = BG_Counts / BG_Time #[counts/sec]

print(f"\tBackground Rate: {R_b:0.3f}[counts/sec]")

n_bar_initial = 5.2
m = 150*n_bar_initial #[measurements]

with open('Data/statistics_cobalt.tsv') as file:
    co_voltages = []; co_counts = []; i=0
    for line in file:
        l=line.split('\n')[0]
        l=l.split('\t')
        if i>10:
            co_voltages.append(int(l[1]))
            co_counts.append(int(l[2]))
        i += 1

n_bar = np.mean(co_counts)
std_n = np.std(co_counts)
std_n_bar = std_n/np.sqrt(m-1)

epsilon = (co_counts-n_bar)**3
K_3 = (1/(m-1))*np.sum(epsilon)
var_eps = statistics.variance(epsilon)
var_K_3 = var_eps / (m-1)
std_K_3 = np.sqrt(var_K_3)

X = np.linspace(0,15,16)
P = [900*(n_bar**x)*np.exp(-n_bar)/(math.factorial(x)) for x in X]

print(f"\tAvg. n: {n_bar:0.3f}±{std_n_bar:0.3f}, K3: {K_3:0.3f}±{std_K_3:0.3f}")

plt.figure()
plt.hist(co_counts, bins=20, label='Data')
plt.plot(X, P, label='Poisson Model')
plt.xlabel("Measurements")
plt.ylabel("Counts")
plt.legend()
plt.grid()
#plt.show()

#%% Part 3: Inverse Squared Law

print("\nPart 3: Inverse Squared Law...")

inverse_squared_data = pd.read_csv(f'Data/inverse_squared_thallium.csv', header=0, usecols=['Distance [mm]', 'CPS'])
distances = inverse_squared_data['Distance [mm]']
CPS = inverse_squared_data['CPS']
CPS_corrected = CPS - R_b

y = 1/np.sqrt(CPS_corrected)
res = linregress(distances, y) # linear regression

y_pred = res.slope*distances + res.intercept
R_squared = res.rvalue**2

plt.figure()
plt.scatter(distances, y, label='Data', color='red')
plt.plot(distances, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("Distance [mm]")
plt.ylabel("$1/\sqrt{R-R_b}$")
plt.legend()
plt.grid()

b = 1/(res.slope**2)
a = res.intercept * np.sqrt(b)
a_err = 0
print(f"\ta: {a:0.3f}±{a_err:0.3f}[mm]")

d = distances + a
CPS_pred = b/(d**2)

rms = mean_squared_error(CPS_corrected, CPS_pred, squared=False)
chi2 = sum(Chi2(CPS_pred, CPS_corrected))

plt.figure()
plt.scatter(distances, CPS_corrected, label='Data', color='red')
plt.plot(distances, CPS_pred, label=f'Model, RMSE={rms:.3f}, $\chi^2$={chi2:.3f}')
plt.xlabel("Distance [mm]")
plt.ylabel("Counts Per Second")
plt.legend()
plt.grid()
#plt.show()

#%% Part 4: Alpha Decay

print("\nPart 4: Alpha Decay...")

alpha_data = pd.read_csv(f'Data/alpha_decay_polonium.csv', header=0, usecols=['Distance [mm]', 'CPS'])
distances = alpha_data['Distance [mm]']
CPS = alpha_data['CPS']

dist_corrected = distances + a #[mm]
CPS_corrected = CPS - R_b

res = linregress(dist_corrected, CPS_corrected) # linear regression
CPS_pred = res.slope*dist_corrected + res.intercept
R_squared = res.rvalue**2

tot_range = -res.intercept/res.slope #[mm]
mean_range = 23.57379 #[mm]
energy = ((mean_range/10) + 2.62)/1.24 #[MeV]
chi2 = Chi2(energy, 3.9)

chi2_mean_range = Chi2(mean_range/10, 2.5)
chi2_tot_range = Chi2(tot_range/10, 3.4)
print(chi2_mean_range, chi2_tot_range)

print(f"\tMean Range: {mean_range/10:0.3f}[cm], Total Range: {tot_range/10:0.3f}[cm], Energy: {energy:0.3f}[MeV]")

plt.figure()
plt.scatter(dist_corrected, CPS_corrected/max(CPS_corrected), label='Data', color='red')
plt.plot(dist_corrected, CPS_pred/max(CPS_corrected), label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.axhline(0.5, label = 'Mean Range Threshold', color='grey')
plt.axvline(mean_range, linestyle = '--', label = 'Mean Range', color='grey')
plt.axvline(tot_range, linestyle = '--', label = 'Total Range', color='black')
plt.xlabel("Distance [mm]")
plt.ylabel("Relative Counts Per Second")
plt.legend()
plt.grid()
#plt.show()

#%% Part 5: Beta Decay

print("\nPart 5: Beta Decay...")

d = 27.5 #[mm]

# Source 1: Thallium-204
beta_data_thallium = pd.read_csv(f'Data/beta_decay_thallium.csv', header=0, usecols=['Density [mg/cm^2]', 'Thickness [mm]', 'CPS'])
thicknesses = beta_data_thallium['Thickness [mm]']
densities = beta_data_thallium['Density [mg/cm^2]']
CPS = beta_data_thallium['CPS']
CPS_corrected = CPS - R_b

res = linregress(densities, np.log(CPS_corrected)) # linear regression
ln_CPS_pred = res.slope*densities + res.intercept
R_squared = res.rvalue**2

mu = -res.slope #[cm^2/mg]
CPS_pred = np.exp(res.intercept)*np.exp(-mu*densities)
range = (np.log(R_b) - res.intercept)/(-mu)

energy = np.exp(6.63 - 3.2376*np.sqrt(10.2146 - np.log(range)))

chi2_energy_tl = Chi2(energy, 0.7)
print(chi2_energy_tl*100)

print(f"\tThallium - Mu: {mu:0.3f}[cm^2/mg], Range: {range:0.3f}[mg/cm^2], Energy: {energy:0.3f}[MeV]")

rms = mean_squared_error(CPS_corrected, CPS_pred, squared=False)
chi2 = sum(Chi2(CPS_pred, CPS_corrected))

plt.figure()
plt.scatter(densities, np.log(CPS_corrected), label='Thallium Data', color='red')
plt.plot(densities, ln_CPS_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("Density [mg/cm^2]")
plt.ylabel("ln(CPS)")
plt.legend()
plt.grid()

x = list(densities)
x.append(range)
y = list(CPS_pred)
y.append(R_b)

plt.figure()
plt.scatter(densities, CPS_corrected, label='Thallium Data', color='red')
plt.plot(x, y, label=f'Model, RMSE={rms:.3f}, $\chi^2$={chi2:.3f}')
plt.axhline(R_b, label = 'Background', color='grey', linestyle = '--')
plt.xlabel("Density [mg/cm^2]")
plt.ylabel("Counts Per Second")
plt.legend()
plt.grid()

# Source 2: Strontium-
beta_data_strontium = pd.read_csv(f'Data/beta_decay_strontium.csv', header=0, usecols=['Density [mg/cm^2]', 'Thickness [mm]', 'CPS'])
thicknesses = beta_data_strontium['Thickness [mm]']
densities = beta_data_strontium['Density [mg/cm^2]']
CPS = beta_data_strontium['CPS']
CPS_corrected = CPS - R_b

res = linregress(densities, np.log(CPS_corrected)) # linear regression
ln_CPS_pred = res.slope*densities + res.intercept
R_squared = res.rvalue**2

mu = -res.slope #[cm^2/mg]
CPS_pred = np.exp(res.intercept)*np.exp(-mu*densities)
range = (np.log(R_b) - res.intercept)/(-mu)

energy = np.exp(6.63 - 3.2376*np.sqrt(10.2146 - np.log(range)))

chi2_energy_sr = Chi2(energy, 2.8)
print(chi2_energy_sr*100)

print(f"\tStrontium - Mu: {mu:0.3f}[cm^2/mg], Range: {range:0.3f}[mg/cm^2], Energy: {energy:0.3f}[MeV]")

rms = mean_squared_error(CPS_corrected, CPS_pred, squared=False)
chi2 = sum(Chi2(CPS_pred, CPS_corrected))

plt.figure()
plt.scatter(densities, np.log(CPS_corrected), label='Strontium Data', color='red')
plt.plot(densities, ln_CPS_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("Density [mg/cm^2]")
plt.ylabel("ln(CPS)")
plt.legend()
plt.grid()

x = list(densities)
x.append(range)
y = list(CPS_pred)
y.append(R_b)

plt.figure()
plt.scatter(densities, CPS_corrected, label='Strontium Data', color='red')
plt.plot(x, y, label=f'Model, RMSE={rms:.3f}, $\chi^2$={chi2:.3f}')
plt.axhline(R_b, label = 'Background', color='grey', linestyle = '--')
plt.xlabel("Density [mg/cm^2]")
plt.ylabel("Counts Per Second")
plt.legend()
plt.grid()
plt.show()