import math, statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

## Functions
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts #running average
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def ProductQuotient_Error(F, x, y, x_err, y_err):
    F_err = F * np.sqrt((x_err / x)**2 + (y_err / y)**2)
    return F_err

def SlopeError(x, y, y_pred):
    x_mean = np.average(x)
    error = np.sqrt((np.sum((1/(len(x)-2)) * (y - y_pred)**2) / np.sum((x - x_mean)**2)))
    return error

def Chi2(Observed, Expected):
    return (Observed - Expected)**2 / Expected

def Channel2Energy(ch, a, b):
    return a*ch + b

def Energy2Wavelength(energy):
    h = 4.135667696e-15 #[eV sec]
    return h*scipy.constants.c/energy

def Compton(angle):
    E = 17479.3 #[eV]
    lambda_e = 2.42631023867e-12 #[m]
    lambd = Energy2Wavelength(E) #[m]
    return lambd + lambda_e*(1-np.cos(np.deg2rad(angle)))

def Klein_Nishina(angle, energy):
    E = 17479.3 #[eV]
    lambda_over_lambda_tag = np.array(energy)/E
    r_e = 2.8179403262e-15 #[m]
    return 0.5 * (r_e**2) * (lambda_over_lambda_tag**2) * (lambda_over_lambda_tag + (1/lambda_over_lambda_tag) - (np.sin(np.deg2rad(angle))**2))

#%% Part 1: Background Spectrum (Molybdenum + Copper)

print("\nPart 1: Background Spectrum...")

BG_Data = pd.read_csv('Data/Background.txt', sep='\t', header=1) # Read the data
Counts = np.array(BG_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # Smooth the data over 10 channels
Channels = np.array(BG_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=10)
correct_peaks = [peaks[0], peaks[2], peaks[3]]

x = correct_peaks
y = [8047.8, 17479.3, 19608.3] # Peak 1 = Cu (alpha), Peak 2 = Mo (alpha), Peak 3 = Mo (beta)

X_Vec = []; Y_Vec = []
for val in x:
    X_Vec.append(val)
for val in y:
    Y_Vec.append(val)

plt.figure()
plt.plot(Channels, Counts_smoothed, label='Background Spectrum')
plt.plot(Channels[correct_peaks], Counts_smoothed[correct_peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.grid()
plt.legend()

#%% Part 2: Descloizite Spectrum

print("\nPart 2: Descloizite Spectrum...")

Desc_Data = pd.read_csv('Data/Descloizite.txt', sep='\t', header=1) # Read the data
Counts = np.array(Desc_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # Smooth the data over 10 channels
Channels = np.array(Desc_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=10, prominence=5)
correct_peaks = [peaks[3], peaks[5], peaks[6], peaks[7]]

x = correct_peaks
y = [8638.9, 10551.5, 12613.7, 14764.4] # Peak 1 = Zn (alpha), Peak 2 = Pb (alpha), Peak 3 = Pb (beta), Peak 4 = Pb (gamma)

for val in x:
    X_Vec.append(val)
for val in y:
    Y_Vec.append(val)

X_Vec = np.array(X_Vec)
Y_Vec = np.array(Y_Vec)

reg = linregress(X_Vec, Y_Vec)
A = reg.slope
B = reg.intercept
R_squared = reg.rvalue**2

plt.figure()
plt.plot(Channels, Counts_smoothed, label='Descloizite Spectrum')
plt.plot(Channels[correct_peaks], Counts_smoothed[correct_peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.grid()
plt.legend()

plt.figure()
plt.scatter(X_Vec, Y_Vec/1000, label='Data', color='red')
plt.plot(X_Vec, Channel2Energy(X_Vec, A, B)/1000, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("Channels")
plt.ylabel("Energy [keV]")
plt.grid()
plt.legend()

#%% Part 3: Scattering

print("\nPart 3: Scattering...")

energy_vec = []; intensity_vec = []
angles = np.linspace(20, 150, 14)
for a in angles:
    scatter_data = pd.read_csv(f'Data/Scattering_Angle={int(a)}.txt', sep='\t', header=1) # read the data

    Counts = np.array(scatter_data['Impulses/#']) # Impulses
    Counts_smoothed = smooth(Counts, 10) # Smooth the data over 10 channels
    Channels = np.array(scatter_data['Channel/#']) # Channel

    Energies = Channel2Energy(Channels, A, B)

    peaks, _ = find_peaks(Counts_smoothed, height=80, prominence=20)
    energy = Channel2Energy(peaks[0], A, B)
    intensity = Counts_smoothed[peaks[0]]

    energy_vec.append(energy)
    intensity_vec.append(intensity)

    """
    plt.figure()
    plt.plot(Energies/1000, Counts_smoothed, label='Scattering Spectrum')
    plt.plot(Energies[peaks]/1000, Counts_smoothed[peaks], "x", color='red', label='Peaks')
    plt.ylabel('Impulses')
    plt.xlabel('Energy [keV]')
    plt.grid()
    plt.legend()
    """

lambda_tag = Energy2Wavelength(np.array(energy_vec))
lambda_pred = Compton(angles)

rms = mean_squared_error(lambda_tag, lambda_pred, squared=False)*1e12
chi2 = sum(Chi2(lambda_pred, lambda_tag))*1e12

plt.figure()
plt.plot(angles, np.array(energy_vec)/1000)
plt.xlabel('Angle [deg]')
plt.ylabel('Energy [keV]')
plt.grid()

plt.figure()
plt.scatter(angles, lambda_tag*1e12, label='Data', color='red')
plt.plot(angles, lambda_pred*1e12, label=f'Compton Model, RMSE={rms:.3f}, $\chi^2$={chi2:.3f}')
plt.xlabel('Angle [deg]')
plt.ylabel('Wavelength [nm]')
plt.grid()
plt.legend()

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

res_pred = Klein_Nishina(angles, energy_vec)

def func(x, a):
    return a*1e30*x

popt, _ = curve_fit(func, res_pred, intensity_vec)

C = popt[0]*1e30
I_pred = C*res_pred

rms = mean_squared_error(intensity_vec, I_pred, squared=False)
chi2 = sum(Chi2(I_pred, intensity_vec))

plt.figure()
plt.scatter(angles, intensity_vec, label='Data', color='red')
plt.plot(angles, I_pred, label=f'Klein-Nishina Model, RMSE={rms:.3f}, $\chi^2$={chi2:.3f}')
plt.xlabel('Angle [deg]')
plt.ylabel('Intensity')
plt.grid()
plt.legend()
plt.show()