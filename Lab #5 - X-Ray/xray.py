import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
from scipy.signal import find_peaks         #for find local peaks in a signal

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts #running average
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

X_Vec = []
Y_Vec = []

#%% Molybdenum

Mo_Data = pd.read_csv('molybdenum_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Mo_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Mo_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=20)
correct_peaks = [peaks[0], int(np.mean(peaks[1:3])), peaks[3]]
#print("Molybdenum peaks", correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Molybdenum Spectrum')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

x = correct_peaks
#y = [2293.2, 2394.8]
y = [8027.8, 8905.3, 17479.3]
#y = [17374.3, 17479.3]

for val in x:
    X_Vec.append(val)
for val in y:
    Y_Vec.append(val)

reg = linregress(x, y)
a = reg.slope
b = reg.intercept

#print(a, b)

unknown_peak = a*correct_peaks[2] + b
#print(unknown_peak) 

#%% Copper

Cu_Data = pd.read_csv('copper_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Cu_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Cu_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=50) #was 20
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Copper Spectrum')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

#x = correct_peaks
#y = [929.7, 949.8]
#y = [8027.8, 8905.3]

#for val in x:
 #   X_Vec.append(val)
#for val in y:
#    Y_Vec.append(val)

#reg = linregress(x, y)
#a = reg.slope
#b = reg.intercept

#print(a, b)

#unknown_peak = a*correct_peaks[2] + b
#print(unknown_peak)

#%% Nickel

Ni_Data = pd.read_csv('nickel_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ni_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ni_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=10)
correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Nickel Spectrum')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

x = correct_peaks
#y = [851.5, 868.8]
y = [7460.9, 8264.7] #7478.2]

for val in x:
    X_Vec.append(val)
for val in y:
    Y_Vec.append(val)

reg = linregress(x, y)
a = reg.slope
b = reg.intercept

#print(a, b)

#%% Titanium

Ti_Data = pd.read_csv('titanium_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ti_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ti_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=10)
print("Titanium peaks", peaks)
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Titanium Spectrum')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

x = peaks[:1]
y = [4510.8]#, 4931.8]

for val in x:
    X_Vec.append(val)
for val in y:
    Y_Vec.append(val)

#reg = linregress(x, y)
#a = reg.slope
#b = reg.intercept

#print(a, b)

#%%
reg = linregress(X_Vec, Y_Vec)
a = reg.slope
b = reg.intercept

#print(a, b)

#print(X_Vec, Y_Vec)
plt.figure(dpi=300)
plt.plot(X_Vec, Y_Vec, '.')
plt.xlabel("Channels")
plt.ylabel("Energy [eV]")

X_Vec = np.array(X_Vec)
Y_Vec = np.array(Y_Vec)
reg = linregress(X_Vec, Y_Vec)
a = reg.slope
b = reg.intercept
plt.plot(X_Vec, a*X_Vec+b)
#plt.show()

print(a, b)

#coin was ten shekels

#%% Unknown Source

Ti_Data = pd.read_csv('unknown_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ti_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ti_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=20)
print("Unknown source Peaks: ", peaks)
peaks = np.array(peaks)
energy_values = a*peaks + b
print("Unknown source energy values: ", energy_values)
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Spectrum of Unknown Source')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

#%% Unknown Source (Coin)

Ti_Data = pd.read_csv('unknown_coin_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ti_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ti_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=15) 
print("Unknown Coin Peaks: ", peaks)
peaks = np.array(peaks)
energy_values = a*peaks + b
print("Unknown Coin energy values: ", energy_values)
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Spectrum of 10 Cent Coin')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

#%% Strontianit

Ti_Data = pd.read_csv('strontianit_yurr.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ti_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ti_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=20, prominence=10)
print("Strontianit Peaks: ", peaks)
peaks = np.array(peaks)
energy_values = a*peaks + b
print("Strontianit energy values: ", energy_values)
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Spectrum of Strontianit Sample')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()

#%% Pyrrhotin

Ti_Data = pd.read_csv('pyrrhotin_yuur.txt',sep='\t',header=1) # read the data.
Counts = np.array(Ti_Data['Impulses/#']) # Impulses
Counts_smoothed = smooth(Counts, 10) # smooth the data over 10 channels
Channels = np.array(Ti_Data['Channel/#']) # Channel

peaks, _ = find_peaks(Counts_smoothed, height=10, prominence=10)
print("Pyrrhotin Peaks: ", peaks)
peaks = np.array(peaks)
energy_values = a*peaks + b
print("Pyrrhotin energy values: ", energy_values)
#correct_peaks = [int(np.mean(peaks[0:3])), peaks[3]]
#print(correct_peaks)

plt.figure(dpi=300)
plt.plot(Channels, Counts_smoothed, label='Spectrum of Pyrrhotin Sample')
plt.plot(Channels[peaks], Counts_smoothed[peaks], "x", color='red', label='Peaks')
plt.ylabel('Impulses')
plt.xlabel('Channels')
plt.legend()
plt.show()