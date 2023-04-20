import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants

#%% Part A.1: ESR Absorption Signal Calibration
print("\nPart A.1: ESR Absorption Signal Calibration...")

Oscillator_Feedback = 44 #[%]
Small_Coil_Freq = 96.25 + ((95.4-96.25)/(40-0))*(Oscillator_Feedback-0) #[MHz]

ni_RF = Small_Coil_Freq * 10**6 #[Hz]
h = scipy.constants.h #[J*sec], Planck's constant
g = 2.0036 # [dimensionless], for electrons in DPPH
mu_B = scipy.constants.physical_constants['Bohr magneton'][0] #[J/T], Bohr magneton

B_res = (h*ni_RF) / (g*mu_B) #[T]

print(f"\tSmall Coil Resonance Frequency: {Small_Coil_Freq:0.3f}[MHz], Resonant Magnetic Field: {B_res*10**3:0.3f}[mT]")

#%% Part A.2: Measuring k under Minimal Modulation for Resonance
print("\nPart A.2: Measuring k under Minimal Modulation for Resonance...")

Min_Modulation = 6 # arbitrary units

esra1 = pd.read_csv(f'Data/first_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

t1 = np.array(esra1['Time (s)']) # time array
V1_1 = np.array(esra1['1 (VOLT)']) # accelerating voltage array
V2_1 = np.array(esra1['2 (VOLT)']) # current array

mu_0 = scipy.constants.mu_0 #[N/A^2], vacuum permeability
H = B_res / mu_0

V = 0.5*(max(V1_1) - min(V1_1)) #[V]
R = 0.82 #[Ohm]
I = V/R #[A]
print(V)

k1 = H/I

print(k1)

plt.figure(0)
plt.plot(t1, V1_1, label='V_1')
#plt.plot(t1, V2_1, label='V_2')
plt.show()

#Oscillator_Feedback = 44 #[%]
#Small_Coil_Freq = 96.25 + ((95.4-96.25)/(40-0))*(44-0) #[MHz]
#print(f"\tSmall Coil Resonance Frequency: {Small_Coil_Freq:0.2f}[MHz]")

#%% Part A.3: Measuring k with Direct Current
print("\nPart A.3: Measuring k with Direct Current...")

esra3 = pd.read_csv(f'Data/third_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

I_0 = 0.0933 #[A]

k2 = H/I_0

print(k2)

t3 = np.array(esra3['Time (s)']) # time array
V1_2 = np.array(esra3['1 (VOLT)']) # accelerating voltage array
V2_2 = np.array(esra3['2 (VOLT)']) # current array

#%% Part A.4: Resonance Detection using the XY Method
print("\nPart A.4: Resonance Detection using the XY Method...")

#esra2 = pd.read_csv(f'Data/second_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

#t2 = np.array(esra2['Time (s)']) # time array
#V1_2 = np.array(esra2['1 (VOLT)']) # accelerating voltage array
#V2_2 = np.array(esra2['2 (VOLT)']) # current array