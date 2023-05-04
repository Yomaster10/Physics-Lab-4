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
#B_res_Gauss = B_res*10**4 #[G]
#print(B_res_Gauss)


print(f"\tSmall Coil Resonance Frequency: {Small_Coil_Freq:0.3f}[MHz], Resonant Magnetic Field: {B_res*10**3:0.3f}[mT]")

#%% Part A.2: Measuring k under Minimal Modulation for Resonance
print("\nPart A.2: Measuring k under Minimal Modulation for Resonance...")

Min_Modulation = 6 # arbitrary units

esra1 = pd.read_csv(f'Data/first_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

t1 = np.array(esra1['Time (s)']) # time array
V1_1 = np.array(esra1['1 (VOLT)']) # accelerating voltage array
V2_1 = np.array(esra1['2 (VOLT)']) # current array

mu_0 = scipy.constants.mu_0 #[N/A^2], vacuum permeability
#H = B_res # / mu_0
H = B_res*10**3 #[mT]

V = 0.5*(max(V1_1) - min(V1_1)) #[V]
R = 0.82 #[Ohm]
I = V/R #[A]
print(I)

k1 = H/I #[mT/A]

print(f"\tk1: {k1:0.3f}[mT/A]")

plt.figure(0)
plt.plot(t1, V1_1, label='Voltage Measurement')
plt.plot(t1, V2_1, label='Current Measurement')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid()
plt.show()

#Oscillator_Feedback = 44 #[%]
#Small_Coil_Freq = 96.25 + ((95.4-96.25)/(40-0))*(44-0) #[MHz]
#print(f"\tSmall Coil Resonance Frequency: {Small_Coil_Freq:0.2f}[MHz]")

#%% Part A.3: Measuring k with Direct Current
print("\nPart A.3: Measuring k with Direct Current...")

esra3 = pd.read_csv(f'Data/third_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

I_0 = 0.0933 #[A]
k2 = H/I_0 #[mT/A]

print(k2)

t2 = np.array(esra3['Time (s)']) # time array
V1_2 = np.array(esra3['1 (VOLT)']) # accelerating voltage array
V2_2 = np.array(esra3['2 (VOLT)']) # current array

plt.figure(1)
plt.plot(t2, V1_2, label='Voltage Measurement')
plt.plot(t2, V2_2, label='Current Measurement')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid()
plt.show()

#%% Part A.4: Resonance Detection using the XY Method
print("\nPart A.4: Resonance Detection using the XY Method...")

#esra2 = pd.read_csv(f'Data/second_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

#t2 = np.array(esra2['Time (s)']) # time array
#V1_2 = np.array(esra2['1 (VOLT)']) # accelerating voltage array
#V2_2 = np.array(esra2['2 (VOLT)']) # current array

#%% Part B.1: Measuring the Derivative of the Absorption Signal
print("\nPart B.1: Measuring the Derivative of the Absorption Signal...")

esrb = pd.read_csv(f'Data/Part B Data.csv', header=0, usecols=['Avg(X) = I_0', 'Amp(Y)/Amp(X)']) # read the data
I_dc = esrb['Avg(X) = I_0'] #[mA]
derivative = esrb['Amp(Y)/Amp(X)']

plt.figure(2)
plt.plot(I_dc/1000, derivative)
plt.xlabel("DC Current [A]")
plt.ylabel("Derivative of the Absorption Signal")
plt.title("Derivative of the Absorption Signal vs. DC Current")
plt.grid()
plt.show()

wave_gen_amplitude = 1.22 #[V] peak-to-peak

I_start = 0.380 #[A]
I_end = 0.540 #[A]

absorption = scipy.integrate.cumtrapz(derivative, x=None, dx=1.0, axis=-1, initial=None)
plt.figure(3)
plt.plot(I_dc[1:]/1000, absorption)
plt.xlabel("DC Current [A]")
plt.ylabel("Absorption Signal")
plt.title("Absorption Signal vs. DC Current")
plt.grid()
plt.show()