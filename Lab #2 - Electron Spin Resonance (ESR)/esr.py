import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.constants

#%% Part A.1: ESR Absorption Signal Calibration
print("\nPart A.1: ESR Absorption Signal Calibration...")

Oscillator_Feedback = 44 #[%]
Small_Coil_Freq = 96.25 + ((95.4-96.25)/(40-0))*(Oscillator_Feedback-0) #[MHz]

nu_RF = Small_Coil_Freq * 10**6 #[Hz]
h = scipy.constants.h #[J*sec], Planck's constant
g = 2.0036 # [dimensionless], for electrons in DPPH
mu_B = scipy.constants.physical_constants['Bohr magneton'][0] #[J/T], Bohr magneton
B_res = (h*nu_RF) / (g*mu_B) #[T]

print(f"\tSmall Coil Resonance Frequency (nu_RF): {Small_Coil_Freq:0.3f}[MHz], Resonant Magnetic Field (B_res): {B_res*10**3:0.3f}[mT]")

#% Error Analysis
del_g = 0.0002
del_nu = 0.1 #[MHz]

del_B = B_res*10**3 * np.sqrt((del_nu/Small_Coil_Freq)**2 + (del_g/g)**2) #[mT]

#%% Part A.2: Measuring k under Minimal Modulation for Resonance
print("\nPart A.2: Measuring k under Minimal Modulation for Resonance...")

Min_Modulation = 6 # arbitrary units

esra1 = pd.read_csv(f'Data/first_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data
t1 = np.array(esra1['Time (s)']) # time array
V1_1 = np.array(esra1['1 (VOLT)']) # accelerating voltage array
V2_1 = np.array(esra1['2 (VOLT)']) # current array

mu_0 = scipy.constants.mu_0 #[N/A^2], vacuum permeability
H = B_res / mu_0 #[A/m]

V = 0.5*(max(V1_1) - min(V1_1)) #[V]
R = 0.82 #[Ohm]
I = V/R #[A]
print(f"\tMagnetic Field Strength (H): {H:0.3f}[A/m], External Coil Current (I_coil): {I:0.3f}[A]")

k1 = H/I #[1/m]
print(f"\tCoil Constant (k): {k1:0.3f}[1/m]")

plt.figure(0)
plt.plot(t1, V1_1, label='External Coil Voltage')
plt.plot(t1, V2_1, label='Oscillator Signal')
plt.xlabel("Relative Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Signal Measurements for Minimal Modulation Resonance")
plt.legend(loc='lower right')
plt.grid()
plt.show()

#%% Part A.3: Measuring k with Direct Current
print("\nPart A.3: Measuring k with Direct Current...")

esra3 = pd.read_csv(f'Data/third_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

t3 = np.array(esra3['Time (s)']) # time array
V1_3 = np.array(esra3['1 (VOLT)']) # accelerating voltage array
V2_3 = np.array(esra3['2 (VOLT)']) # current array

V_0 = 0.390 #[V]
I_0 = V_0 / R #[A]
print(f"\tDC Current (I_0): {I_0:0.3f}[A]")

peaks = []
for v in range(len(V1_3)):
    if abs(V1_3[v]-V_0) < 1*10**-2:
        peaks.append(t3[v])

k2 = H/I_0 #[1/m]
print(f"\tCoil Constant (k): {k2:0.3f}[1/m]")

plt.figure(1)
plt.plot(t3, V1_3, label='External Coil Voltage')
plt.plot(t3, V2_3, label='Oscillator Signal')
for p in peaks: 
    plt.axvline(x = p, color = 'grey', linestyle = '--')
plt.axvline(x = -0.019, color = 'grey', linestyle = '--', label='Peaks')
plt.axvline(x = -0.00913, color = 'grey', linestyle = '--')
plt.axhline(y = 0.39, color = 'black', linestyle = '--', label='$V_0$')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Signal Measurements for Direct Current Measurement")
plt.legend()
plt.grid()
plt.show()

#%% Part A.4: Resonance Detection using the XY Method
print("\nPart A.4: Resonance Detection using the XY Method...")

esra2 = pd.read_csv(f'Data/second_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data

t2 = np.array(esra2['Time (s)']) # time array
V1_2 = np.array(esra2['1 (VOLT)']) # accelerating voltage array
V2_2 = np.array(esra2['2 (VOLT)']) # current array

V_XY = 0.388 #[V]
I_XY = V_XY / R #[A]
print(f"\tExternal Coil Current (I_coil): {I_XY:0.3f}[A]")
k3 = H/I_XY #[1/m]
print(f"\tCoil Constant (k): {k3:0.3f}[1/m]")

peaks = [-0.022135,-0.0021875,0.01781]
for v in range(len(V1_2)):
    if abs(V1_2[v]-V_XY) < 0.2*10**-2 or abs(V1_2[v]-(-V_XY)) < 0.2*10**-2:
        peaks.append(t2[v])

plt.figure(2)
plt.plot(V1_2, V2_2, label='Signal')    
plt.xlabel("External Coil Voltage [V]")
plt.ylabel("Oscillator Signal [V]")
plt.axvline(x = V_XY, color = 'black', linestyle = '--', label='Resonance')
plt.axvline(x = -V_XY, color = 'black', linestyle = '--')
plt.title("XY Signal Representation")
plt.grid()
plt.legend()

plt.figure(3)
plt.plot(t2, V1_2, label='External Coil Voltage', linewidth=2)
plt.axhline(y = V_XY, color = 'black', linestyle = '--', label='Resonance')
plt.axhline(y = -V_XY, color = 'black', linestyle = '--')
plt.axvline(x = peaks[0], color = 'grey', linestyle = '--', label='Peaks')
for p in peaks[1:]:
    plt.axvline(x = p, color = 'grey', linestyle = '--')
plt.plot(t2, V2_2, label='Oscillator Signal', linewidth=2)    
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Signal Measurements for XY Method Measurement")
plt.legend()
plt.grid()
plt.show()

#%% Part B.1: Measuring the Derivative of the Absorption Signal
print("\nPart B.1: Measuring the Derivative of the Absorption Signal...")

wave_gen_amplitude = 1.22 #[V] peak-to-peak
I_start = 0.380 #[A]
I_end = 0.540 #[A]

esrb = pd.read_csv(f'Data/Part B Data.csv', header=0, usecols=['Avg(X) = I_0', 'Amp(Y)/Amp(X)']) # read the data
I_dc = esrb['Avg(X) = I_0'] #[mA]
I = list(I_dc[1:]/1000) #[A]

P_tag = esrb['Amp(Y)/Amp(X)']
P = scipy.integrate.cumtrapz(P_tag, x=None, dx=1.0, axis=-1, initial=None)

k = np.mean(k1+k2+k3)
omega = list((2*np.pi*g*mu_B*mu_0*k/h)*(I_dc/1000)*10**-9) #[Grad/s]

idx_0 = np.argmax(P)
omega_0 = omega[idx_0+1] #[Grad/s]
print(f"\tFree Precession Frequency (w_0): {omega_0:0.3f}[Grad/s]")

T_2 = 2*np.pi / (omega[idx_0+1]) #[ns]
print(f"\tSpin-Spin Relaxation Time (T_2): {T_2:0.3f}[ns]")

plt.figure(4)
plt.plot(I_dc/1000, P_tag)
plt.xlabel("DC Current [A]")
plt.ylabel("Derivative of the Absorption Signal")
plt.title("Derivative of the Absorption Signal vs. DC Current")
plt.grid()

plt.figure(5)
plt.plot(I, P)
plt.xlabel("DC Current [A]")
plt.ylabel("Absorption Signal")
plt.title("Absorption Signal vs. DC Current")
plt.grid()

plt.figure(6)
plt.plot(omega[1:], P, label='Signal')
plt.axvline(x = omega_0, color = 'grey', linestyle = '--', label='Resonance')
plt.plot(omega_0, P[idx_0], '.', label='Peak')
plt.xlabel("$\omega$ [Grad/s]")
plt.ylabel("Absorption Signal")
plt.title("Absorption Signal vs. Free Precession Frequency")
plt.grid()
plt.legend()
plt.show()