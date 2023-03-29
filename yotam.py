import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
#from sklearn.metrics import r2_score # import function that calculates R^2 score

#%% Ohm

def I_R(V2, R1):
    return V2 / R1

def V_R(V1, V2):
    return V1 - V2

def R_t(V_R, I_R):
    return V_R / I_R

def P_t(V_R, I_R):
    return V_R * I_R

def Energy(P_t, t):
    return np.trapz(P_t, x=t)#, dx=1.0, axis=-1)

R1 = 5.48 #[Ohm]
R_data = pd.read_csv('Data/ohm.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # TO DO: explain header
R_data.rename(columns = {'Time (s)':'t', '1 (VOLT)':'V1', '2 (VOLT)':'V2'}, inplace = True)

t_vec = []; R_t_vec = []; P_t_vec = []; Q_t_vec = []
for i in range(len(R_data)):
    t = R_data.iloc[i]['t'] #[s]
    V1 = R_data.iloc[i]['V1'] #[V]
    V2 = R_data.iloc[i]['V2'] #[V]
    
    v_R = V_R(V1, V2) #[V]
    i_R = I_R(V2, R1) #[A]
    r_t = R_t(v_R, i_R) #[Ohm]
    p_t = P_t(v_R, i_R) #[W]

    t_vec.append(t)
    R_t_vec.append(r_t)
    P_t_vec.append(p_t)

    q_t = Energy(P_t_vec, t_vec) #[J]
    Q_t_vec.append(q_t)

# Let's examine the linear region only
new_Q_t_vec = []; new_R_t_vec = []
for j in range(len(t_vec)):
    if Q_t_vec[j] < 10**-2 or Q_t_vec[j] > 0.97:
        continue
    new_Q_t_vec.append(Q_t_vec[j])
    new_R_t_vec.append(R_t_vec[j])

result = linregress(new_Q_t_vec, new_R_t_vec)
R_0 = result.intercept #[Ohm]
alpha_times_R_0_over_C_heat = result.slope #[Ohm/J]
alpha_over_C_heat = alpha_times_R_0_over_C_heat / R_0 #[1/J]

print(f"R0: {R_0:0.2f}[Ohm], Alpha/C_heat: {alpha_over_C_heat:0.2f}[1/J]")
# We get R0: 1.88[Ohm], Alpha/C_heat: 0.26[1/J], this is the same as the solution!

"""
plt.figure(0)
plt.plot(new_Q_t_vec, new_R_t_vec, 'b', label='measurement')
plt.plot(new_Q_t_vec, R_0 + np.dot(alpha_times_R_0_over_C_heat, new_Q_t_vec), 'orange', label='regression')
plt.legend()
plt.xlabel("Energy [J]")
plt.ylabel("Resistance [Ohm]")
plt.title("Resistance vs. Energy")
plt.grid()
plt.show()
"""

#%% Inductance

def flux(time, voltage):
    return -1*np.trapz(voltage, x=time)

h = np.array([30, 24, 18, 14, 8]) / 100 #[m]

Ind_data = []; t_coil = []; h_over_t_coil = []
for n in range(0,5):
    df = pd.read_csv('Data/Trace %d.csv'%n, header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)'])
    df.rename(columns = {'Time (s)':'t', '1 (VOLT)':'ref', '2 (VOLT)':'signal'}, inplace = True)
    Ind_data.append(df) 

    plt.figure(1)
    # Voltage vs. Time
    plt.plot(df['t'], df['ref'], '--', label=f'ref {h[n]:0.2f}')
    plt.plot(df['t'], df['signal'], label=f'signal {h[n]:0.2f}')

    time_vec = []; ref_vec = []; signal_vec = []
    ref_flux_vec = []; signal_flux_vec = []
    for t in range(len(df['t'])):
        time_vec.append(df['t'][t])
        ref_vec.append(df['ref'][t])
        signal_vec.append(df['signal'][t])

        ref_flux = flux(time_vec, ref_vec) #[T*m]
        ref_flux_vec.append(ref_flux)

        signal_flux = flux(time_vec, signal_vec) #[T*m]
        signal_flux_vec.append(signal_flux)
    
    ref_max_flux_idx = np.argmax(np.abs(ref_flux_vec))
    signal_max_flux_idx = np.argmax(np.abs(signal_flux_vec))
    
    plt.figure(2)
    # Flux vs. Time
    plt.plot(time_vec, ref_flux_vec, '--', label=f'ref {h[n]:0.2f}')
    plt.plot(time_vec, signal_flux_vec, label=f'signal {h[n]:0.2f}')
    # Max. flux points
    plt.plot(time_vec[ref_max_flux_idx], ref_flux_vec[ref_max_flux_idx], 'ro')
    plt.plot(time_vec[signal_max_flux_idx], signal_flux_vec[signal_max_flux_idx], 'ro')

    ref_max_flux_time = time_vec[ref_max_flux_idx] #[s]
    adjusted_time_vec = [t - ref_max_flux_time for t in time_vec]
    t_coil.append(adjusted_time_vec[signal_max_flux_idx])
    h_over_t_coil.append(h[n] / t_coil[n])

res = linregress(t_coil, h_over_t_coil)
v_0 = res.intercept #[m/s]
a = 2*res.slope #[m/s^2]

print(f"v0: {v_0:0.2f}[m/s], a: {a:0.2f}[m/s^2]")
# We get v0: 1.88[Ohm], Alpha/C_heat: 0.26[1/J], this is the same as the solution!

plt.figure(1)
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs. Time")
plt.grid()

plt.figure(2)
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Magnetic Flux [T*m]")
plt.title("Magnetic Flux vs. Time")
plt.grid()

plt.figure(3)
plt.plot(t_coil, h_over_t_coil, 'bo', label='measurements')
plt.plot(t_coil, v_0 + np.dot(a/2, t_coil), 'green', label='regression')
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Height / Time [m/s]")
plt.title("Height / Time vs. Time")
plt.grid()

plt.show()