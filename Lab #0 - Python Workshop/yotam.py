import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
from sklearn.metrics import r2_score # import function that calculates R^2 score

#%% Ohm

def I_R(V2, R1):
    """ (2) Calculates the current through the resistor, given the voltage and nominal resistance """
    return V2 / R1

def V_R(V1, V2):
    """ (3) Calculates the voltage drop across the resistor, given the voltage at each end """
    return V1 - V2

def R_t(V_R, I_R):
    """ (4) Calculates the resistance of the resistor, given the voltage and current across it """
    return V_R / I_R

def P_t(V_R, I_R):
    """ (5) Calculates the power consumed by the resistor, given the voltage and current across it """
    return V_R * I_R

def Energy(P_t, t):
    """ (6) Calculates the energy change of the resistor at a given time T, given an array of power values and an array of corresponding time values from 0 to T """
    return np.trapz(P_t, x=t)#, dx=1.0, axis=-1)

print("\nOhm's Law Section...")
R1 = 5.48 #[Ohm]

## (7-9)
R_data = pd.read_csv('Data/ohm.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # TO DO: explain header

## (10)
R_data.rename(columns = {'Time (s)':'t', '1 (VOLT)':'V1', '2 (VOLT)':'V2'}, inplace = True)

## (11)
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

## (13-14)
result = linregress(new_Q_t_vec, new_R_t_vec)
R_0 = result.intercept #[Ohm]
alpha_times_R_0_over_C_heat = result.slope #[Ohm/J]
alpha_over_C_heat = alpha_times_R_0_over_C_heat / R_0 #[1/J]

print(f"\tR0: {R_0:0.2f}[Ohm], Alpha/C_heat: {alpha_over_C_heat:0.2f}[1/J]")
# We get R0: 1.88[Ohm], Alpha/C_heat: 0.26[1/J], this is the same as the solution!

## (12-13)
plt.figure(0)
plt.plot(new_Q_t_vec, new_R_t_vec, 'b', label='measurement')
plt.plot(new_Q_t_vec, R_0 + np.dot(alpha_times_R_0_over_C_heat, new_Q_t_vec), 'orange', label='regression')
plt.legend()
plt.xlabel("Energy [J]")
plt.ylabel("Resistance [Ohm]")
plt.title("Resistance vs. Energy")
plt.grid()
plt.show()

#%% Inductance

def flux(time, voltage):
    """ (6) Calculates the magnetic flux a given time T, given an array of voltage values and an array of corresponding time values from 0 to T """
    return -1*np.trapz(voltage, x=time)

print("\nInductance Section...")

## (2)
h = np.array([30, 24, 18, 14, 8]) / 100 #[m]

Ind_data = []; t_coil = []; h_over_t_coil = []
for n in range(0,5):
    ## (3)
    df = pd.read_csv('Data/Trace %d.csv'%n, header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)'])
    ## (4)
    df.rename(columns = {'Time (s)':'t', '1 (VOLT)':'ref', '2 (VOLT)':'signal'}, inplace = True)
    Ind_data.append(df) 

    ## (5)
    plt.figure(1)
    # Voltage vs. Time
    plt.plot(df['t'], df['ref'], '--', label=f'ref {h[n]:0.2f}')
    plt.plot(df['t'], df['signal'], label=f'signal {h[n]:0.2f}')

    ## (7)
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
    
    ## (8)
    ref_max_flux_idx = np.argmax(np.abs(ref_flux_vec))
    signal_max_flux_idx = np.argmax(np.abs(signal_flux_vec))
    
    ## (7,9)
    plt.figure(2)
    # Flux vs. Time
    plt.plot(time_vec, ref_flux_vec, '--', label=f'ref {h[n]:0.2f}')
    plt.plot(time_vec, signal_flux_vec, label=f'signal {h[n]:0.2f}')
    # Max. flux points, yes these are the right ones!
    plt.plot(time_vec[ref_max_flux_idx], ref_flux_vec[ref_max_flux_idx], 'ro')
    plt.plot(time_vec[signal_max_flux_idx], signal_flux_vec[signal_max_flux_idx], 'ro')

    ## (10)
    ref_max_flux_time = time_vec[ref_max_flux_idx] #[s]
    adjusted_time_vec = [t - ref_max_flux_time for t in time_vec]
    ## (11)
    t_coil.append(adjusted_time_vec[signal_max_flux_idx])
    h_over_t_coil.append(h[n] / t_coil[n])

## (13)
t_coil_err = 2*10**-3 #[s]
h_err = 0.1*10**-2 #[m]
h_over_t_coil_err = []
for i in range(len(h_over_t_coil)):
    err = h_over_t_coil[i] * np.sqrt((t_coil_err/t_coil[i])**2 + (h_err/h[i])**2)
    h_over_t_coil_err.append(err)

## (14)
res = linregress(t_coil, h_over_t_coil)
v_0 = res.intercept #[m/s]
a = 2*res.slope #[m/s^2]
h_over_t_coil_pred = v_0 + np.dot(a/2, t_coil) #[m/s]
print(f"\tv0: {v_0:0.2f}[m/s], a: {a:0.2f}[m/s^2]")
# We get v0: 1.45[m/s], a: 8.36[m/s^2], this is the same as the solution!
# The acceleration here is less than free-fall acceleration (9.81[m/s^2]).

## (16)
R_squared_reg = res.rvalue**2
# Alternative (equivalent) command: R_squared_reg = r2_score(h_over_t_coil, h_over_t_coil_pred)
print(f"\tR^2 Value: {R_squared_reg:0.2f}")

## (17)
e = np.sqrt(np.power(h_over_t_coil_err,2) + (res.slope*t_coil_err)**2) # from the Data Analysis Booklet (pg. 10)
Chi_squared = sum(((h_over_t_coil - h_over_t_coil_pred) / e)**2) # from the Data Analysis Booklet (pg. 10)
print(f"\tChi^2 Value: {Chi_squared:0.3f}")

N = 5 # we have 5 samples total
DoF = N - 2 # from the Data Analysis Booklet (pg. 10)
print(f"\tDegrees of Freedom: {DoF}")

p_value = 1 - scipy.stats.chi2.cdf(Chi_squared, DoF)
print(f"\tp-value: {p_value:0.3f}")

## (5)
plt.figure(1)
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs. Time")
plt.grid()

## (7)
plt.figure(2)
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Magnetic Flux [T*m]")
plt.title("Magnetic Flux vs. Time")
plt.grid()

plt.figure(3)
## (12)
plt.plot(t_coil, h_over_t_coil, 'bo', label='measurements')
## (13)
plt.errorbar(t_coil, h_over_t_coil, xerr=t_coil_err, yerr=h_over_t_coil_err, fmt='none', color='orange', label='errors')
## (15)
plt.plot(t_coil, h_over_t_coil_pred, 'green', label='regression')
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Height / Time [m/s]")
plt.title("Height / Time vs. Time")
plt.grid()

plt.show()