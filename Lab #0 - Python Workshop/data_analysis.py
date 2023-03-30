# -*- coding: utf-8 -*-
import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
from sklearn.metrics import r2_score # import function that calculates R^2 score

import warnings
warnings.simplefilter("ignore", RuntimeWarning) # command for ignoring runtime warnings sometimes caused by the np.log() function

#%% Potential

def potential(x,y,a,C):
    """ (4) Calculates the electric potential function given x,y,a,C """
    V = -C*np.log(np.sqrt((x-a)**2+y**2)/a) + C*np.log(np.sqrt((x+a)**2+y**2)/a)
    return V

print("\nElectric Potential Mapping Section...")

## (2)
C = 1
a = 1
L = 3
N = 100
coord = np.linspace(-L, L , N) # defines coordinates
coord_x, coord_y = np.meshgrid(coord, coord)

## (3) linspace(arg1,arg2,arg3) creates an array starting the value of arg1 and ends with arg2 having arg3 
# elements, while meshgrid creates a rectangular grid out of 2 given arrays (in one dimension) that denotes the Matrix.

## (5-6)
V_xy = potential(coord_x, coord_y, a, C) # calculates the potential function for the given values 
plt.figure(0) # creates a Matplotlib figure
plt.pcolormesh(coord_x, coord_y, V_xy) # gives colors for different values in the graph
plt.colorbar() # legend for the map between colors and numbers, placed on the right 

## (7)
for i in range(1,10):
   plt.contour(coord_x, coord_y, V_xy, np.sort([-i , 0 , i]), cmap='spring')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Spatial Distribution of the Electric Potential Function")
plt.grid()
plt.show()   
# cmap is the map from numbers to colors

## (8)
x = np.linspace(-1, 1, N)
V_x = potential(x, 0, a, C)

plt.figure(1)
plt.plot(x, V_x, '.', label='calculated potential')
plt.xlabel("x [m]")
plt.ylabel("Potential [V]")
plt.title("Potential Function vs. x-Coordinate")
plt.grid()
plt.legend()
plt.show()

#%% Capacitor

def V_decay(t,tau,V0):
    """ (9) Calculates the voltage decay given t, tau, and V0 """
    return V0*np.exp(-t/tau)

print("\nCapacitor Section...")

## (2)
eps0 = scipy.constants.epsilon_0 #[F/m]
D = 18e-2 #[m]
d = 0.5e-3 #[m]
C_theoretical = (eps0*np.pi*D**2)/(4*d)
print(f"\tTheoretical C: {C_theoretical*10**10:0.1f}e-10[F]")
# We get C_theoretical = 4.5e-10[F], this is the same as the solution!

## (3)
R_tot = 38.4e+3 #[Ohm]
R = 977 #[Ohm]
tau_theoretical = R_tot*C_theoretical
print(f"\tTheoretical tau: {tau_theoretical*10**5:0.2f}e-5[sec]")
# We get tau_theoretical = 1.73e-5[sec], this is the same as the solution!

## (4)
C_data = pd.read_csv('Data/capacitor.csv')

## (5)
C_data = C_data.rename(columns = {"time (sec)":'t', "ch2":'V_R'})

## (6)
C_data['V_C'] = C_data['ch1'] - C_data['V_R']

## (7)
t = np.asarray(C_data['t'].values)
V_C = np.asarray(C_data['V_C'].values)
plt.figure(2)
plt.plot(t*10**6, V_C, '.b', label="measured V_C")

## (8) Curve fitting is a tool for creating the representive curve based on discrete data.
# This curve can characterize the relation between the two variables.

## (9)
p_optimal, p_covariance = cfit(V_decay, t, V_C, p0=[1.73e-5,4]) # non-linear curve fitting

## (10) cfit finds V0 (initial voltage) and tau (time constant)

## (11)
tau_fit = p_optimal[0]
V0_fit = p_optimal[1]
print(f"\tFit tau: {tau_fit*10**5:0.3f}e-5[sec], Fit V0: {V0_fit:0.2f}[V]")
# We get tau_fit = 2.044e-5[sec] and V0_fit = 3.99[V], this is the same as the solution!

## (12)
err = np.sqrt(np.diag(p_covariance))
tau_fit_err = err[0]
V0_fit_err = err[1]
print(f"\tFit tau error: {tau_fit_err*10**8:0.0f}e-8[sec], Fit V0 error: {V0_fit_err:0.2f}[V]")
# We get tau_fit_err = 8e-8[sec] and V0_fit_err = 0.01[V], this is the same as the solution!

## (13-14)
V_t = V_decay(t, p_optimal[0], p_optimal[1])
plt.plot(t*10**6, V_t, color='orange', label='fitted curve')
plt.xlabel("Time [$\mu$ sec]")
plt.ylabel("Voltage on Capacitor [V]")
plt.title("Voltage on Capacitor vs. Time")
plt.legend()
plt.grid()
plt.show()
## (13) The fitted curve does not fit well with the measurements because we didn't provide approximated initial values.
## (14) p0 is giving an initial guess for the parameters for a better fit, so now the curve fits the measurements well.

## (15)
V_C_err = 0.05 #[V]
r = V_C - V_t
Chi2 = sum((r / V_C_err)**2)
print(f"\n\tChi^2 Value: {Chi2:0.1f}")
# Chi2 = 267.5, as expected

N = 420 # we have 420 samples total
DoF = N - 2 # from the Data Analysis Booklet (pg. 10)
print(f"\tDegrees of Freedom: {DoF}")
# DoF = 418, as expected

## (16)
p_value = 1 - scipy.stats.chi2.cdf(Chi2, DoF)
print(f"\tp-value: {p_value:0.3f}")
# p_value ~ 1, as expected

## (17)
R_squared_of_fit = r2_score(V_C, V_t) 
print(f"\tFit R^2 Value: {R_squared_of_fit:0.3f}")
# R^2 = 0.998 (as expected), this value is reasonable since it seems like a good fit

## (18)
plt.figure(3)
plt.plot(t, np.log(V_C), '.b')
plt.grid()

## (19)
t1, t2 = 0, 0.00004
inds = (C_data['t'] > t1) & (C_data['t'] < t2)
plt.plot(C_data['t'][inds], np.log(C_data['V_C'][inds]), '.', label='data', color='orange')

## (20) Linear regression is the predictive analysis aim to describe relation between variables.
 
## (21)
reg = linregress(C_data['t'][inds], np.log(C_data['V_C'][inds]))
print("\n\t", reg, "\n")

## (22)
tau_reg = -1/reg.slope
V0_reg = np.exp(reg.intercept)
print(f"\tReg. tau: {tau_reg*10**5:0.3f}e-5[sec], Reg. V0: {V0_reg:0.2f}[V]")
# We get tau_reg = 2.067e-5[sec] and V0_reg = 3.98[V], this is the same as the solution!

## (23)
tau_reg_err = -tau_reg * reg.stderr / reg.slope #[sec], error propagation
V0_reg_err = V0_reg * reg.intercept_stderr #[V], error propagation
print(f"\tReg. tau error: {tau_reg_err*10**8:0.0f}e-8[sec], Reg. V0 error: {V0_reg_err:0.2f}[V]")
# We get tau_reg_err = 7e-8[sec] and V0_reg_err = 0.02[V], this is the same as the solution!
 
## (24)
y_pred = reg.slope*C_data['t'][inds] + reg.intercept
R_squared_of_reg = r2_score(y_pred, np.log(C_data['V_C'][inds])) 
print(f"\tReg. R^2 Value: {R_squared_of_reg:0.3f}")
# R^2 = 0.999, as expected

## (25)
plt.plot(t[inds], np.log(V_decay(t, tau_reg, V0_reg)[inds]), label='regression line', color='green')
plt.xlabel("Time [sec]")
plt.ylabel("log(Voltage on Capacitor)")
plt.xlim([-0.00001, 0.00011])
plt.ylim([-4.5, 1.5])
plt.title("log(Voltage on Capacitor) vs. Time")
plt.legend()
plt.show()

## (26)
V_R = C_data['V_R']
C_data['int_V_R'] = scipy.integrate.cumtrapz(V_R, x=t, initial=0)

## (27)
delta_V = V_C - V0_fit
plt.figure(4)
plt.plot(C_data['int_V_R'], delta_V, '.b', label='measurements')

## (28)
reg = linregress(C_data['int_V_R'], delta_V)
y_pred = reg.slope*C_data['int_V_R'] + reg.intercept
#r2_reg = r2_score(y_pred, delta_V) 

C_meas = 1/(R*reg.slope) #[F]
print(f"\tMeasured C: {int(C_meas*10**11)/10:0.1f}e-10[F]")
# We get C_meas = 4.8e-10[F], this is the same as the solution!

## (29)
plt.plot(C_data['int_V_R'], y_pred, label='fitted line', color='orange')
plt.xlabel("Integral of $V_R$")
plt.ylabel("$\Delta V_C$")
plt.title("$\Delta V_C$ as a function of Integral of $V_R$")
plt.legend()
plt.grid()
plt.show()

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
R_data = pd.read_csv('Data/ohm.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)'])
## (8): The 'header' parameter determines which row in the .csv file to use as the names for the columns
# in the dataframe - here we choose the second row to be the column names (since the first row doesn't provide
# helpful names for this case), thus we set 'header=1'

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
plt.figure(5)
plt.plot(new_Q_t_vec, new_R_t_vec, 'b', label='measurement')
plt.plot(new_Q_t_vec, R_0 + np.dot(alpha_times_R_0_over_C_heat, new_Q_t_vec), 'orange', label='regression')
plt.legend()
plt.xlabel("Energy [J]")
plt.ylabel("Resistance [$\Omega$]")
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
    plt.figure(6)
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
    plt.figure(7)
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
plt.figure(6)
plt.legend(loc='lower right')
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [V]")
plt.title("Voltage vs. Time")
plt.grid()

## (7)
plt.figure(7)
plt.legend(loc='center right')
plt.xlabel("Time [sec]")
plt.ylabel("Magnetic Flux [T*m]")
plt.title("Magnetic Flux vs. Time")
plt.grid()

## (12) The x-axis is t_coil and the y-axis is h_over_t_coil, this allows us to use a linear regression, 
# since we expect h_over_t_coil = (a/2)*t_coil + v_0
plt.figure(8)
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

#%% Useful & Not Well-Understood Functions

## Useful Functions:
# np.linspace()
# np.meshgrid()
# pd.read_csv()
# cfit()
# abs()
# sum()
# r2_score()
# linregress()
# np.log()
# np.exp()
# np.sqrt()
# np.pow()

## Not Well-Understood Functions
# N/A

#%% Bonus

print("\nBonus Section...")

consts = {'Electron Charge [C]':scipy.constants.e, 'Vacuum Permeability [N/A^2]':scipy.constants.mu_0, 'Vacuum Permittivity [F/m]':scipy.constants.epsilon_0, 'Electron Mass [kg]':scipy.constants.m_e}
for i in consts:
    print(f"\t{i}: {consts[i]}")

print("\nDone!\n")