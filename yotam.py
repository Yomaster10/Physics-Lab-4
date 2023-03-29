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
    t = R_data.iloc[i]['t']
    V1 = R_data.iloc[i]['V1']
    V2 = R_data.iloc[i]['V2']
    
    v_R = V_R(V1, V2)
    i_R = I_R(V2, R1)
    r_t = R_t(v_R, i_R)
    p_t = P_t(v_R, i_R)

    t_vec.append(t)
    R_t_vec.append(r_t)
    P_t_vec.append(p_t)

    q_t = Energy(P_t_vec, t_vec)
    Q_t_vec.append(q_t)

# Let's examine the linear region only
new_Q_t_vec = []; new_R_t_vec = []
for j in range(len(t_vec)):
    if Q_t_vec[j] < 10**-2 or Q_t_vec[j] > 0.97:
        continue
    new_Q_t_vec.append(Q_t_vec[j])
    new_R_t_vec.append(R_t_vec[j])

result = linregress(new_Q_t_vec, new_R_t_vec)
R_0 = result.intercept
alpha_times_R_0_over_C_heat = result.slope
alpha_over_C_heat = alpha_times_R_0_over_C_heat / R_0

print(f"R0: {R_0:0.2f}[Ohm], Alpha/C_heat: {alpha_over_C_heat:0.2f}[1/J]")
# We get R0: 1.88[Ohm], Alpha/C_heat: 0.26[1/J], this is the same as the solution!

plt.plot(new_Q_t_vec, new_R_t_vec, 'b', label='measurement')
plt.plot(new_Q_t_vec, R_0 + np.dot(alpha_times_R_0_over_C_heat, new_Q_t_vec), 'orange', label='regression')
plt.legend()
plt.xlabel("Energy [J]")
plt.ylabel("Resistance [Ohm]")
plt.title("Resistance vs. Energy")
plt.grid()
plt.show()