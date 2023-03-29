# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:49:27 2023

@author: Ofirs
"""

import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
from sklearn.metrics import r2_score # import function that calculates R^2 score

#%% potential

C = 1
a = 1
L = 3
N = 100
coord = np.linspace(-L, L , N) # defines coordinates
coord_x, coord_y = np.meshgrid(coord, coord)

def potential(x,y,a,C):
    V = -C*np.log(np.sqrt((x-a)**2+y**2)/a)+C*np.log(np.sqrt((x+a)**2+y**2)/a)
    return V

V_xy = potential(coord_x, coord_y, a, C)

plt.figure()
plt.pcolormesh(coord_x, coord_y, V_xy)
plt.colorbar()

for i in range(1,10):
   plt.contour(coord_x, coord_y, V_xy, np.sort([-i , 0 , i]), cmap='hot')

# plt.plot(x,V_x,'.', label=”calculated potential”)

#%% capacitor

eps0 = scipy.constants.epsilon_0 # F/m
D = 18e-2 # m
d = 0.5e-3 # m

C_theoretical = (eps0*np.pi*D**2)/(4*d)

# C_theoretical = 4.5e-10

R_tot = 38.4e+3
R=977

tau_theoretical = R_tot*C_theoretical

# tau_theoretical = 1.73e-5

C_data = pd.read_csv('Data/capacitor.csv')
C_data = C_data.rename(columns = {"time (sec)":"t", "ch2":"V_R"})
C_data["V_C"] = C_data["ch1"] - C_data["V_R"]
t = np.asarray(C_data['t'].values)
V_C = np.asarray(C_data['V_C'].values)

# plotting the measured data

plt.figure(1)
"""
locs, labels = plt.xticks()
print(labels)
new_labels = [f'{i:0.2f}' for float(i) in labels]
print(new_labels)
"""
plt.plot(t,V_C,label="measured V_C as a function of time")


# 8. curve fitting: a tool for creating the closest graph created by the data we're putting.
# This graph can indicate the type of the relation between the two variables.

def V_decay(t,tau,V0):
    return V0*np.exp(-t/tau)

# 13. the fit curve does not fit with the measurements because the scale is different
# p0 is giving an initial guess for the parameters for a better fit
# we know that V_initial is around 4 and we know tau_theoretical so we can use them

parameters, covariance = cfit(V_decay,t, V_C,p0=[1.73e-5,4]) # non linear curve fitting

# 11. the p_optimal will give us two parameters we need

V0_fit = parameters[1]
tau_fit = parameters[0]


# V0_fit = 3.985 V
# tau_fit = 2.04e-5 sec

plt.plot(t, V_decay(t,parameters[0],parameters[1]),label='fitted curve')
plt.legend()

# 12. p_covarience will give us the errors with:
    
V0_fit_err = abs(covariance[1][1])

print('this',V0_fit_err)

print('cov',covariance)

plt.figure(2)

plt.plot(t, np.log(V_decay(t,parameters[0],parameters[1])),label='Log(V)(t)')
plt.grid()

inds = (C_data['t'] > 0) & (C_data['t'] < 0.00004)
plt.plot(C_data['t'][inds], np.log(C_data["V_C"][inds]),'.', label='data')

# 20. linear regression is 
print(covariance)
r =  V_C- V_decay(t,parameters[0],parameters[1])

chi2 = sum((r / 0.05) ** 2)
print('chi2',chi2)

# 420 measurements and 418 dof

p_value = scipy.stats.chi2.cdf(chi2, 418)
print('p value',p_value)
'''
covariance = np.array(covariance)

chi2 = r.T @ np.linalg.inv(covariance) @ r
'''

r2=r2_score(V_C,V_decay(t,parameters[0],parameters[1])) 

# R^2 = 0.997 this value is reasonable since we choose the area where the values are the
# closest to the theory


reg = linregress(C_data['t'][inds], np.log(C_data["V_C"][inds]))
print('reg slope',reg.slope)

V0_reg = np.exp(reg.intercept)
tau_reg = -1/reg.slope

print(V0_reg, tau_reg)

# finding the errors

print('1',reg.stderr,reg.intercept_stderr)
 
# reg r2

r2_reg=r2_score(reg,np.log(C_data["V_C"][inds])) 

# final

#C_data['int_V_R'] = scipy.integrate.cumtrapz(V_R, x = t, initial = 0)

plt.figure(3)

plt.plot(t, V_C,label='Delta V')

plt.xlabel("integral of V_R")
plt.ylabel("$\Delta V_C$")
plt.legend()
plt.grid()

# limits of x