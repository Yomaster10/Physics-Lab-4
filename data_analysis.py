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

C_theoretical = (eps0*np.pi*D**2)/4*d

# C_theoretical = 1.1265581811409686e-16

R_tot = 38.4e+3
R=977

tau_theoretical = R_tot*C_theoretical

# tau_theoretical = 4.325983415581319e-12

C_data = pd.read_csv('capacitor.csv')
C_data = C_data.rename(columns = {"time (sec)":"t", "ch2":"V_R"})
C_data["V_C"] = C_data["ch1"] - C_data["V_R"]
t = np.array(C_data['t'].values)
V_C = np.array(C_data['V_C'].values)

plt.plot(t,V_C,label="V_C as a function of time")


# 8. curve fitting: a tool for creating the closest graph created by the data we're putting.
# This graph can indicate the type of the relation between the two variables.

def V_decay(t,tau,V0):
    return V0*np.exp(-t/tau)

p_optimal, p_covariance = cfit(V_decay,C_data['t'], C_data["V_C"]) # non linear curve fitting

plt.plot(p_covariance,p_optimal)





