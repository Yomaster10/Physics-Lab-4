import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
from scipy.stats import linregress

#%% Part 0: 
print("\nPart 0: ...")

"""
esra1 = pd.read_csv(f'Data/first_plot.csv', header=1, usecols=['Time (s)', '1 (VOLT)', '2 (VOLT)']) # read the data
t1 = np.array(esra1['Time (s)']) # time array
V1_1 = np.array(esra1['1 (VOLT)']) # accelerating voltage array
V2_1 = np.array(esra1['2 (VOLT)']) # current array
"""

xData = None
yData = None

res = linregress(xData, yData) # use linear regression
print("The slope is {:.3f} \nThe intercept value is {:.3f}".format(res.slope,res.intercept) ) # print the results of the regression

plt.figure(0)
plt.plot(xData, yData, label='')
plt.xlabel("[]]")
plt.ylabel("[]")
plt.legend()
plt.grid()
plt.show()

#%% Part 1: 
print("\nPart 1: ...")
