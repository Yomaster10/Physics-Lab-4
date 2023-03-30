import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% I - V curve heater curent
Heater_Current = 0.25 #A
fh1 = pd.read_csv('Data/FH1.csv', sep='\t', header=5) # read the data

Va1 = np.array(fh1['Va(V)_1']) # accelerating voltage array
I1 = np.array(fh1['Ia(E-12 A)_1']) # Current array
T1 = np.array(fh1['T(c)_1']) #temperature array

plt.figure()
plt.plot(Va1, I1, label='Heater current {:.2f}[A]'.format(Heater_Current))
plt.ylabel('Current [pA]')
plt.xlabel('Acceleration voltage [V]')
plt.grid()
plt.legend()