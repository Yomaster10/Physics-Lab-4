import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.stats import linregress

#%% Part 1: Plateau Experiment
with open('Data/plateau_talium.tsv') as file:
    plat_voltages = []; plat_counts = []; i=0
    for line in file:
        l=line.split('\n')[0]
        l=l.split('\t')
        if i>10:
            plat_voltages.append(int(l[1]))
            plat_counts.append(int(l[2]))
        i += 1

plt.figure()
plt.plot(plat_voltages, plat_counts)
plt.plot(1020, plat_counts[104], marker='.', ms=10)
plt.xlabel("Voltage [V]")
plt.ylabel("Counts")
plt.grid()
plt.show()

#%% Part 2: Background Measurement

BG_Voltage = 1020 #[V]
BG_Time = 100 #[sec]
BG_Counts = 46 #[counts]

R_b = BG_Counts / BG_Time #[counts/sec]

#%% Part 3: Inverse Squared Law