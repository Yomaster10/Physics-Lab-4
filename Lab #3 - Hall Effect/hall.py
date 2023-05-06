import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
from scipy.stats import linregress

#%% Part 0: Measure I-V (current-voltage) characteristic of the sample.
print("\nPart 0: Measure I-V (current-voltage) characteristic of the sample....")

part0data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 0')
I_p = part0data['I_p [mA]'] #[mA]
U_p = part0data['U_p [V]'] #[V]

res = linregress(I_p, U_p) # use linear regression
R_0 = res.slope*10**3 #[Ω]
print("The resistance of the semiconductor is {:.3f}[Ω]".format(R_0) ) # print the results of the regression

plt.figure(0)
plt.plot(I_p, U_p)
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_p$ [V]")
plt.title("Initial I-V Curve")
plt.grid()
#plt.show()

#%% Part 1: Measure Hall voltage vs. control current
print("\nPart 1: Measure Hall voltage vs. control current...")

# Semiconductor Type: P-Type Germanium
B = -251 #[mT]
d = 1.00 #[mm]

part1data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 1')
I_p = pd.to_numeric(part1data['I_p [mA]'][:11]) #[mA]
U_H = part1data['U_H [mV]'][:11] #[mV]

res = linregress(I_p, U_H) # use linear regression
R = res.slope #[Ω]
R_H = (d*R)/(B*10**-3) #[Ω·mm/T]
print("\tThe Hall coefficient of the semiconductor is {:.3f}[Ω·mm/T]".format(R_H))

if np.sign(R_H) > 0:
    print("\tHoles are the dominant charge carriers!")
else:
    print("\tElectrons are the dominant charge carriers!")
# R_H = 1/nq
# Since the density n must be positive, and we got R_H > 0, we see that q > 0 --> holes are dominant!

q = scipy.constants.e #[C]
n = 1 / ((R_H/1000)*q) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e21[m^-3]".format(n*10**-21))

plt.figure(1)
plt.plot(I_p, U_H)
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_H$ [mV]")
plt.title("Hall Voltage vs. Control Current")
plt.grid()
plt.show()

#%% Part 2: Measure Hall voltage vs. magnetic field
print("\nPart 2: Measure Hall voltage vs. magnetic field...")

#%% Part 3: Measure sample voltage vs. magnetic field
print("\nPart 3: Measure sample voltage vs. magnetic field...")

#%% Part 4: Measure sample voltage vs. temperature
print("\nPart 4: Measure sample voltage vs. temperature...")

#%% Part 5: Measure the Hall voltage vs. temperature
print("\nPart 5: Measure the Hall voltage vs temperature...")
