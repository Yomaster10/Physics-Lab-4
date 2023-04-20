import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

V_a_err = 0.010 #[V]
def I_err_calc(I):
    return np.dot(0.004,I) + 0.4 #[pA]

def Get_Peaks(I_heater, V_acc, peak_V_lims):
    i = 0; peaks = []; start_idx = None
    while i < len(peak_V_lims)-1:
        end_idx = None
        for v in range(len(V_acc)):
            if V_acc[v] > peak_V_lims[i] and start_idx is None:
                start_idx = v
            if V_acc[v] > peak_V_lims[i+1] and end_idx is None:
                end_idx = v
            if start_idx is not None and end_idx is not None:
                break
        peaks.append(start_idx + np.argmax(I_heater[start_idx:end_idx+1]))
        start_idx = end_idx
        i += 1
    return peaks

def Get_ExcVoltage_and_ContactVoltage(I, V_acc, peak_V_lims, f, add_label=False):
    peak_indices = Get_Peaks(I, V_acc, peak_V_lims)
    I_maxima = [I[p] for p in peak_indices]
    V_maxima = [V_acc[p] for p in peak_indices]
    
    plt.figure(f)
    I_err = I_err_calc(I_maxima)
    if add_label:
        plt.plot(V_maxima, I_maxima, 'bo', label='current peaks')
        plt.errorbar(V_maxima, I_maxima, xerr=V_a_err, yerr=I_err, fmt='none', color='orange', label='errors')
    else:
        plt.plot(V_maxima, I_maxima, 'bo')
        plt.errorbar(V_maxima, I_maxima, xerr=V_a_err, yerr=I_err, fmt='none', color='orange')

    del_V_vec = []; V_contact_vec = []
    for i in range(len(V_maxima)-1):
        del_V = V_maxima[i+1] - V_maxima[i]
        del_V_vec.append(del_V)
        V_contact_vec.append(V_maxima[0] - del_V)

    V_excitation = np.mean(del_V_vec)
    V_contact = np.mean(V_contact_vec)
    return V_excitation, V_contact

#%% Part 1.1: I-V Curve with variable braking voltage (V_r)

print("\nPart 1.1: I-V Curve with Variable Braking Voltage...")

Heater_Current = 0.330 #[A]
peak_V_lims = [5,8,13,18]
outliers = {1.3:[125], 1.4:[507], 1.6:[114,161,402], 1.7:[0], 1.8:[33,34,35,314,470]}

V_e_vec = []; V_c_vec = []; V_r_vec = []
for i in range(1,9):
    Braking_Voltage = round(1 + 0.1*i, 2)
    V_r_vec.append(Braking_Voltage)

    fh1 = pd.read_csv(f'Data/FH1_Vr={Braking_Voltage:0.1f}_Ih=330.csv', sep='\t', header=5) # read the data
    # Removing outliers from the data
    if Braking_Voltage in outliers:
        for j in outliers[Braking_Voltage]:
            fh1.drop(j, inplace=True)

    Va1 = np.array(fh1['Va(V)_1']) # accelerating voltage array
    I1 = np.array(fh1['Ia(E-12 A)_1']) # current array
    T1 = np.array(fh1['T(c)_1']) # temperature array

    add_label = False
    if i == 8:
        add_label = True

    V_excitation, V_contact = Get_ExcVoltage_and_ContactVoltage(I1, Va1, peak_V_lims, 0, add_label=add_label)
    V_e_vec.append(V_excitation)
    V_c_vec.append(V_contact)

    plt.figure(0)
    plt.plot(Va1, I1, label='V_r: {:.1f}[V]'.format(Braking_Voltage))
    
V_e_avg_1 = np.mean(V_e_vec)
V_c_avg_1 = np.mean(V_c_vec)
print(f"\tAvg. Excitation Energy: {V_e_avg_1:0.3f}[eV], Avg. Contact Voltage: {V_c_avg_1:0.3f}[V]")

plt.figure(0)
plt.xlabel('Acceleration Voltage [V]')
plt.ylabel('Current [pA]')
#plt.title('Part 1.1: Current vs. Acceleration Voltage (FH Curve) - I_h={:.2f}[A]'.format(Heater_Current))
plt.grid()
plt.legend()
plt.show()

#%% Part 1.2: I-V Curve with variable heater current (I_h)

print("\nPart 1.2: I-V Curve with Variable Heater Current...")

Braking_Voltage = 1.5 #[V]
Heater_Currents = [0.30, 0.30, 0.31, 0.31, 0.32] #[A]
peak_V_lims = [5,8,13,18]
outliers = {4.9:[44,45,46,47,48,49,51,329,330,331,332,333,335,337], 5.1:[254], 5.3:[262,263,264,265,266,267,344],
            5.5:[99,381,470,471,472,473,474,475,476,477,478,479,480]}

plt.figure(3)
V_e_vec = []; V_c_vec = []; V_h_vec = []
for i in range(5):
    Heater_Voltage = round(4.9 + 0.2*i, 2)
    V_h_vec.append(Heater_Voltage)

    fh2 = pd.read_csv(f'Data/FH2_Vr=1.5_Vh={Heater_Voltage:0.1f}.csv', sep='\t', header=5) # read the data
    # Removing outliers from the data
    if Heater_Voltage in outliers:
        for j in outliers[Heater_Voltage]:
            fh2.drop(j, inplace=True)

    Va2 = np.array(fh2['Va(V)_1']) # accelerating voltage array
    I2 = np.array(fh2['Ia(E-12 A)_1']) # current array
    T2 = np.array(fh2['T(c)_1']) # temperature array

    add_label = False
    if i == 4:
        add_label = True
    V_excitation, V_contact = Get_ExcVoltage_and_ContactVoltage(I2, Va2, peak_V_lims, 3, add_label = add_label)
    V_e_vec.append(V_excitation)
    V_c_vec.append(V_contact)

    plt.figure(3)
    plt.plot(Va2, I2, label='V_h: {:.1f}[V], I_h: {:.2f}[A]'.format(Heater_Voltage, Heater_Currents[i]))
    
V_e_avg_2 = np.mean(V_e_vec)
V_c_avg_2 = np.mean(V_c_vec)
print(f"\tAvg. Excitation Energy: {V_e_avg_2:0.3f}[eV], Avg. Contact Voltage: {V_c_avg_2:0.3f}[V]")

plt.figure(3)
plt.xlabel('Acceleration Voltage [V]')
plt.ylabel('Current [pA]')
#plt.title('Part 1.2: Current vs. Acceleration Voltage (FH Curve) - V_r={:.1f}[V]'.format(Braking_Voltage))
plt.grid()
plt.legend()
plt.show()

#%% Part 2.1: Ionization Energy of Mercury Atoms

print("\nPart 2.1: Ionization Energy of Mercury Atoms...")

fh3 = pd.read_csv(f'Data/FH3_Vr=1.2_Vh=5.4.csv', sep='\t', header=5) # read the data
Va3 = np.array(fh3['Va(V)_1']) # accelerating voltage array
I3 = np.array(fh3['Ia(E-12 A)_1']) # Current array
T3 = np.array(fh3['T(c)_1']) #temperature array

Braking_Voltage = 1.2 #[V]
Heater_Voltage = 5.4 #[V]

nonzero = []
for i in range(len(I3)):
    if I3[i] > 0.5: #cutoff
        nonzero.append(Va3[i])
V_a_star = min(nonzero)
I_a_star = I3[list(Va3).index(V_a_star)]
V_ion = V_a_star-V_c_avg_1

print(f"\tIonization Voltage: {V_ion:0.3f}[V]")

plt.figure(6)
plt.plot(Va3, I3, 'r', label='data')
plt.plot(V_a_star, I_a_star, 'bo', label='ionization point')

I_err = I_err_calc(I_a_star)
plt.errorbar(V_a_star, I_a_star, xerr=V_a_err, yerr=I_err, fmt='none', color='orange', label='error')

plt.xlabel('Acceleration Voltage [V]')
plt.ylabel('Current [pA]')
#plt.title('Part 2.1: Current vs. Acceleration Voltage (FH Curve) - V_r={:.1f}[V]'.format(Braking_Voltage))
plt.grid()
plt.legend()
plt.show()