import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
from scipy.stats import linregress

## Functions
def ProductQuotient_Error(F, x, y, x_err, y_err):
    F_err = F * np.sqrt((x_err / x)**2 + (y_err / y)**2)
    return F_err

def SlopeError(x, y, y_pred):
    x_mean = np.average(x)
    error = np.sqrt((np.sum((1/(len(x)-2)) * (y - y_pred)**2) / np.sum((x - x_mean)**2)))
    return error

def Chi2(Observed, Expected):
    return (Observed - Expected)**2 / Expected

#%% Part 0: Measure I-V (current-voltage) characteristic of the sample
print("\nPart 0: Measure I-V (current-voltage) characteristic of the sample....")

part0data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 0')
I_p = pd.to_numeric(part0data['I_p [mA]'])*10**-3 #[A]
U_p = pd.to_numeric(part0data['U_p [V]']) #[V]

I_err = 5e-4 #[A]
U_err = 5e-5 #[V]

res = linregress(I_p, U_p)
R_0 = res.slope #[Ω]
R_0_error = res.stderr #[Ω]
R_squared = res.rvalue**2
U_p_pred = R_0 *I_p + res.intercept
print("\tThe resistance of the semiconductor is {:.3f}±{:.3f}[Ω]".format(R_0, R_0_error))

chi2_R_0 = Chi2(R_0, 50)

## Initial I-V Curve
plt.figure(0)
plt.plot(I_p*10**3, U_p, label='Data')
plt.plot(I_p*10**3, U_p_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.errorbar(I_p*10**3, U_p, xerr=I_err*10**3, yerr=U_err, fmt='none', color='black', label='Error Bars')
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_p$ [V]")
plt.legend()
plt.grid()

#%% Part 1: Measure Hall voltage vs. control current
print("\nPart 1: Measure Hall voltage vs. control current...")

# Semiconductor Type: P-Type Germanium
B = -251*10**-3 #[T]
d = 1.00*10**-3 #[m]

part1data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 1')
I_p = pd.to_numeric(part1data['I_p [mA]'][:11])*10**-3 #[A]
U_H = pd.to_numeric(part1data['U_H [mV]'][:11])*10**-3 #[V]

res = linregress(I_p, U_H)
X = res.slope #[Ω]
X_error = res.stderr #[Ω]
R_squared = res.rvalue**2
U_H_pred = X*I_p + res.intercept #[V]

R_H = (d*X)/B #[Ω·m/T]=[m^3/C]
B_error = 1e-3 #[T]
R_H_error = ProductQuotient_Error(R_H, X, B, X_error, B_error) #[m^3/C]
print("\tThe Hall coefficient of the semiconductor is {:.3f}±{:.3f}[cm^3/C]".format(R_H*10**6, R_H_error*10**6))

if np.sign(R_H) > 0:
    print("\tHoles are the dominant charge carriers!")
else:
    print("\tElectrons are the dominant charge carriers!")
# Since the density n must be positive, and we got R_H > 0, we see that q > 0 --> holes are dominant!

q = scipy.constants.e #[C]
p = 1 / (R_H*q) #[m^-3]
p_error = ProductQuotient_Error(p, R_H, 1, R_H_error, 0) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e21±{:.3f}e21[m^-3]".format(p*10**-21, p_error*10**-21))

# mobility of the majority charge carriers using:
W = 10*10**-3 #[m]
L = 16*10**-3 #[m]

rho_0 = R_0 * d * W / L #[Ω·m]
rho_error = R_0_error * d * W / L #[Ω·m]

mu = abs(R_H) / rho_0 #[1/T]=[m^2/V·s]
mu_error = ProductQuotient_Error(mu, R_H, rho_0, R_H_error, rho_error) #[m^2/V·s]
print("\tThe resistivity is {:.3f}±{:.3f}[Ω·cm], and the mobility of the majority charge carriers is {:.3f}±{:.3f}[cm^2/V·s]".format(rho_0*10**2, rho_error*10**2, mu*10**4, mu_error*10**4))

chi2_R_H = Chi2(R_H, 4170e-6)
chi2_p = Chi2(p*10**-21, 1.49)
chi2_rho_0 = Chi2(rho_0, 1.750e-2)
chi2_mu = Chi2(mu, 2380e-4)

## Hall Voltage vs. Control Current
plt.figure(1)
plt.plot(I_p*10**3, U_H*10**3, label='Data')
plt.plot(I_p*10**3, U_H_pred*10**3, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.errorbar(I_p*10**3, U_H*10**3, xerr=I_err*10**3, yerr=U_err*10**3, fmt='none', color='black', label='Error Bars')
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_H$ [mV]")
plt.legend()
plt.grid()

#%% Part 2: Measure Hall voltage vs. magnetic field
print("\nPart 2: Measure Hall voltage vs. magnetic field...")

I = -30*10**-3 #[A]

part2data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 2')
B = pd.to_numeric(part2data['B [mT]'][:32])*10**-3 #[T]
U_H = pd.to_numeric(part2data['U_H [mV]'][:32])*10**-3 #[V]

res = linregress(B, U_H) # use linear regression
X = res.slope #[V/T]
X_error = res.stderr #[V/T]

R_squared = res.rvalue**2
U_H_pred = X*B + res.intercept #[V]

R_H = (d*X)/I #[V·m/T·A]=[m^3/C]
R_H_error = ProductQuotient_Error(R_H, X, I, X_error, I_err) #[m^3/C]
print("\tThe Hall coefficient of the semiconductor is {:.3f}±{:.3f}[cm^3/C]".format(R_H*10**6, R_H_error*10**6))

p = 1 / (R_H*q) #[m^-3]
p_error = ProductQuotient_Error(p, R_H, 1, R_H_error, 0) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e20±{:.3f}e20[m^-3]".format(p*10**-20, p_error*10**-20))

mu = abs(R_H) / rho_0 #[m^2/V·s]
mu_error = ProductQuotient_Error(mu, R_H, rho_0, R_H_error, rho_error) #[m^2/V·s]
print("\tThe mobility of the majority charge carriers is {:.3f}±{:.3f}[cm^2/V·s]".format(mu*10**4, mu_error*10**4))

chi2_R_H = Chi2(R_H, 4170e-6)
chi2_p = Chi2(p*10**-21, 1.49)
chi2_mu = Chi2(mu, 2380e-4)

## Hall Voltage vs. Magnetic Field
plt.figure(2)
plt.plot(B*10**3, U_H*10**3, label='Data')
plt.plot(B*10**3, U_H_pred*10**3, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.errorbar(B*10**3, U_H*10**3, xerr=B_error*10**3, yerr=U_err*10**3, fmt='none', color='black', label='Error Bars')
plt.xlabel("$B$ [mT]")
plt.ylabel("$U_H$ [mV]")
plt.legend()
plt.grid()

#%% Part 3: Measure sample voltage vs. magnetic field
print("\nPart 3: Measure sample voltage vs. magnetic field...")

part3data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 3')
B = pd.to_numeric(part3data['B [mT]'][:16])*10**-3 #[T]
U_p = pd.to_numeric(part3data['U_P [V]'][:16]) #[V]

I_p = 30*10**-3 #[A]
R = list(U_p / I_p) #[Ω]
R_error = list(ProductQuotient_Error(R, I_p, U_p, I_err, U_err)) #[Ω]

rho_xx = np.array(R) * d * W / L #[Ω·m]
rho_error = np.array(R_error) * d * W / L #[Ω·m]

res, residuals, _, _, _ = np.polyfit(B, rho_xx, deg=2, full=True)
rho_xx_pred = np.polyval(res, B)

## Resistivity vs. Magnetic Field
plt.figure(3)
plt.plot(B*10**3, rho_xx*10**2, label='Data')
plt.plot(B*10**3, rho_xx_pred*10**2, label=f'2nd Order Polynomial Fit, RSS={residuals[0]*10**9:.3f}e-9', linestyle = '--')
plt.errorbar(B*10**3, rho_xx*10**2, xerr=B_error*10**3, yerr=rho_error*10**2, fmt='none', color='black', label='Error Bars')
plt.xlabel("$B$ [mT]")
plt.ylabel("ρ_xx [Ω·cm]")
plt.ylim(2.6,2.9)
plt.legend()
plt.grid()

#%% Part 4: Measure sample voltage vs. temperature
print("\nPart 4: Measure sample voltage vs. temperature...")

part4data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 4')
T = pd.to_numeric(part4data['T [deg C]']) + 273.15 #[K]
U_p = pd.to_numeric(part4data['U_P [V]']) #[V]

T_limit = 100 + 273.15 #[K]
x = 1/(T[T >= T_limit])
y = np.log(U_p[T >= T_limit])

res = linregress(x, y) # use linear regression
X = res.slope #[K]
X_error = res.stderr #[K]
R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
E_g_eV_error = 2*k_B*X_error*6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}±{:.3f}[eV]".format(E_g_eV, E_g_eV_error))

T_err = 0.5 #[K]
InverseT_error = ProductQuotient_Error(1/T, T, 1, T_err, 0) #[1/K]
InverseUp_error = ProductQuotient_Error(1/U_p, U_p, 1, U_err, 0) #[1/V]
logUp_error = U_err / U_p

chi2_E_g = Chi2(E_g_eV, 0.72)

## Inverse Sample Voltage vs. Inverse Temperature
plt.figure(4)
plt.plot(1/T, 1/U_p, label='Data')
plt.errorbar(1/T, 1/U_p, xerr=InverseT_error, yerr=InverseUp_error, fmt='none', color='black', label='Error Bars')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$1/U_p$ [1/V]")
plt.legend()
plt.grid()

## Natural Logarithm of the Sample Voltage vs. Inverse Temperature
plt.figure(5)
plt.plot(1/T, np.log(U_p), label='Data')
plt.plot(x, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.errorbar(1/T, np.log(U_p), xerr=InverseT_error, yerr=logUp_error, fmt='none', color='black', label='Error Bars')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$ln(U_p)$")
plt.legend()
plt.grid()

#%% Part 5: Measure the Hall voltage vs. temperature
print("\nPart 5: Measure the Hall voltage vs temperature...")

part5data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 5')
T = pd.to_numeric(part5data['T [deg C]']) + 273.15 #[K]
U_H = -pd.to_numeric(part5data['U_H [mV]'])*10**-3 #[V]

#InverseUp_error = ProductQuotient_Error(1/U_H, U_p, 1, U_err, 0) #[1/V]
InverseT_error = ProductQuotient_Error(1/T, T, 1, T_err, 0) #[1/K]
logUH_error = U_err / U_H

T_limit = 100+273.15 #[K]
x = 1/(T[T >= T_limit])
y = np.log(U_H[T >= T_limit])

res = linregress(x, y) # use linear regression
X = res.slope #[K]
X_error = res.stderr #[K]
R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
E_g_eV_error = 2*k_B*X_error*6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}±{:.3f}[eV]".format(E_g_eV, E_g_eV_error))

chi2_E_g = Chi2(E_g_eV, 0.72)

## Natural Logarithm of the Hall Voltage vs. Inverse Temperature
plt.figure(6)
plt.plot(1/T, np.log(U_H), label='Data')
plt.plot(x, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.errorbar(1/T, np.log(U_H), xerr=InverseT_error, yerr=logUH_error, fmt='none', color='black', label='Error Bars')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$ln(U_H)$")
plt.legend()
plt.grid()
plt.show()