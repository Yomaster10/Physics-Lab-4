import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
from scipy.stats import linregress

#%% Part 0: Measure I-V (current-voltage) characteristic of the sample
print("\nPart 0: Measure I-V (current-voltage) characteristic of the sample....")

def deming_regression(x, y, x_err, y_err, m=None):
    """
    Perform Deming regression of y on x. Credit: ChatGPT

    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    x_err : array_like
        Measurement errors on x.
    y_err : array_like
        Measurement errors on y.
    m : float, optional
        Known value of the slope of the true regression line.

    Returns
    -------
    slope : float
        Slope of the estimated regression line.
    slope_err : float
        Error on the slope estimate.
    intercept : float
        Intercept of the estimated regression line.
    r_squared : float
        R-squared value of the fit.
    """

    if m is None:
        m = np.cov(x, y)[0, 1] / np.var(x)

    x_err = np.array([x_err]*len(x))
    y_err = np.array([y_err]*len(y))

    # Compute the means of x and y
    x_mean = np.average(x, weights=1/x_err**2)
    y_mean = np.average(y, weights=1/y_err**2)

    # Compute the variances of x and y
    x_var = np.average((x - x_mean)**2, weights=1/x_err**2)
    y_var = np.average((y - y_mean)**2, weights=1/y_err**2)

    # Compute the covariance between x and y
    xy_cov = np.average((x - x_mean) * (y - y_mean), weights=1/(x_err * y_err))

    # Compute the total variance of the errors
    sigma_sq = (y_var - m**2 * x_var + 2 * m * xy_cov) / (1 + m**2)

    # Compute the estimated slope and intercept of the regression line
    slope = np.sqrt(y_var / x_var) * np.sign(m)
    slope_err = np.sqrt(sigma_sq / x_var)
    intercept = y_mean - slope * x_mean

    # Compute the R-squared value of the fit
    y_pred = slope * x + intercept
    ss_tot = np.sum(((y - y_mean) / y_err)**2)
    ss_res = np.sum(((y - y_pred) / y_err)**2)
    r_squared = 1 - ss_res / ss_tot

    return slope, slope_err, intercept, r_squared

part0data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 0')
I_p = pd.to_numeric(part0data['I_p [mA]'])*10**-3 #[A]
U_p = pd.to_numeric(part0data['U_p [V]']) #[V]

#res = linregress(I_p, U_p) # use linear regression
#R_0 = res.slope #[Ω]
#print("\tThe resistance of the semiconductor is {:.3f}[Ω]".format(R_0) ) # print the results of the regression

#R_squared = res.rvalue**2
#U_p_pred = res.slope*I_p + res.intercept
#print(res.slope, res.stderr, res.intercept)

Ip_err = 5e-4 #[A]
Up_err = 5e-5 #[V]
#Ip_err = np.array([5e-4]*len(I_p)) #[A]
#Up_err = np.array([5e-5]*len(U_p)) #[V]
R_0, R_0_err, intercept, r_squared = deming_regression(I_p, U_p, Ip_err, Up_err)

#print(R_0_err)

print("\tThe resistance of the semiconductor is {:.3f}[Ω]".format(R_0))
print(R_0, R_0_err, intercept, r_squared)
#R_squared2 = res.rvalue**2
U_p_pred2 = R_0*I_p + intercept

plt.figure(0)
plt.plot(I_p*10**3, U_p, label='Data')
#plt.plot(I_p*10**3, U_p_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.plot(I_p*10**3, U_p_pred2, label=f'Deming Regression, $R^2$={r_squared:.3f}', linestyle = '--')
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_p$ [V]")
plt.title("Initial I-V Curve")
plt.legend()
plt.grid()

#%% Part 1: Measure Hall voltage vs. control current
print("\nPart 1: Measure Hall voltage vs. control current...")

# Semiconductor Type: P-Type Germanium
B = -251*10**-3 #[T]
d = 1.00*10**-3 #[m]

part1data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 1')
I_p = pd.to_numeric(part1data['I_p [mA]'][:11]) #[mA]
U_H = pd.to_numeric(part1data['U_H [mV]'][:11]) #[mV]

res = linregress(I_p, U_H) # use linear regression
R = res.slope #[Ω]
R_H = (d*R)/B #[Ω·m/T]=[m^3/C]
print("\tThe Hall coefficient of the semiconductor is {:.3f}[cm^3/C]".format(R_H*10**6))

R_squared = res.rvalue**2
U_H_pred = res.slope*I_p + res.intercept

if np.sign(R_H) > 0:
    print("\tHoles are the dominant charge carriers!")
else:
    print("\tElectrons are the dominant charge carriers!")
# R_H = 1/nq
# Since the density n must be positive, and we got R_H > 0, we see that q > 0 --> holes are dominant!

q = scipy.constants.e #[C]
n = 1 / (R_H*q) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e21[m^-3]".format(n*10**-21))

# mobility of the majority charge carriers using:
W = 10*10**-3 #[m]
L = 16*10**-3 #[m]

rho_0 = R_0 * d * W / L #[Ω·m]
mu = abs(R_H) / rho_0 #[1/T]=[m^2/V·s]
print("\tThe resistivity is {:.3f}[Ω·cm], and the mobility of the majority charge carriers is {:.3f}[cm^2/V·s]".format(rho_0*10**2, mu*10**4))

plt.figure(1)
plt.plot(I_p, U_H, label='Data')
plt.plot(I_p, U_H_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_H$ [mV]")
plt.title("Hall Voltage vs. Control Current")
plt.legend()
plt.grid()

#%% Part 2: Measure Hall voltage vs. magnetic field
print("\nPart 2: Measure Hall voltage vs. magnetic field...")

I = -30*10**-3 #[A]

part2data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 2')
B = pd.to_numeric(part2data['B [mT]'][:32]) #[mT]
U_H = pd.to_numeric(part2data['U_H [mV]'][:32]) #[mV]

res = linregress(B, U_H) # use linear regression
X = res.slope #[V/T]
R_H = (d*X)/I #[V·m/T·A]=[m^3/C]
print("\tThe Hall coefficient of the semiconductor is {:.3f}[cm^3/C]".format(R_H*10**6))

R_squared = res.rvalue**2
U_H_pred = res.slope*B + res.intercept

n = 1 / (R_H*q) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e20[m^-3]".format(n*10**-20))

mu = abs(R_H) / rho_0 #[1/T]=[m^2/V·s]
print("\tThe mobility of the majority charge carriers is {:.3f}[cm^2/V·s]".format(mu*10**4))

plt.figure(2)
plt.plot(B, U_H, label='Data')
plt.plot(B, U_H_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("$B$ [mT]")
plt.ylabel("$U_H$ [mV]")
plt.title("Hall Voltage vs. Magnetic Field")
plt.legend()
plt.grid()

#%% Part 3: Measure sample voltage vs. magnetic field
print("\nPart 3: Measure sample voltage vs. magnetic field...")

part3data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 3')
B = pd.to_numeric(part3data['B [mT]'][:16]) #[mT]
U_p = pd.to_numeric(part3data['U_P [V]'][:16]) #[V]

I_p = 30*10**-3 #[A]

R = U_p / I_p #[Ω]
rho = R * d * W / L #[Ω·m]

res, residuals, _, _, _ = np.polyfit(B*10**-3, rho, deg=2, full=True)
rho_pred = np.polyval(res, B*10**-3)

## WORK IN PROGESS
#print(res[2])

sigma = 1/res[2]
sigma_0 = 1/rho_0 

#print(sigma, sigma_0)
#res = linregress(B, rho) # use linear regression
#X = res.slope #[V/T]
#R_H = (d*X)/I #[V·m/T·A]=[m^3/C]
#R_squared = res.rvalue**2
#print(res[0])
rho_pred2 = np.polyval(res, -251*10**-3) 
#print(rho_pred2)

##

plt.figure(3)
plt.plot(B, rho*10**2, label='Data')
plt.plot(B, rho_pred*10**2, label=f'2nd Order Polynomial Fit, RSS={residuals[0]*10**9:.3f}e-9', linestyle = '--')
plt.xlabel("$B$ [mT]")
plt.ylabel("ρ [Ω·cm]")
plt.title("Resistivity vs. Magnetic Field")
plt.legend()
plt.grid()

#%% Part 4: Measure sample voltage vs. temperature
print("\nPart 4: Measure sample voltage vs. temperature...")

part4data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 4')
T = pd.to_numeric(part4data['T [deg C]']) + 273.15 #[K]
U_p = pd.to_numeric(part4data['U_P [V]']) #[V]

T_limit = 100+273.15 #[K]
x = 1/(T[T >= T_limit])
y = np.log(U_p[T >= T_limit])

res = linregress(x, y) # use linear regression
X = res.slope #[K]
k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}[eV]".format(E_g_eV))

R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

plt.figure(4)
plt.plot(1/T, 1/U_p)
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$1/U_p$ [1/V]")
plt.title("Inverse Sample Voltage vs. Inverse Temperature")
plt.grid()

plt.figure(5)
plt.plot(1/T, np.log(U_p), label='Data')
plt.plot(x, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$ln(U_p)$")
plt.title("Natural Logarithm of the Sample Voltage vs. Inverse Temperature")
plt.legend()
plt.grid()

#%% Part 5: Measure the Hall voltage vs. temperature
print("\nPart 5: Measure the Hall voltage vs temperature...")

part5data = pd.read_excel('Data/HallData.xlsx', sheet_name='Part 5')
T = pd.to_numeric(part5data['T [deg C]']) + 273.15 #[K]
U_H = -pd.to_numeric(part5data['U_H [mV]'])*10**-3 #[V]

T_limit = 100+273.15 #[K]
x = 1/(T[T >= T_limit])
y = np.log(U_H[T >= T_limit])

res = linregress(x, y) # use linear regression
X = res.slope #[K]
k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}[eV]".format(E_g_eV))

R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

plt.figure(6)
plt.plot(1/T, np.log(U_H), label='Data')
plt.plot(x, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$ln(U_H)$")
plt.title("Natural Logarithm of the Hall Voltage vs. Inverse Temperature")
plt.legend()
plt.grid()
plt.show()