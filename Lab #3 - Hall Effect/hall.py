import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
from scipy.stats import linregress

## Functions
def deming_regression(x, y, X_err, Y_err, m=None):
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

    x_err = np.array([X_err]*len(x))
    y_err = np.array([Y_err]*len(y))
    #x_err = np.array([X_err for _ in range(len(x))])
    #y_err = np.array([Y_err for _ in range(len(y))])
    
    # Compute the means of x and y
    x_mean = np.average(x, weights=1/x_err**2)
    y_mean = np.average(y, weights=1/y_err**2)

    # Compute the variances of x and y
    x_var = np.average((x - x_mean)**2, weights=1/x_err**2)
    y_var = np.average((y - y_mean)**2, weights=1/y_err**2)
    #print(x_var, y_var)

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

    return slope, slope_err, y_pred, r_squared

def deming_regression2(x, y, err_x, err_y):
    # Calculate the covariance matrix of x and y
    cov = np.cov(x, y)

    # Calculate the variance of the errors of x and y
    var_err_x = err_x**2
    var_err_y = err_y**2

    # Calculate the slope of the regression line
    slope = (cov[0, 1] * (np.sqrt(var_err_y) / np.sqrt(var_err_x))) / (cov[0, 0] + cov[1, 1] * (var_err_y / var_err_x))

    # Calculate the intercept of the regression line
    intercept = np.mean(y) - slope * np.mean(x)

    # Calculate the error of the slope estimate
    error_slope = np.sqrt(var_err_y / (cov[0, 0] * var_err_x + cov[1, 1] * var_err_y - 2 * cov[0, 1] * np.sqrt(var_err_x * var_err_y)))

    # Calculate the predicted y values
    y_pred = slope * x + intercept

    # Calculate the residual sum of squares and total sum of squares
    RSS = np.sum((y - y_pred)**2)
    TSS = np.sum((y - np.mean(y))**2)

    # Calculate the R-squared value
    r_squared = 1 - (RSS / TSS)

    return slope, error_slope, y_pred, r_squared

def ProductQuotient_Error(F, x, y, x_err, y_err):
    F_err = F * np.sqrt((x_err / x)**2 + (y_err / y)**2)
    return F_err

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

plt.figure(0)
plt.plot(I_p*10**3, U_p, label='Data')
plt.plot(I_p*10**3, U_p_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
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
n = 1 / (R_H*q) #[m^-3]
n_error = ProductQuotient_Error(n, R_H, 1, R_H_error, 0) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e21±{:.3f}e21[m^-3]".format(n*10**-21, n_error*10**-21))

# mobility of the majority charge carriers using:
W = 10*10**-3 #[m]
L = 16*10**-3 #[m]

rho_0 = R_0 * d * W / L #[Ω·m]
rho_error = R_0_error * d * W / L #[Ω·m]

mu = abs(R_H) / rho_0 #[1/T]=[m^2/V·s]
mu_error = ProductQuotient_Error(mu, R_H, rho_0, R_H_error, rho_error) #[m^2/V·s]

print("\tThe resistivity is {:.3f}±{:.3f}[Ω·cm], and the mobility of the majority charge carriers is {:.3f}±{:.3f}[cm^2/V·s]".format(rho_0*10**2, rho_error*10**2, mu*10**4, mu_error*10**4))

plt.figure(1)
plt.plot(I_p*10**3, U_H*10**3, label='Data')
plt.plot(I_p*10**3, U_H_pred*10**3, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("$I_p$ [mA]")
plt.ylabel("$U_H$ [mV]")
plt.title("Hall Voltage vs. Control Current")
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
print(X, X_error)
R_squared = res.rvalue**2
U_H_pred = X*B + res.intercept #[V]

R_H = (d*X)/I #[V·m/T·A]=[m^3/C]
R_H_error = ProductQuotient_Error(R_H, X, I, X_error, I_err) #[m^3/C]

print("\tThe Hall coefficient of the semiconductor is {:.3f}±{:.3f}[cm^3/C]".format(R_H*10**6, R_H_error*10**6))

n = 1 / (R_H*q) #[m^-3]
n_error = ProductQuotient_Error(n, R_H, 1, R_H_error, 0) #[m^-3]
print("\tThe density of the majority charge carriers is {:.3f}e20±{:.3f}e20[m^-3]".format(n*10**-20, n_error*10**-20))

mu = abs(R_H) / rho_0 #[m^2/V·s]
mu_error = ProductQuotient_Error(mu, R_H, rho_0, R_H_error, rho_error) #[m^2/V·s]
print("\tThe mobility of the majority charge carriers is {:.3f}±{:.3f}[cm^2/V·s]".format(mu*10**4, mu_error*10**4))

plt.figure(2)
plt.plot(B*10**3, U_H*10**3, label='Data')
plt.plot(B*10**3, U_H_pred*10**3, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
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
X_error = res.stderr #[K]
R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
E_g_eV_error = 2*k_B*X_error*6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}±{:.3f}[eV]".format(E_g_eV, E_g_eV_error))

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
X_error = res.stderr #[K]
R_squared = res.rvalue**2
y_pred = res.slope*x + res.intercept

k_B = scipy.constants.k #[m^2·kg/s^2·K]
E_g = 2*k_B*X #[J]
E_g_eV = E_g * 6.241509e18 #[eV]
E_g_eV_error = 2*k_B*X_error*6.241509e18 #[eV]
print("\tThe energy gap of the semiconductor is {:.3f}±{:.3f}[eV]".format(E_g_eV, E_g_eV_error))

plt.figure(6)
plt.plot(1/T, np.log(U_H), label='Data')
plt.plot(x, y_pred, label=f'Linear Fit, $R^2$={R_squared:.3f}', linestyle = '--')
plt.xlabel("1/T $[1/K]$")
plt.ylabel("$ln(U_H)$")
plt.title("Natural Logarithm of the Hall Voltage vs. Inverse Temperature")
plt.legend()
plt.grid()
plt.show()