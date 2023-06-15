import math, statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

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

#%% Part 1: 

print("\nPart 1: ...")