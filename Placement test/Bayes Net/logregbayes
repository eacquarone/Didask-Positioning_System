import numpy as np
import pymc as pm, pandas as pd, seaborn

def logistic_setup(df, ind_cols, dep_col):
    '''
        Inputs: pandas Data Frame, list of strings for the independent variables,
        single string for the dependent variable
        Output: PyMC Model
    '''
    
    # model our intercept and error term as above
    b0 = pm.Normal("b0", 0, 0.0001)
    err = pm.Uniform("err", 0, 500)
    
    # initialize a NumPy array to hold our betas 
    # and our observed x values
    b = np.empty(len(ind_cols), dtype=object)
    x = np.empty(len(ind_cols), dtype=object)
    
    # loop through b, and make our ith beta
    # a normal random variable, as in the single variable case
    for i in range(len(b)):
        b[i] = pm.Normal("b" + str(i + 1), 0, 0.0001)
        
    # loop through x, and inform our model about the observed
    # x values that correspond to the ith position
    for i, col in enumerate(ind_cols):
        x[i] = pm.Normal("x" + str(i + 1), 0, 1, value=np.array(df[col]), observed=True)
    
    # as above, but use .dot() for 2D array (i.e., matrix) multiplication
    @pm.deterministic
    def y_pred(b0=b0, b=b, x=x):
        tol = 1e-9
        res = 1.0 / (1. + np.exp(-(b0 + b.dot(x))))
        return np.maximum(np.minimum(res, 1 - tol), tol)
    
    # finally, "model" our observed y values as above
    y = pm.Normal("y", y_pred, err, value=np.array(df[dep_col]), observed=True)
    
    return [b0, pm.Container(b), err, pm.Container(x), y, y_pred]
