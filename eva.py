import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def R2(pred, true):
    # # Calculate the mean of the observed values of the dependent variable
    y_mean = np.mean(true)
    
    # Calculate the total sum of squares (SS_tot)
    ss_tot = np.sum((true - y_mean)**2)
    
    # Calculate the sum of squared residuals (SS_res)
    ss_res = np.sum((true - pred)**2)

    # Calculate R-squared (coefficient of determination)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    r2 = R2(pred, true)
    return mae, mse, rmse, r2
