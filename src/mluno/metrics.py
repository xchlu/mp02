import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def coverage(y_true, y_pred_lower, y_pred_upper):
    return np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))


def sharpness(y_pred_lower, y_pred_upper):
    return np.mean(y_pred_upper - y_pred_lower)
