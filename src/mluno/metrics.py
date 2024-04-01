import numpy as np


def rmse(y_true, y_pred):
    """
    Compute the root mean squared error
    Parameters
    ----------
    y_true: numpy array
        True target values
    y_pred: numpy array
        Predicted target values
    Returns
    -------
    float
        Root mean squared error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Compute the mean absolute error
    Parameters
    ----------
    y_true: numpy array
        True target values
    y_pred: numpy array
        Predicted target values
    Returns
    -------
    float
        Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


def coverage(y_true, y_pred_lower, y_pred_upper):
    """
    Compute the coverage of the conformal interval
    Parameters
    ----------
    y_true: numpy array
        True target values
    y_pred_lower: numpy array
        Lower bound of the conformal interval
    y_pred_upper: numpy array
        Upper bound of the conformal interval
    Returns
    -------
    float
        Coverage
    """
    return np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))


def sharpness(y_pred_lower, y_pred_upper):
    """
    Compute the sharpness of the conformal interval
    Parameters
    ----------
    y_pred_lower: numpy array
        Lower bound of the conformal interval
    y_pred_upper: numpy array
        Upper bound of the conformal interval
    Returns
    -------
    float
        Sharpness
    """
    return np.mean(y_pred_upper - y_pred_lower)
