import numpy as np
import mluno.data as data


class ConformalPredictor:
    """
    Conformal predictor for regression
    Parameters
    ----------
    regressor: sklearn regressor
        Regressor to use for prediction
    alpha: float
        Significance level
    Attributes
    ----------
    X_cal: numpy array
        Feature matrix of the calibration set
    y_cal: numpy array
        Target vector of the calibration set
    y_pred: numpy array
        Predictions of the training set
    y_cal_pred: numpy array
        Predictions of the calibration set
    y_lower: numpy array
        Lower bound of the conformal interval
    y_upper: numpy array
        Upper bound of the conformal interval
    y_pred_test: numpy array
        Predictions of the test set
    scores: numpy array
        Conformal scores
    quantile: float
        1 - alpha quantile of the scores
    """
    def __init__(self, regressor, alpha=0.05):
        self.regressor = regressor
        self.alpha = alpha
        self.X_cal = None
        self.y_cal = None
        self.y_pred = None
        self.y_cal_pred = None
        self.y_lower = None
        self.y_upper = None
        self.y_pred_test = None
        self.scores = None
        self.quantile = None

    def fit(self, X, y):
        """
        Fit the conformal predictor
        Parameters
        ----------
        X: numpy array
            Feature matrix of the training set
        y: numpy array
            Target vector of the training set
        """
        # self.X_train, self.X_cal, self.y_train, self.y_cal = data.split_data(X, y, calibrate_size)
        # self.y_pred = self.regressor.predict(self.X_train)
        self.X_cal = X
        self.y_cal = y
        self.calibrate()

    def calibrate(self):
        self.y_cal_pred = self.regressor.predict(self.X_cal)
        # compute conformal scores for calibration set
        scores = []
        for y, y_pred in zip(self.y_cal, self.y_cal_pred):
            scores.append(np.abs(y - y_pred))
        # compute the 1 - alpha quantile of the scores
        self.quantile = np.percentile(scores, 100 * (1 - self.alpha))
        self.scores = scores

    def predict(self, X):
        """
        Predict the target variable and compute the conformal interval
        Parameters
        ----------
        X: numpy array
            Feature matrix of the test set
        Returns
        -------
        y_pred_test: numpy array
            Predictions of the test set
        y_lower: numpy array
            Lower bound of the conformal interval
        y_upper: numpy array
            Upper bound of the conformal interval
        """
        y_pred = self.regressor.predict(X)
        self.quantile = np.ravel(self.quantile)
        self.y_pred_test = y_pred

        # compute conformal interval
        self.y_lower = y_pred - self.quantile
        self.y_upper = y_pred + self.quantile

        return self.y_pred_test, self.y_lower, self.y_upper
