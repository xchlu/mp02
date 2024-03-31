import numpy as np
import data


class ConformalPredictor:
    def __init__(self, regressor, alpha=0.05):
        self.regressor = regressor
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.X_cal = None
        self.y_cal = None
        self.y_pred = None
        self.y_cal_pred = None
        self.intervals = None
        self.score_threshold = None

    def fit(self, X, y, calibrate_size=0.2):
        self.X_train, self.y_train, self.X_cal, self.y_cal = data.split_data(X, y, calibrate_size)

        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_train)

    def calibrate(self):
        self.y_cal_pred = self.regressor.predict(self.X_cal)
        # compute conformal scores for calibration set
        scores = []
        for y, y_pred in zip(self.y_cal, self.y_cal_pred):
            scores.append(np.abs(y - y_pred))
        # compute the 1 - alpha quantile of the scores
        self.score_threshold = np.percentile(scores, 100 * (1 - self.alpha))

    def predict(self, X):
        self.calibrate()
        self.intervals = []
        for x in X:
            y_pred = self.regressor.predict(x)
            # compute conformal interval
            intervals = [y_pred - self.score_threshold, y_pred + self.score_threshold]
            self.intervals.append(intervals)
        return np.array(self.intervals)
