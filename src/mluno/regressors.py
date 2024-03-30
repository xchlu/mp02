import numpy as np


def KNNRegressor(self, k=5):
    def __init__(self):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_y = self.y[nearest_indices]

            y_pred.append(np.mean(nearest_y))
        return np.array(y_pred)


def LinearRegressor(self):
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X.dot(self.beta)


def BinnedRegressor(self, n_bins=10):
    def __init__(self):
        self.n_bins = n_bins
        self.bin_edges = None
        self.y_means = None

    def fit(self, X, y):
        self.bin_edges = np.linspace(X.min(), X.max(), self.n_bins + 1)
        self.y_means = np.array(
            [y[(X >= self.bin_edges[i]) & (X < self.bin_edges[i + 1])].mean() for i in range(self.n_bins)])

    def predict(self, X):
        bin_indices = np.digitize(X, self.bin_edges) - 1
        return self.y_means[bin_indices]
