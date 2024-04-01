import numpy as np


class KNNRegressor:
    """
    K-nearest neighbors regressor.
    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. Default is 5.
    """
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fit the model to the data.
        Parameters
        ----------
        X : numpy array
            Feature matrix of shape (n_samples, n_features).
        y : numpy array
            Target vector of shape (n_samples, 1).
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
        Predict the target values for the input data.
        ...
        Parameters
        ----------
        X : numpy array
            Feature matrix of shape (n_samples, n_features).
        Returns
        -------
        y_pred : numpy array
            Predicted target values of shape (n_samples, 1).
        """
        y_pred = []
        for x in X:
            # compute distances to all training points with manhattan distance

            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_y = self.y[nearest_indices]

            y_pred.append(np.mean(nearest_y, axis=0))
        return np.array(y_pred)


class LinearRegressor:
    """
    Linear regression model.
    Attributes
    ----------
    beta : numpy array
        Coefficients of the linear model.
    """

    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        """
        Fit the model to the data.
        Parameters
        ----------
        X : numpy array
            Feature matrix of shape (n_samples, n_features).
        y : numpy array
            Target vector of shape (n_samples, 1).
        """
        X = np.c_[np.ones(len(X)), X]
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        """
        Predict the target values for the input data.
        Parameters
        ----------
        X : numpy array
            Feature matrix of shape (n_samples, n_features).
        Returns
        -------
        y_pred : numpy array
            Predicted target values of shape (n_samples, 1).
        """
        X = np.c_[np.ones(len(X)), X]
        return X.dot(self.beta)


def BinnedRegressor():
    def __init__(self, n_bins=10):
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
