import numpy as np


def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
    # generate numpydoc string
    """Generate data from a linear model
    Parameters
    ----------
    n_samples: int
        Number of samples to generate
    beta_0: float
        Intercept of the linear model
    beta_1: float
        Slope of the linear model
    sd: float
        Standard deviation of the noise
    X_low: float
        Lower bound of the feature space
    X_high: float
        Upper bound of the feature space
    random_seed: int
        Random seed for reproducibility
    Returns
    -------
    X: numpy array
        Feature matrix of shape (n_samples, 1)
    y: numpy array
        Target vector of shape (n_samples, 1)
    """
    np.random.seed(random_seed)

    X = np.random.uniform(X_low, X_high, n_samples).reshape(-1, 1)
    y = beta_0 + beta_1 * X + np.random.normal(0, sd, n_samples).reshape(-1, 1)
    return X, y


def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    # generate numpydoc string
    """Generate data from a sine model
    Parameters
    ----------
    n_samples: int
        Number of samples to generate
    sd: float
        Standard deviation of the noise
    X_low: float
        Lower bound of the feature space
    X_high: float
        Upper bound of the feature space
    random_seed: int
        Random seed for reproducibility
    Returns
    -------
    X: numpy array
        Feature matrix of shape (n_samples, 1)
    y: numpy array
        Target vector of shape (n_samples, 1)
    """
    np.random.seed(random_seed)
    X = np.random.uniform(X_low, X_high, n_samples).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(0, sd, n_samples)
    return X, y


def split_data(X, y, holdout_size=0.2, random_seed=None):
    # generate numpydoc string
    """Split data into training and holdout sets
    Parameters
    ----------
    X: numpy array
        Feature matrix of shape (n_samples, n_features)
    y: numpy array
        Target vector of shape (n_samples, 1)
    holdout_size: float
        Proportion of samples to hold out
    random_seed: int
        Random seed for reproducibility
    Returns
    -------
    X_train: numpy array
        Feature matrix of training set
    X_holdout: numpy array
        Feature matrix of holdout set
    y_train: numpy array
        Target vector of training set
    y_holdout: numpy array
        Target vector of holdout set
    """
    np.random.seed(random_seed)
    n_samples = len(y)
    n_holdout = int(n_samples * holdout_size)
    holdout_idx = np.random.choice(n_samples, n_holdout, replace=False)
    train_idx = np.setdiff1d(np.arange(n_samples), holdout_idx)
    return X[train_idx], X[holdout_idx], y[train_idx], y[holdout_idx]
