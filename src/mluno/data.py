import numpy as np


def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
    np.random.seed(random_seed)
    X = np.random.uniform(X_low, X_high, n_samples)
    y = beta_0 + beta_1 * X + np.random.normal(0, sd, n_samples)
    return X, y


def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    np.random.seed(random_seed)
    X = np.random.uniform(X_low, X_high, n_samples)
    y = np.sin(X) + np.random.normal(0, sd, n_samples)
    return X, y


def split_data(X, y, holdout_size=0.2, random_seed=None):
    np.random.seed(random_seed)
    n_samples = len(y)
    n_holdout = int(n_samples * holdout_size)
    holdout_idx = np.random.choice(n_samples, n_holdout, replace=False)
    train_idx = np.setdiff1d(np.arange(n_samples), holdout_idx)
    return X[train_idx], y[train_idx], X[holdout_idx], y[holdout_idx]
