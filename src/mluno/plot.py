import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(X, y, regressor, conformal=False, title=""):
    """
    Plot the predictions of a regressor

    Parameters
    ----------
    X: numpy array
        Feature matrix
    y: numpy array
        Target vector
    regressor: object
        Regressor object with a predict method
    conformal: bool
        Whether to plot conformal predictions
    title: str
        Title of the plot
    """
    if conformal:
        y_pred, y_lower, y_upper = regressor.predict(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color="red", s=6)
        plt.plot(X, y_pred, color="blue", linewidth=2)
        # plt.fill_between(X[:, 0], y_lower, y_upper, color="pink", alpha=0.5)
        plt.fill_between(np.ravel(X), np.ravel(y_lower), np.ravel(y_upper), color="pink", alpha=0.5)
    else:
        y_pred = regressor.predict(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color="red", s=6)
        plt.plot(X, y_pred, color="blue", linewidth=2)

    fig, ax = plt.gcf(), plt.gca()
    # set title
    ax.set_title(title)
    fig.show()
    return fig, ax
