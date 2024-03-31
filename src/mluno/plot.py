import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(X, y, regressor, conformal=False, title=""):
    x_min, x_max = X.min(), X.max()
    x_test = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    y_pred = regressor.predict(x_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", s=6)
    plt.plot(x_test, y_pred, color="red", linewidth=2)
    plt.title(title)
    plt.show()

