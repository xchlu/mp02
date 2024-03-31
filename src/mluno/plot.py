import matplotlib.pyplot as plt
from conformal import ConformalPredictor


def plot_predictions(X, y, regressor, conformal=False, title=""):
    if conformal:
        conformal_predictor = ConformalPredictor(regressor)
        y_pred = conformal_predictor.predict(X)
        plt.figure(figsize=(10, 6))
        for i, (x, y_true) in enumerate(zip(X, y)):
            plt.plot([x, x], y_pred[i], color="blue", linewidth=2)
            plt.scatter(x, y_true, color="red", s=6)
    else:
        y_pred = regressor.predict(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color="red", s=6)
        plt.plot(X, y_pred, color="blue", linewidth=2)
    plt.title(title)
    plt.show()
