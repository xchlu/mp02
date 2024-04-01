import matplotlib.pyplot as plt
from mluno.conformal import ConformalPredictor
from mluno.plot import plot_predictions

from mluno.data import make_line_data
from sklearn.linear_model import LinearRegression

def test_plot_predictions():
    X, y = make_line_data()
    regressor = LinearRegression().fit(X, y)
    regressor.fit(X, y)

    fig, ax = plot_predictions(X, y, regressor, title="Test Plot")

    # Check that the function returns a Figure and Axes object
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that the title of the plot is correct
    assert ax.get_title() == "Test Plot"

def test_plot_predictions_conformal():
    X, y = make_line_data()
    regressor = LinearRegression().fit(X, y)
    regressor.fit(X, y)
    conformal = ConformalPredictor(regressor)
    conformal.fit(X, y)

    fig, ax = plot_predictions(X, y, conformal, conformal=True, title="Test Plot Conformal")

    # Check that the function returns a Figure and Axes object
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that the title of the plot is correct
    assert ax.get_title() == "Test Plot Conformal"