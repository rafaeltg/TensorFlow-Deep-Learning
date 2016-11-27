import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred):
    """ Root Mean Squared Error	"""

    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """ Mean Absolute Percentage Error """

    return np.mean(np.absolute((y_pred - y_true) / y_true))


available_metrics = {
    'accuracy': metrics.accuracy_score,
    'rmse': rmse,
    'mse': metrics.mean_squared_error,
    'mae': metrics.mean_absolute_error,
    'mape': mape,
    'variance': metrics.explained_variance_score,
}
