import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))