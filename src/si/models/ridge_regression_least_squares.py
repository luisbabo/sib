import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Model):
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Modelo linear de Ridge Regression (método dos Mínimos Quadrados)
        """
        super().__init__(**kwargs)

        self.l2_penalty = l2_penalty
        self.scale = scale
        
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset):
        """
        Estimar os coeficientes theta e theta_zero, a média e o desvio padrão
        """
        X = dataset.X
        n_samples, n_features = dataset.shape()

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1.0
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(n_features)
            self.std = np.ones(n_features)

        X_with_intercept = np.c_[np.ones(n_samples), X]

        penalty_matrix = self.l2_penalty * np.eye(n_features + 1)

        penalty_matrix[0, 0] = 0

        X_T = X_with_intercept.T
        inverse_matrix = np.linalg.inv(X_T.dot(X_with_intercept) + penalty_matrix)
        thetas = inverse_matrix.dot(X_T).dot(dataset.y)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset):
        """
        Prever a variável dependente Y usando os coeficientes estimados.
        """
        X = dataset.X
        n_samples = dataset.shape()[0]

        if self.scale:
            X = (X - self.mean) / self.std

        X_with_intercept = np.c_[np.ones(n_samples), X]

        thetas = np.r_[self.theta_zero, self.theta]
        
        y_pred = X_with_intercept.dot(thetas)

        return y_pred

    def _score(self, dataset: Dataset):
        """
        Calcular o erro MSE entre os valores reais e previstos de Y.
        """
        y_pred = self._predict(dataset)
        
        return mse(dataset.y, y_pred)