import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression(Model):
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Modelo de Regressão Logística que utiliza Gradient Descent com regularização L2.
        """
        super().__init__(**kwargs)
        # Parâmetros
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        
        # Parâmetros estimados
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset):
        """
        Estima os coeficientes theta e theta_zero, a média, o desvio padrão e o histórico de custo.
        """
        X = dataset.X
        m, n = dataset.shape()

        # regularizar dados se necessario
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1.0 
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(n)
            self.std = np.ones(n)

        # Inicializa os coeficientes
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        
        i = 0
        early_stopping = 0
        best_cost = float('inf')

        # 8. Run until max_iter or patience is reached
        while i < self.max_iter and early_stopping < self.patience:
            # 2. Estimates the value of Y using the sigmoid_function
            z = np.dot(X, self.theta) + self.theta_zero
            y_pred = sigmoid_function(z)

            #update the gradient / regularization term L2
            error = y_pred - dataset.y
            
            grad_theta = (1 / m) * np.dot(X.T, error) + (self.l2_penalty / m) * self.theta
            grad_theta_zero = (1 / m) * np.sum(error)

            self.theta -= self.alpha * grad_theta
            self.theta_zero -= self.alpha * grad_theta_zero

            #Compute the cost function
            current_cost = self.cost(dataset)
            self.cost_history[i] = current_cost

            # early stopping
            if current_cost < best_cost:
                best_cost = current_cost
                early_stopping = 0
            else:
                early_stopping += 1

            i += 1

        return self

    def _predict(self, dataset: Dataset):
        """
        Prevê a variável dependente Y em formato binário.
        """
        X = dataset.X
        
        # Escalar os dados
        if self.scale:
            X = (X - self.mean) / self.std
            
        z = np.dot(X, self.theta) + self.theta_zero
        probs = sigmoid_function(z)
        
        return np.where(probs >= 0.5, 1, 0)

    def cost(self, dataset: Dataset):
        """
        Calcula a função de custo J (Log Loss) com regularização L2.
        """
        m = dataset.shape()[0]
        
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        
        cost = dataset.y * np.log(predictions) + (1 - dataset.y) * np.log(1 - predictions)
        cost = np.sum(cost)
        cost = cost * (-1 / m)
        reg_term = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        
        return cost + reg_term

    def _score(self, dataset: Dataset):
        """
        Calcula a exatidão (accuracy) entre os valores reais e previstos.
        """
        y_pred = self._predict(dataset)
        
        return accuracy(dataset.y, y_pred)
    