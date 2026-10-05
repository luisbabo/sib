import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse 

class RidgeRegression(Model):
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Modelo linear de Ridge Regression que utiliza Gradient Descent com regularização L2.
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

        #Escalar os dados, se necessário
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            #Evita divisão por zero
            self.std[self.std == 0] = 1.0 
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(n)
            self.std = np.ones(n)

        #Inicializa os coeficientes
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        
        i = 0
        early_stopping = 0
        best_cost = float('inf')

        #Executa até atingir max_iter ou o limite de patience
        while i < self.max_iter and early_stopping < self.patience:
            #Prever os valores de Y com os dados já escalados internamente
            y_pred = np.dot(X, self.theta) + self.theta_zero

            #Calcular os gradientes e atualizar os thetas
            error = y_pred - dataset.y
            
            grad_theta = (1 / m) * np.dot(X.T, error) + (self.l2_penalty / m) * self.theta
            grad_theta_zero = (1 / m) * np.sum(error)

            #Atualiza theta e theta_zero com a taxa de aprendizagem
            self.theta -= self.alpha * grad_theta
            self.theta_zero -= self.alpha * grad_theta_zero

            #Calcular e guardar a função de custo atual
            current_cost = self.cost(dataset)
            self.cost_history[i] = current_cost

            #early stopping
            if current_cost < best_cost:
                best_cost = current_cost
                early_stopping = 0
            else:
                early_stopping += 1

            i += 1

        return self

    def _predict(self, dataset: Dataset):
        """
        Prevê a variável dependente Y usando os coeficientes estimados.
        """
        X = dataset.X
        #Escalar os dados usando a média e std estimados no fit
        if self.scale:
            X = (X - self.mean) / self.std
            
        #Prever Y
        return np.dot(X, self.theta) + self.theta_zero

    def cost(self, dataset: Dataset):
        """
        Calcula a função de custo J com regularização L2.
        """
        #Prever Y
        y_pred = self._predict(dataset)
        m = dataset.shape()[0]
        
        #Calcular a função de custo
        sq_error = np.sum((y_pred - dataset.y) ** 2)
        reg_term = self.l2_penalty * np.sum(self.theta ** 2)
        
        return (sq_error + reg_term) / (2 * m)

    def _score(self, dataset: Dataset):
        """
        Calcula o erro MSE entre os valores reais e previstos.
        """
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)

if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    
    caminho_cpu = "sib/datasets/cpu/cpu.csv"
    cpu_dataset = read_csv(caminho_cpu, sep=",", features=True, label=True)
    
    train_dataset, test_dataset = train_test_split(cpu_dataset, test_size=0.2, random_state=42)
    
    ridge = RidgeRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000, patience=5, scale=True)
    
    ridge.fit(train_dataset)
    
    score_final = ridge.score(test_dataset)
    custo_final = ridge.cost(test_dataset)

    print(f"Score (MSE): {score_final}")
    print(f"Cost: {custo_final}")