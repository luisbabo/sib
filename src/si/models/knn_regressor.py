import numpy as np
from si.base.model import Model
from si.metrics.rmse import rmse

class KNNRegressor(Model):
    def __init__(self, k, distance):
        """
        Algoritmo kNN regressão.
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset):
        """
        Guardar o dataset de treino.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        """
        Estima o valor para cada amostra com base na média dos k vizinhos mais próximos.
        """
        # Array para guardar as previsões
        predictions = np.zeros(dataset.shape()[0])
        
        for i in range(dataset.shape()[0]):
            sample = dataset.X[i]
            
            #Calcular a distância entre a amostra e o dataset de treino
            distances = self.distance(sample, self.dataset.X)
            
            #Obter os índices dos k exemplos mais similares
            k_nearest_indexes = np.argsort(distances)[:self.k]
            
            #Recuperar os valores correspondentes no Y de treino
            k_nearest_values = self.dataset.y[k_nearest_indexes]
            
            #Média dos valores obtidos
            predictions[i] = np.mean(k_nearest_values)
            
        return predictions

    def _score(self, dataset):
        """
        Calcular o erro RMSE entre os valores estimados e os reais.
        """
        predictions = self._predict(dataset)
        return rmse(dataset.y, predictions)