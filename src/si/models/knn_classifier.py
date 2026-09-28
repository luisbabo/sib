import numpy as np
from si.base.model import Model
from si.metrics.accuracy import accuracy

class KNNClassifier(Model):
    def __init__(self, k: int, distance):
        """
        Algoritmo kNN para classificação.
        
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
        Estimar a classe para cada amostra no dataset de teste.
        """
        predictions = np.zeros(dataset.shape()[0], dtype=object)
        
        for i in range(dataset.shape()[0]):
            sample = dataset.X[i]
            
            distances = self.distance(sample, self.dataset.X)
            
            k_nearest_indexes = np.argsort(distances)[:self.k]
            
            k_nearest_classes = self.dataset.y[k_nearest_indexes]
            
            labels, counts = np.unique(k_nearest_classes, return_counts=True)
            most_common_class = labels[np.argmax(counts)]
            
            predictions[i] = most_common_class
            
        return predictions

    def _score(self, dataset):
        """
        Calcular a accuracy entre as classes estimadas e reais.
        """
        predictions = self._predict(dataset)
        
        return accuracy(dataset.y, predictions)