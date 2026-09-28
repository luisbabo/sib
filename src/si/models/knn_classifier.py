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

    def _get_closest_label(self, sample):
        """
        Encontrar a classe mais comum para uma única amostra.
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest_indexes = np.argsort(distances)[:self.k]
        k_nearest_classes = self.dataset.y[k_nearest_indexes]
        
        labels, counts = np.unique(k_nearest_classes, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset):
        predictions = np.zeros(dataset.shape()[0], dtype=object)
        
        for i in range(dataset.shape()[0]):
            predictions[i] = self._get_closest_label(dataset.X[i])
            
        return predictions

    def _score(self, dataset):
        """
        Calcular a accuracy entre as classes estimadas e reais.
        """
        predictions = self._predict(dataset)
        
        return accuracy(dataset.y, predictions)