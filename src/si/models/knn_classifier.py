import numpy as np

class KNNClassifier:
    def __init__(self, k, distance):
        """
        Algoritmo kNN
        """
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
        Estimar a classe para cada uma das amostras com base nos k vizinhos mais próximos.
        """
        predictions = np.zeros(dataset.shape()[0])
        
        for i in range(dataset.shape()[0]):
            sample = dataset.X[i]
            
            #Calcular a distância
            distances = self.distance(sample, self.dataset.X)
            
            #Obter os índices dos k exemplos mais similares
            k_nearest_indexes = np.argsort(distances)[:self.k]
            
            #Recuperar classes
            k_nearest_classes = self.dataset.y[k_nearest_indexes]
            
            #Obter a classe mais comum
            unique_classes, counts = np.unique(k_nearest_classes, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            
            predictions[i] = most_common_class
            
        return predictions

    def _score(self, dataset):
        """
        Calcular accuracy entre as classes estimadas e as reais.
        """
        from si.metrics.accuracy import accuracy
        
        predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)