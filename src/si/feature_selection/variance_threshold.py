import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class VarianceThreshold(Transformer):
    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Transformer que seleciona features com base na sua variância.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset):
        """
        Estimar a variância de cada feature do dataset (coluna a coluna).
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selecionar todas as features com variância superior ao threshold e devolve um novo Dataset.
        """
        #Criar uma máscara booleana onde True significa que a variância é maior que o threshold
        mask = self.variance > self.threshold
        
        #Filtrar a matriz X mantendo apenas as colunas (features) que cumprem a condição
        X_transformed = dataset.X[:, mask]
        
        #Filtrar o nome das features
        features_transformed = None
        if dataset.features is not None:
            #Converte para array numpy para poder aplicar a máscara, depois volta a lista
            features_transformed = np.array(dataset.features)[mask].tolist()
            
        # 4. Devolver um novo objeto Dataset com os dados transformados
        return Dataset(X=X_transformed, y=dataset.y, features=features_transformed, label=dataset.label)