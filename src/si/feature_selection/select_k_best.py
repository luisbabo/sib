import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest(Transformer):
    def __init__(self, score_func=f_classification, k: int = 10, **kwargs):
        """
        Seleciona as k features com a pontuação mais alta.
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset):
        """
        Estima os valores F e p para cada feature usando a função de pontuação (score_func).
        """
        # A função f_classification devolve um tuplo com (F, p)
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona as top k features com o valor F mais alto e devolve o dataset transformado.
        """
        # O np.argsort devolve os índices que ordenariam o array de forma crescente.
        # Para obter os k maiores valores, seleciona- se os últimos k elementos com o [-self.k:]
        top_k_idxs = np.argsort(self.F)[-self.k:]
        
        # Filtra a matriz X mantendo apenas as colunas correspondentes aos melhores índices
        X_transformed = dataset.X[:, top_k_idxs]
        
        # Filtra os nomes das features (se o dataset original as tiver)
        features_transformed = None
        if dataset.features is not None:
            features_transformed = np.array(dataset.features)[top_k_idxs].tolist()
            
        # Devolve o novo objeto Dataset transformado
        return Dataset(X=X_transformed, y=dataset.y, features=features_transformed, label=dataset.label)