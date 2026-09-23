import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = dataset.shape()[0]
    
    n_test = int(n_samples * test_size)
    
    permutations = np.random.permutation(n_samples)
    
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]
    
    train_x = dataset.X[train_idxs]
    train_y = dataset.y[train_idxs] if dataset.has_label() else None
    
    test_x = dataset.X[test_idxs]
    test_y = dataset.y[test_idxs] if dataset.has_label() else None
    
    train_dataset = Dataset(X=train_x, y=train_y, features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=test_x, y=test_y, features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    if random_state is not None:
        np.random.seed(random_state)
        
    # 1. Obter as classes únicas e a contagem de amostras de cada uma
    labels, counts = np.unique(dataset.y, return_counts=True)
    
    # 2. Inicializar listas vazias para os índices de treino e teste
    train_idxs = []
    test_idxs = []
    
    # 3. Iterar sobre as classes únicas
    for label, count in zip(labels, counts):
        # 4. Calcular o número de amostras de teste para a classe atual
        n_test = int(count * test_size)
        
        # Obter todos os índices do dataset onde a classe corresponde à classe atual
        class_idxs = np.where(dataset.y == label)[0]
        
        # 5. Baralhar e selecionar os índices para teste
        permutations = np.random.permutation(class_idxs)
        test_idxs.extend(permutations[:n_test])
        
        # 6. Adicionar os restantes índices aos de treino
        train_idxs.extend(permutations[n_test:])
        
    # 7. Após o ciclo, extrair os dados e criar os Datasets de treino e teste
    train_x = dataset.X[train_idxs]
    train_y = dataset.y[train_idxs]
    
    test_x = dataset.X[test_idxs]
    test_y = dataset.y[test_idxs]
    
    train_dataset = Dataset(X=train_x, y=train_y, features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=test_x, y=test_y, features=dataset.features, label=dataset.label)
    
    # 8. Retornar os datasets de treino e teste
    return train_dataset, test_dataset