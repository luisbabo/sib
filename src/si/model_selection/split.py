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