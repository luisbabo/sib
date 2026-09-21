import numpy as np
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str = ',', label: bool = False) -> Dataset:
    data = np.genfromtxt(filename, delimiter=sep)
    
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None
        
    return Dataset(X=X, y=y)


def write_data_file(filename: str, dataset: Dataset, sep: str = ',', label: bool = False):
    # Juntar o X e o y numa única matriz se a flag label for True e o dataset tiver y
    if label and dataset.has_label():
        # Redimensionar o y para ser uma coluna e juntá-lo horizontalmente (hstack) ao X
        y_reshaped = dataset.y.reshape(-1, 1)
        data = np.hstack((dataset.X, y_reshaped))
    else:
        data = dataset.X
        
    np.savetxt(filename, data, delimiter=sep)