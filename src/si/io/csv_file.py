import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    if features:
        df = pd.read_csv(filename, sep=sep, header=0)
    else:
        df = pd.read_csv(filename, sep=sep, header=None)

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        features_names = df.columns[:-1].tolist() if features else None
        label_name = str(df.columns[-1]) if features else None
    else:
        X = df.to_numpy()
        y = None
        features_names = df.columns.tolist() if features else None
        label_name = None

    return Dataset(X=X, y=y, features=features_names, label=label_name)

def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False):
    df = dataset.to_dataframe()
    df.to_csv(filename, sep=sep, index=False, header=features)
