def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False):
    # A sua classe Dataset já tem um método to_dataframe(), o que facilita o processo
    df = dataset.to_dataframe()
    
    # O pandas guarda o ficheiro; index=False evita guardar os números das linhas
    # O argumento header define se escrevemos ou não os nomes das features
    df.to_csv(filename, sep=sep, index=False, header=features)