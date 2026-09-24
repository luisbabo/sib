from scipy import stats

def f_classification(dataset):
    """
    Analisa a variância do dataset agrupando as amostras pelas suas classes.
    """
    classes = dataset.get_classes()
    
    #guardar os grupos de amostras
    groups = []
    
    #Iterar sobre cada classe e extrair as amostras correspondentes
    for c in classes:
        #isto cria uma máscara booleana para filtrar as linhas onde a classe (y) é igual a 'c'
        mask = dataset.y == c
        #depois guards apenas as features (X) dessas linhas específicas
        group_samples = dataset.X[mask]
        groups.append(group_samples)
        
    #calcula os valores F e p usando a função do scipy.stats
    # O * desempacota a lista 'groups' para passar cada grupo como um argumento separado
    F, p = stats.f_oneway(*groups)
    
    #Devolve um tuplo contendo os valores F e os valores p
    return tuple(F), tuple(p)