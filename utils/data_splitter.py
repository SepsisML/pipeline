from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# Se encarga de dividir el dataset inicial en train & test.


def shuffle_split(df, test_size=0.2):
    # Son las variables que usa el modelo para entrenar.
    X = (df.drop(['SepsisLabel', 'Paciente', 'Resp', '_id'], axis=1))
    # Es la variable a predecir.
    y = (df["SepsisLabel"])
    return train_test_split(
        X, y, test_size=test_size, random_state=4, stratify=y)


# Se encarga de dividir el dataset
def repeated_stratified_k_fold(n_splits, n_repeats, random_state):
    cross_validation = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return cross_validation
