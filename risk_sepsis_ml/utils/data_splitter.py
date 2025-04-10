from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
import json
import os
# Se encarga de dividir el dataset inicial en train & test.


def stratified_shuffle_split(df, features, test_size=0.3, partitions_file='sss_partitions.json'):
    # Son las variables que usa el modelo para entrenar.
    X = df[features]
    # Es la variable a predecir.
    y = (df["SepsisLabel"])
    # Si ya existen particiones guardadas, cargarlas y usarlas
    if os.path.exists(partitions_file):
        with open(partitions_file, "r") as f:
            partitions = json.load(f)
        print("Particiones cargadas desde archivo existente.")
    else:
        # Generar nuevas particiones
        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42)
        partitions = {}

        for train_idx, test_idx in stratified_shuffle_split.split(X, y):
            partitions = {"train_idx": train_idx.tolist(),
                          "test_idx": test_idx.tolist()}

        # Guardar las particiones generadas
        with open(partitions_file, "w") as f:
            json.dump(partitions, f)

        print("Particiones guardadas correctamente en", partitions_file)

    train_idx = partitions["train_idx"]
    test_idx = partitions["test_idx"]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def repeated_stratified_k_fold(n_splits, n_repeats, random_state):
    cross_validation = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return cross_validation
