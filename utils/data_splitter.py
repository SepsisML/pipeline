from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2):
    # Son las variables que usa el modelo para entrenar.
    X = (df.drop(['SepsisLabel', 'Paciente', 'Resp'], axis=1))
    # Es la variable a predecir.
    y = (df["SepsisLabel"])
    return train_test_split(
        X, y, test_size=test_size, random_state=4, stratify=y)
