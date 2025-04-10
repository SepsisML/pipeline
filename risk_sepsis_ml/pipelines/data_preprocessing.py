import pandas as pd
import numpy as np
import miceforest as mf
from utils import stratified_shuffle_split, repeated_stratified_k_fold
from sklearn.impute import KNNImputer
from pymongo import MongoClient


class DataPreprocessingStep:
    def __init__(self, input_path, n_splits=5, n_repeats=3, random_state=1, imputation_strategy="custom-mean"):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        # half_length = len(self.df) // 32
        # self.df = self.df.iloc[:half_length]

    # Organize the dataframe and split the dataset into training and testing sets.

    def preprocess_data(self):
        # Loads data from csv file
        self.load_data()
        # Defines lab attr and vital attr
        lab_attributes = [
            "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
            "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
            "Fibrinogen", "Platelets"
        ]
        vital_attributes = ["HR", "O2Sat",
                            "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

        features = lab_attributes + vital_attributes

        # Imputes missing data with specified strategy and create
        if (self.imputation_strategy == "knn"):
            self.knn_imputation(dataframe=self.df,
                                lab_attributes=lab_attributes, vital_attributes=vital_attributes)
        elif (self.imputation_strategy == "miceforest"):
            self.miceforest_imputation(
                dataframe=self.df, lab_attributes=lab_attributes, vital_attributes=vital_attributes)
        elif (self.imputation_strategy == "custom-mean"):
            self.mean_imputation(
                dataframe=self.df, lab_attributes=lab_attributes, vital_attributes=vital_attributes)

        # Splits data (30 % evaluation - 70% training), saves partitions index.
        X_train, X_test, y_train, y_test = stratified_shuffle_split(
            self.df, features)

        # CV stategy
        cross_validation = repeated_stratified_k_fold(
            self.n_splits, self.n_repeats, self.random_state)

        return X_train, X_test, y_train, y_test, cross_validation

    def knn_imputation(self, dataframe, lab_attributes, vital_attributes):
        lab_cols = lab_attributes
        vital_cols = vital_attributes

        # Crear imputadores con diferentes configuraciones
        imputer_5h = KNNImputer(n_neighbors=12).set_output(
            transform='pandas')  # KNN con 5 vecinos
        imputer_12h = KNNImputer(n_neighbors=2).set_output(
            transform='pandas')  # KNN con 12 vecinos
        dataframe[lab_cols] = dataframe[lab_cols].replace(-9999, np.nan)
        dataframe[vital_cols] = dataframe[vital_cols].replace(-9999, np.nan)
        # Aplicar imputadores por separado
        dataframe[lab_cols] = imputer_5h.fit_transform(dataframe[lab_cols])
        dataframe[vital_cols] = imputer_12h.fit_transform(
            dataframe[vital_cols])

        self.write_collection(dataframe, "knn_imputation")

    def miceforest_imputation(self, dataframe, lab_attributes, vital_attributes):
        lab_cols = lab_attributes
        vital_cols = vital_attributes

        dataframe[lab_cols] = dataframe[lab_cols].replace(-9999, np.nan)
        dataframe[vital_cols] = dataframe[vital_cols].replace(-9999, np.nan)
        # Create kernel for lab vars
        lab_attributes_kernel = mf.ImputationKernel(
            dataframe[lab_cols],
            random_state=1991
        )
        # Create kernel for vital vars
        vital_attributes_kernel = mf.ImputationKernel(
            dataframe[vital_cols],
            random_state=1991
        )

        # Run the MICE algorithm for 2 iterations
        lab_attributes_kernel.mice(2)
        vital_attributes_kernel.mice(2)

        # Return the completed dataset.
        dataframe[lab_cols] = lab_attributes_kernel.complete_data()
        dataframe[vital_cols] = vital_attributes_kernel.complete_data()
        self.write_collection(dataframe, "miceforest")

    def mean_imputation(self, dataframe, lab_attributes, vital_attributes):
        # X_train[lab_cols] = X_train[lab_cols].replace(-9999, np.nan)
        # X_train[vital_cols] = X_train[vital_cols].replace(-9999, np.nan)
        self.vital_imputation(dataframe, vital_attributes)
        self.lab_imputation(dataframe, lab_attributes)
        self.write_collection(dataframe, "mean-imputation")

    def vital_imputation(self, df, vital_attributes):
        """
        Imputa valores faltantes (-9999) en las columnas de signos vitales.
        Considera que la imputación se hace solo si el dato superior e inferior
        pertenecen al mismo paciente y al mismo día y no sean -9999, después de eso saca el promedio.

        Parámetros:
        df: DataFrame con los datos de hospitalización.
        vital_attributes: Lista con los nombres de las columnas de signos vitales.

        Retorna:
        DataFrame con los valores imputados.
        """
        for row in range(0, len(df)):  # Evitamos la primera y última fila
            for col in vital_attributes:  # Iteramos sobre los signos vitales
                if row == 0 and df.loc[row, col] == -9999:
                    df.loc[row, col] = 0
                elif row == len(df) - 1 and df.loc[row, col] == -9999:
                    df.loc[row, col] = 0

                # Si hay un valor faltante
                elif df.loc[row, col] == -9999 and row != 0 and row != len(df) - 1:
                    print("La row vale ", row)
                    # Verificar si la fila anterior y la siguiente son del mismo paciente y día
                    same_patient_prev = df.loc[row,
                                               "Paciente"] == df.loc[row - 1, "Paciente"]
                    same_patient_next = df.loc[row,
                                               "Paciente"] == df.loc[row + 1, "Paciente"]

                    same_day_prev = df.loc[row,
                                           "Day"] == df.loc[row - 1, "Day"]
                    same_day_next = df.loc[row,
                                           "Day"] == df.loc[row + 1, "Day"]

                    inferior = df.loc[row - 1,
                                      col] if same_patient_prev and same_day_prev and df.loc[row - 1, col] != -9999 else None
                    superior = df.loc[row + 1,
                                      col] if same_patient_next and same_day_next and df.loc[row + 1, col] != -9999 else None

                    if inferior is not None and superior is not None:
                        df.loc[row, col] = (inferior + superior) / 2
                    else:
                        df.loc[row, col] = 0

        return df

    # Función para imputar los resultados de laboratorio

    def lab_imputation(self, df, laboratory_attributes):
        """
        Imputa valores faltantes (-9999) en las columnas de variables de laboratorio.
        La imputación considera que los datos pertenezcan al mismo paciente y mismo día.
        Para un día se tienen en cuenta todos los valores de laboratorio distintos de nulo,
        se saca un promedio y este es el que imputará los valores faltantes.

        Parámetros:
        df: DataFrame con los datos de hospitalización.
        vital_attributes: Lista con los nombres de las columnas de valores de laboratorio.

        Retorna:
        DataFrame con los valores imputados.
        """
        # Iterar sobre cada atributo de laboratorio
        for col in laboratory_attributes:
            # Iterar sobre cada día y paciente
            for (Paciente, day), group in df.groupby(['Paciente', 'Day']):
                # Filtrar los valores existentes (distintos de -9999)
                valores_existentes = group[col][group[col] != -9999]

                # Calcular el promedio de los valores existentes
                if len(valores_existentes) > 0:
                    promedio = valores_existentes.mean()
                else:
                    promedio = 0  # Si no hay valores existentes, usar 0

                # Reemplazar los valores faltantes (0) con el promedio
                df.loc[(df['Paciente'] == Paciente) &
                       (df['Day'] == day) &
                       (df[col] == -9999), col] = promedio

        return df

    def write_collection(self, dataframe, collection_name):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['SepsisTraining']
        collection = db[collection_name]
        collection.drop()
        print("El df a escribir es ", dataframe)
        collection.insert_many(dataframe.to_dict(orient='records'))
