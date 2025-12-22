from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from pymongo import MongoClient

class KNNImputerStrategy:
    def __init__(self, dataframe, lab_attributes, vital_attributes, mongo_uri="mongodb://localhost:27017", db_name="imputation"):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.mongo_uri = mongo_uri
        self.db_name = db_name

    def impute(self):
        # Conecta a MongoDB
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]

        # Si la colección existe, la leemos
        if "knn_imputation" in db.list_collection_names():
            collection = db["knn_imputation"]
            data = list(collection.find({}, {"_id": 0}))  # evita traer el _id
            client.close()
            return pd.DataFrame(data)

        # Si no existe, ejecuta la imputación
        self.knn_impute()

        # Guarda los datos
        self.write_collection("knn_imputation")

        client.close()
        return self.df

    def knn_impute(self):
        for patient_id, group in self.df.groupby("Paciente"):
            vital_imputer = KNNImputer(n_neighbors=3, weights='uniform')
            lab_imputer = KNNImputer(n_neighbors=5, weights='distance')

            # --- Imputación de signos vitales ---
            # Detectar columnas completamente vacías
            vital_all_nan = group[self.vital_attributes].isna().all()
            # Mantener solo las columnas parcialmente completas
            vital_cols_to_impute = vital_all_nan[~vital_all_nan].index.tolist()

            if vital_cols_to_impute:  # Solo imputar si hay columnas válidas
                vital_imputed = vital_imputer.fit_transform(group[vital_cols_to_impute])
                self.df.loc[group.index, vital_cols_to_impute] = vital_imputed

            # --- Imputación de laboratorio ---
            lab_all_nan = group[self.lab_attributes].isna().all()
            lab_cols_to_impute = lab_all_nan[~lab_all_nan].index.tolist()

            if lab_cols_to_impute:  # Solo imputar si hay columnas válidas
                lab_imputed = lab_imputer.fit_transform(group[lab_cols_to_impute])
                self.df.loc[group.index, lab_cols_to_impute] = lab_imputed



    def write_collection(self, collection_name):
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[collection_name]

        # Convierte el DataFrame a diccionarios
        records = self.df.to_dict(orient="records")

        # Inserta en MongoDB
        collection.insert_many(records)
        client.close()
