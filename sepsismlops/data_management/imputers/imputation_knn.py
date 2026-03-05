from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from mongo_utils import write_collection, load_collection
class KNNImputerStrategy:
    def __init__(
        self, 
        dataframe, 
        lab_attributes, 
        vital_attributes,
        load_from_db: bool = False,
        write_in_db: bool = False, 
        collection_name="imputation-knn",
        mongo_uri="mongodb://localhost:27017", 
        db_name="imputation"
    ):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.load_from_db = load_from_db
        self.write_in_db = write_in_db
        self.collection_name = collection_name
        self.mongo_uri = mongo_uri
        self.db_name = db_name

    def impute(self):
        if self.load_from_db:
            if not self.collection_name:
                raise ValueError("collection_name es requerido si load_from_db=True")
            return load_collection(self.mongo_uri, self.db_name, self.collection_name)

        self.knn_impute(self.df, self.vital_attributes, self.lab_attributes)
        return self.df

    def knn_impute(self, df, vital_attributes, lab_attributes):
        for patient_id, group in df.groupby("Paciente"):
            vital_imputer = KNNImputer(n_neighbors=3, weights='uniform')
            lab_imputer = KNNImputer(n_neighbors=5, weights='distance')

            # --- Imputación de signos vitales ---
            # Detectar columnas completamente vacías
            vital_all_nan = group[self.vital_attributes].isna().all()
            # Mantener solo las columnas parcialmente completas
            vital_cols_to_impute = vital_all_nan[~vital_all_nan].index.tolist()

            if vital_cols_to_impute:  # Solo imputar si hay columnas válidas
                vital_imputed = vital_imputer.fit_transform(group[vital_cols_to_impute])
                df.loc[group.index, vital_cols_to_impute] = vital_imputed

            # --- Imputación de laboratorio ---
            lab_all_nan = group[self.lab_attributes].isna().all()
            lab_cols_to_impute = lab_all_nan[~lab_all_nan].index.tolist()

            if lab_cols_to_impute:  # Solo imputar si hay columnas válidas
                lab_imputed = lab_imputer.fit_transform(group[lab_cols_to_impute])
                df.loc[group.index, lab_cols_to_impute] = lab_imputed
            
        if self.write_in_db:
            write_collection(self.df, self.mongo_uri, self.db_name, self.collection_name)
        
    

