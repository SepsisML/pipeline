from utils import write_collection, load_collection


class CustomMeanImputationStrategy:
    def __init__(
        self,
        dataframe,
        lab_attributes,
        vital_attributes,
        load_from_db: bool = False,
        write_in_db: bool = False,
        collection_name="imputation-custom-mean",
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

        self.df = self.df.reset_index(drop=True)
        self.df["Day"] = self.df.groupby("Paciente").cumcount() // 24

        self.vital_imputation(self.df, self.vital_attributes)
        self.lab_imputation(self.df, self.lab_attributes)

        if self.write_in_db:
            write_collection(self.df, self.mongo_uri, self.db_name, self.collection_name)

        return self.df

    def vital_imputation(self, df, vital_attributes):
        # Calcular contexto de vecinos una vez, compartido entre todas las columnas
        same_patient_prev = df["Paciente"] == df["Paciente"].shift(1)
        same_patient_next = df["Paciente"] == df["Paciente"].shift(-1)
        same_day_prev = df["Day"] == df["Day"].shift(1)
        same_day_next = df["Day"] == df["Day"].shift(-1)

        for col in vital_attributes:
            mask_missing = df[col].isna()

            prev_val = df[col].shift(1)
            next_val = df[col].shift(-1)

            prev_valid = same_patient_prev & same_day_prev & prev_val.notna()
            next_valid = same_patient_next & same_day_next & next_val.notna()

            # Ambos vecinos válidos: usar su media
            both_valid = mask_missing & prev_valid & next_valid
            df.loc[both_valid, col] = (prev_val[both_valid] + next_val[both_valid]) / 2

            # Resto de valores faltantes: se dejan como NaN para manejo posterior

        return df

    def lab_imputation(self, df, lab_attributes):
        for col in lab_attributes:  
            # Marcar faltantes como NaN para que el groupby los ignore
            df.loc[df[col] == -9999, col] = float("nan")

            # Rellenar cada faltante con la media de valores válidos en el mismo grupo paciente-día
            group_means = df.groupby(["Paciente", "Day"])[col].transform("mean")
            df[col] = df[col].fillna(group_means)

        return df
