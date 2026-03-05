from utils import write_collection, load_collection

class MeanImputationStrategy:
    def __init__(
        self, 
        dataframe, 
        lab_attributes, 
        vital_attributes, 
        patient_column='Paciente',
        load_from_db: bool = False,
        write_in_db: bool = False, 
        collection_name="imputation-mean",
        mongo_uri="mongodb://localhost:27017", 
        db_name="imputation"
    ):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.patient_column = patient_column
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

        self.vital_imputation(self.df, self.patient_column, self.vital_attributes)
        self.lab_imputation(self.df, self.patient_column, self.lab_attributes)

        if self.write_in_db:
            write_collection(self.df, self.mongo_uri, self.db_name, self.collection_name)
            
        return self.df

    def vital_imputation(self, df, patient_column, vital_attributes):
        # Interpolación lineal por paciente
        df[vital_attributes] = (
            df.groupby(patient_column)[vital_attributes]
            .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
            .reset_index(level=0, drop=True)
        )

    def lab_imputation(self, df, patient_column, lab_attributes):
        # forward-fill y back-fill con límite de 12 horas por paciente
        df[lab_attributes] = (
            df.groupby(patient_column)[lab_attributes]
            .apply(lambda group: group.fillna(method='ffill', limit=12).fillna(method='bfill', limit=12))
            .reset_index(level=0, drop=True)
        )