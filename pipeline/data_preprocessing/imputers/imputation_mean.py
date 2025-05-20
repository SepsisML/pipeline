class MeanImputationStrategy:
    def __init__(self, dataframe, lab_attributes, vital_attributes):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes

    def impute(self):
        self.vital_imputation()
        self.lab_imputation()
        self.write_collection("mean-imputation")

    def vital_imputation(self, df, vital_attributes):
        df[vital_attributes] = df[vital_attributes].interpolate(
            method='linear', limit_direction='both')

    # Función para imputar los resultados de laboratorio

    def lab_imputation(self, df, laboratory_attributes):
        df[laboratory_attributes] = df[laboratory_attributes].fillna(
            method='ffill', limit=12).fillna(method='bfill', limit=12)  # hasta 12 horas de imputación
