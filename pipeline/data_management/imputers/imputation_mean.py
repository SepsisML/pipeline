class MeanImputationStrategy:
    def __init__(self, dataframe, lab_attributes, vital_attributes, patient_column='Paciente'):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.patient_column = patient_column

    def impute(self):
        self.vital_imputation()
        self.lab_imputation()
        self.write_collection("mean-imputation")

    def vital_imputation(self):
        # Interpolación lineal por paciente
        self.df[self.vital_attributes] = (
            self.df.groupby(self.patient_column)[self.vital_attributes]
            .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
            .reset_index(level=0, drop=True)
        )

    def lab_imputation(self):
        # forward-fill y back-fill con límite de 12 horas por paciente
        self.df[self.lab_attributes] = (
            self.df.groupby(self.patient_column)[self.lab_attributes]
            .apply(lambda group: group.fillna(method='ffill', limit=12).fillna(method='bfill', limit=12))
            .reset_index(level=0, drop=True)
        )
        

    def write_collection(self, name):
        #Los que no se imputaron con los métodos anteriores, se imputan con la media general:
        self.df[self.lab_attributes] = self.df[self.lab_attributes].fillna(self.df[self.lab_attributes].mean())
        self.df[self.vital_attributes] = self.df[self.vital_attributes].fillna(self.df[self.vital_attributes].mean())
