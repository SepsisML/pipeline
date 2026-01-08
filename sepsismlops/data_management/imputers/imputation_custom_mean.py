class CustomMeanImputationStrategy:
    def __init__(self, dataframe, lab_attributes, vital_attributes):
        self.df = dataframe
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes

    def impute(self):
        self.vital_imputation()
        self.lab_imputation()
        self.write_collection("mean-imputation")

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
