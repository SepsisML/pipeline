import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import stratified_shuffle_split, repeated_stratified_k_fold
from .imputers import KNNImputerStrategy
from .imputers import MiceForestImputationStrategy
from .imputers import CustomMeanImputationStrategy
from .imputers import MeanImputationStrategy

from sklearn.model_selection import StratifiedShuffleSplit


class DataPreprocessingStep:
    def __init__(self, input_path, n_splits=3, n_repeats=1, random_state=1, imputation_strategy="custom-mean"):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy

    def load_data(self):
        self.df = pd.read_csv(self.input_path)

    def generate_sirs_score(self, df):
        df['sirs_temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int)
        df['sirs_hr'] = (df['HR'] > 90).astype(int)
        df['sirs_rr'] = (df['Resp'] > 20).astype(int)
        df['sirs_wbc'] = ((df['WBC'] > 12000) | (df['WBC'] < 4000)).astype(int)  # suponiendo que no tienes % bandas
        df['sirs_score'] = df[['sirs_temp', 'sirs_hr', 'sirs_rr', 'sirs_wbc']].sum(axis=1)
    
    def generate_qsofa_partial(self, df):
        df['qsofa_rr'] = (df['Resp'] >= 22).astype(int)
        df['qsofa_pas'] = (df['SBP'] <= 100).astype(int)
        df['qsofa_score_partial'] = df['qsofa_rr'] + df['qsofa_pas']

    def group_patients(self, df):
        pacientes = df.groupby("Paciente").agg({
            "SepsisLabel": lambda x: int(x.max() >= 1),
            "qsofa_score_partial": lambda x: int((x >= 2).any()), ##Punto de corte de escala
            "sirs_score": lambda x: int((x >= 2).any())
        }).reset_index()
        pacientes["Grupo"] = pacientes[["SepsisLabel", "qsofa_score_partial", "sirs_score"]].astype(str).agg(''.join, axis=1)
        
        # mapping = {
        #     "101": "100",
        # }
        
        # pacientes["Grupo"] = pacientes["Grupo"].replace(mapping)
        
        return pacientes

    def plot_binary_groups(self, df):
        frecuencias = df["Grupo"].value_counts().sort_index().reset_index()
        frecuencias.columns = ["Grupo", "Pacientes"]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=frecuencias, x="Grupo", y="Pacientes", palette="Blues_d")
        plt.title("Distribución de grupos binarios en Hospital A")
        plt.xlabel("Grupo (Sepsis, qSOFA≥2, SIRS≥2)")
        plt.ylabel("Cantidad de pacientes")
        plt.show()

    def preprocess_data(self):
        self.load_data()

        # Define lab and vital attributes
        lab_attributes = [
            "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
            "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
            "Fibrinogen", "Platelets"
        ]
        vital_attributes = ["HR", "O2Sat", "Temp",
                            "SBP", "MAP", "DBP", "Resp"]

        demographic_attributes = ["Age", "ICULOS","Gender"]

        engineering_variables = ["qsofa_score_partial", "sirs_score"]

        features = lab_attributes + vital_attributes + demographic_attributes + engineering_variables

        # Impute missing data based on chosen strategy
        if self.imputation_strategy == "knn":
            imputer = KNNImputerStrategy(
                self.df, lab_attributes, vital_attributes)
        elif self.imputation_strategy == "miceforest":
            imputer = MiceForestImputationStrategy(
                self.df, lab_attributes, vital_attributes)
        elif self.imputation_strategy == "mean":
            imputer = MeanImputationStrategy(
                self.df, lab_attributes, vital_attributes)
        elif self.imputation_strategy == "custom-mean":
            imputer = CustomMeanImputationStrategy(
                self.df, lab_attributes, vital_attributes)

        self.df.replace(-9999, np.nan, inplace=True)
        imputer.impute()

        ## Engineering variables creation
        self.generate_sirs_score(self.df)
        self.generate_qsofa_partial(self.df)
        #self.plot_binary_groups(self.group_patients(self.df))
        
        # Split data and prepare cross-validation
        ## Para guardar los indices con el método antiguo:
        # X_train, X_test, y_train, y_test = stratified_shuffle_split(
        #     self.df, features)

        #Para guardar los estados usando el RandomState:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=42)
        X = self.df[features]
        y = self.df["SepsisLabel"]
        (train_idx, test_idx) = next(sss.split(X, y))
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        #########################################
        cross_validation = repeated_stratified_k_fold(
            self.n_splits, self.n_repeats, self.random_state)

        return X_train, X_test, y_train, y_test, cross_validation
