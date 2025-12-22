import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import stratified_shuffle_split, repeated_stratified_k_fold
from .imputers import KNNImputerStrategy
from .imputers import MiceForestImputationStrategy
from .imputers import CustomMeanImputationStrategy
from .imputers import MeanImputationStrategy
from config import LAB_ATTRIBUTES, VITAL_ATTRIBUTES, DEMOGRAPHIC_ATTRIBUTES, FEATURES
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold

class DataManagementStep:
    def __init__(self, input_path, n_splits=3, n_repeats=1, random_state=42, imputation_strategy="custom-mean", is_data_imputed=False, is_data_split=False, imputed_path=None, test_path=None, train_path=None):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy
        ##Optional data loading
        self.is_data_imputed = is_data_imputed
        self.imputed_path = imputed_path
        self.is_data_split = is_data_split
        self.test_path = test_path
        self.train_path = train_path

    def load_split_data(self):
        self.load_data()
        train_ids = pd.read_csv(self.train_path)
        test_ids = pd.read_csv(self.test_path)
        self.impute_data()
        mask_train = self.df["Paciente"].isin(train_ids["Paciente"])
        mask_test = self.df["Paciente"].isin(test_ids["Paciente"])
        X_train = self.df.loc[mask_train, FEATURES]
        y_train = self.df.loc[mask_train, "SepsisLabel"]
        X_test  = self.df.loc[mask_test, FEATURES]
        y_test  = self.df.loc[mask_test, "SepsisLabel"]
        groups = self.df.loc[mask_train, "Paciente"]
        
        cross_validation = GroupKFold(
            self.n_splits)

        return X_train, X_test, y_train, y_test, cross_validation, groups 
        

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
    
    def load_imputed_data(self):
        self.df = pd.read_csv(self.imputed_path)


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
        # Resumen por paciente (igual que antes)
        pacientes = df.groupby("Paciente").agg({
            "SepsisLabel": lambda x: int(x.max() >= 1),
            "qsofa_score_partial": lambda x: int((x >= 2).any()),
            "sirs_score": lambda x: int((x >= 2).any())
        }).reset_index()
        
        # Crear columna "Grupo"
        pacientes["Grupo"] = pacientes[["SepsisLabel", "qsofa_score_partial", "sirs_score"]]\
                                .astype(str).agg(''.join, axis=1)
        
        # Hacer merge con el df original → ahora cada fila tendrá su grupo
        df = df.merge(pacientes[["Paciente", "Grupo"]], on="Paciente", how="left")
        return df   
        # mapping = {
        #     "101": "100",
        # }
        
        # pacientes["Grupo"] = pacientes["Grupo"].replace(mapping)


    def plot_binary_groups(self, df):
        frecuencias = df.groupby("Paciente")["Grupo"].max().value_counts().reset_index()
        #frecuencias = df["Grupo"].value_counts().sort_index().reset_index()
        frecuencias.columns = ["Grupo", "Pacientes"]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=frecuencias, x="Grupo", y="Pacientes", palette="Blues_d")
        plt.title("Distribución de grupos binarios en Hospital A")
        plt.xlabel("Grupo (Sepsis, qSOFA≥2, SIRS≥2)")
        plt.ylabel("Cantidad de pacientes")
        plt.show()

    def impute_data(self):

        # Impute missing data based on chosen strategy
        if self.imputation_strategy == "knn":
            imputer = KNNImputerStrategy(
                self.df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "miceforest":
            imputer = MiceForestImputationStrategy(
                self.df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "mean":
            imputer = MeanImputationStrategy(
                self.df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "custom-mean":
            imputer = CustomMeanImputationStrategy(
                self.df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)

        self.df.replace(-9999, np.nan, inplace=True)
        imputer.impute()
    
    def group_data(self):
        self.generate_sirs_score(self.df)
        self.generate_qsofa_partial(self.df)
        self.df = self.group_patients(self.df)
        self.plot_binary_groups(self.df)
    
    def split_data(self):
        group_labels = self.df.groupby("Paciente")["Grupo"].max().reset_index()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

        train_groups_idx, test_groups_idx = next(sss.split(
            group_labels["Paciente"], group_labels["Grupo"]
        ))

        train_groups = group_labels.iloc[train_groups_idx]["Paciente"]
        test_groups = group_labels.iloc[test_groups_idx]["Paciente"]
        
        train_mask = self.df["Paciente"].isin(train_groups)
        test_mask = self.df["Paciente"].isin(test_groups)
        
        X_train, X_test = self.df.loc[train_mask, FEATURES], self.df.loc[test_mask, FEATURES]
        y_train, y_test = self.df.loc[train_mask, "SepsisLabel"], self.df.loc[test_mask, "SepsisLabel"]
        
        self.df.loc[train_mask].to_csv("hospitalA_Train.csv", index=False)
        self.df.loc[test_mask].to_csv("hospitalA_Test.csv", index=False)

        self.plot_binary_groups(self.df.loc[train_mask])
        self.plot_binary_groups(self.df.loc[test_mask])
        
        ##Prepare second split
        groups = self.df.loc[train_mask, "Paciente"]
        cross_validation = GroupKFold(
            self.n_splits)

        return X_train, X_test, y_train, y_test, cross_validation, groups
    
    def preprocess_data(self):
        if self.is_data_split:
            X_train, X_test, y_train, y_test, cross_validation, groups = self.load_split_data()
            return X_train, X_test, y_train, y_test, cross_validation, groups

        ## Imputed data without splitting
        if self.is_data_imputed:
            self.load_imputed_data()
        else:
            self.load_data()
            self.impute_data()
        
        
        self.group_data()

        X_train, X_test, y_train, y_test, cross_validation, groups = self.split_data()
        
        return X_train, X_test, y_train, y_test, cross_validation, groups
