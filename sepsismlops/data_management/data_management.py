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
    def __init__(self, n_splits=3, n_repeats=1, random_state=42, imputation_strategy="custom-mean"):
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy
        

    def load_split_data(self, df: pd.DataFrame, train_ids: pd.DataFrame, test_ids: pd.DataFrame):
        imputed_df = self.impute_data(df)
        mask_train = imputed_df["Paciente"].isin(train_ids["Paciente"])
        mask_test = imputed_df["Paciente"].isin(test_ids["Paciente"])
        X_train = imputed_df.loc[mask_train, FEATURES]
        y_train = imputed_df.loc[mask_train, "SepsisLabel"]
        X_test  = imputed_df.loc[mask_test, FEATURES]
        y_test  = imputed_df.loc[mask_test, "SepsisLabel"]
        groups = imputed_df.loc[mask_train, "Paciente"]
        
        cross_validation = GroupKFold(
            self.n_splits)

        return X_train, X_test, y_train, y_test, cross_validation, groups 
        

    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    

    def generate_sirs_score(self, df: pd.DataFrame) -> pd.DataFrame: 
        df['sirs_temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int)
        df['sirs_hr'] = (df['HR'] > 90).astype(int)
        df['sirs_rr'] = (df['Resp'] > 20).astype(int)
        df['sirs_wbc'] = ((df['WBC'] > 12000) | (df['WBC'] < 4000)).astype(int)  # suponiendo que no tienes % bandas
        df['sirs_score'] = df[['sirs_temp', 'sirs_hr', 'sirs_rr', 'sirs_wbc']].sum(axis=1)
        return df

    def generate_qsofa_partial(self, df: pd.DataFrame) -> pd.DataFrame:
        df['qsofa_rr'] = (df['Resp'] >= 22).astype(int)
        df['qsofa_pas'] = (df['SBP'] <= 100).astype(int)
        df['qsofa_score_partial'] = df['qsofa_rr'] + df['qsofa_pas']
        return df
        
    def group_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        pacientes = df.groupby("Paciente").agg({
            "SepsisLabel": lambda x: int(x.max() >= 1),
            "qsofa_score_partial": lambda x: int((x >= 2).any()),
            "sirs_score": lambda x: int((x >= 2).any())
        }).reset_index()
        
        pacientes["Grupo"] = pacientes[["SepsisLabel", "qsofa_score_partial", "sirs_score"]]\
                                .astype(str).agg(''.join, axis=1)
        
        df = df.merge(pacientes[["Paciente", "Grupo"]], on="Paciente", how="left")
        return df   
        # mapping = {
        #     "101": "100",
        # }
        
        # pacientes["Grupo"] = pacientes["Grupo"].replace(mapping)


    def plot_binary_groups(self, df: pd.DataFrame):
        frecuencias = df.groupby("Paciente")["Grupo"].max().value_counts().reset_index()
        #frecuencias = df["Grupo"].value_counts().sort_index().reset_index()
        frecuencias.columns = ["Grupo", "Pacientes"]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=frecuencias, x="Grupo", y="Pacientes", palette="Blues_d")
        plt.title("Distribución de grupos binarios en Hospital A")
        plt.xlabel("Grupo (Sepsis, qSOFA≥2, SIRS≥2)")
        plt.ylabel("Cantidad de pacientes")
        plt.show()

    def impute_data(self, df: pd.DataFrame) -> pd.DataFrame:

        # Impute missing data based on chosen strategy
        if self.imputation_strategy == "knn":
            imputer = KNNImputerStrategy(
                df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "miceforest":
            imputer = MiceForestImputationStrategy(
                df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "mean":
            imputer = MeanImputationStrategy(
                df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
        elif self.imputation_strategy == "custom-mean":
            imputer = CustomMeanImputationStrategy(
                df, LAB_ATTRIBUTES, VITAL_ATTRIBUTES)

        df.replace(-9999, np.nan, inplace=True)
        df = imputer.impute()
        return df
    
    def group_data(self, df: pd.DataFrame):
        sirs_df = self.generate_sirs_score(df)
        qsofa_df = self.generate_qsofa_partial(sirs_df)
        df = self.group_patients(qsofa_df)
        self.plot_binary_groups(df)
        return df
    
    def split_data(self, df: pd.DataFrame):
        group_labels = df.groupby("Paciente")["Grupo"].max().reset_index()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)

        train_groups_idx, test_groups_idx = next(sss.split(
            group_labels["Paciente"], group_labels["Grupo"]
        ))

        train_groups = group_labels.iloc[train_groups_idx]["Paciente"]
        test_groups = group_labels.iloc[test_groups_idx]["Paciente"]
        
        train_mask = df["Paciente"].isin(train_groups)
        test_mask = df["Paciente"].isin(test_groups)
        
        X_train, X_test = df.loc[train_mask, FEATURES], df.loc[test_mask, FEATURES]
        y_train, y_test = df.loc[train_mask, "SepsisLabel"], df.loc[test_mask, "SepsisLabel"]
        
        ## Save train and test data to csv -> improve to save into data/sandbox-'expid'/imputed-'imputation_strategy'/train.csv and test .csv
        #self.df.loc[train_mask].to_csv("hospitalA_Train.csv", index=False)
        #self.df.loc[test_mask].to_csv("hospitalA_Test.csv", index=False)

        self.plot_binary_groups(df.loc[train_mask])
        self.plot_binary_groups(df.loc[test_mask])
        
        ##Prepare second split
        groups = df.loc[train_mask, "Paciente"]
        cross_validation = GroupKFold(
            self.n_splits)

        return X_train, X_test, y_train, y_test, cross_validation, groups
    