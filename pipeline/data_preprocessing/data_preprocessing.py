import pandas as pd
from utils import stratified_shuffle_split, repeated_stratified_k_fold
from .imputers import KNNImputerStrategy
from .imputers import MiceForestImputationStrategy
from .imputers import CustomMeanImputationStrategy
from .imputers import MeanImputationStrategy

from sklearn.model_selection import StratifiedShuffleSplit


class DataPreprocessingStep:
    def __init__(self, input_path, n_splits=5, n_repeats=3, random_state=1, imputation_strategy="custom-mean"):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.imputation_strategy = imputation_strategy

    def load_data(self):
        self.df = pd.read_csv(self.input_path)

    def preprocess_data(self):
        self.load_data()

        # Define lab and vital attributes
        lab_attributes = [
            "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
            "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
            "Fibrinogen", "Platelets"
        ]
        vital_attributes = ["HR", "O2Sat", "Temp",
                            "SBP", "MAP", "DBP", "Resp", "EtCO2"]

        demographic_attributes = ["Age", "ICULOS","Gender"]

        features = lab_attributes + vital_attributes + demographic_attributes

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


        imputer.impute()

        # Split data and prepare cross-validation
        ## Para guardar los indices con el método antiguo:
        # X_train, X_test, y_train, y_test = stratified_shuffle_split(
        #     self.df, features)

        #Para guardar los estados usando el RandomState:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42)
        (train_idx, test_idx) = sss.split(self.df[features], self.df[SepsisLabel])
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        #########################################
        
        cross_validation = repeated_stratified_k_fold(
            self.n_splits, self.n_repeats, self.random_state)

        return X_train, X_test, y_train, y_test, cross_validation
