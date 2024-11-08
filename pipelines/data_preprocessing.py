import pandas as pd
from utils import shuffle_split, repeated_stratified_k_fold


# Se debe cambiar nombre  a PreProcessing


class DataPreprocessingStep:
    def __init__(self, input_path, n_splits=5, n_repeats=3, random_state=1):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        # half_length = len(self.df) // 16
        # self.df = self.df.iloc[:half_length]

    # Organize the dataframe and split the dataset into training and testing sets.

    def preprocess_data(self):
        self.load_data()
        X_train, X_test, y_train, y_test = shuffle_split(self.df)
        cross_validation = repeated_stratified_k_fold(
            self.n_splits, self.n_repeats, self.random_state)

        return X_train, X_test, y_train, y_test, cross_validation
