import pandas as pd
from utils import split_data

# Se debe cambiar nombre  a PreProcessing


class DataProcessingStep:
    def __init__(self, input_path):
        self.input_path = input_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)

    # Organize the dataframe and split the dataset into training and testing sets.

    def process_data(self):
        self.load_data()
        self.df.rename({"Unnamed: 0": "a"}, axis="columns", inplace=True)
        self.df.drop(["a"], axis=1, inplace=True)

        # self.df.Resp = self.df.Resp.replace(["No valido"], -1)
        # df.astype({'Respiracion': 'float64'}).dtypes

        X_train, X_test, y_train, y_test = split_data(self.df, 0.2)
        return X_train, X_test, y_train, y_test
