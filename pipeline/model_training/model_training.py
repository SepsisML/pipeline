# Orchestrate model training


class ModelTrainingStep:
    def __init__(self, train_data, algorithm, params=None):
        self.train_data = train_data
        self.params = params
        self.algorithm = algorithm

    def train(self):
        X_train, y_train = self.train_data
        best_model, selected_hyperparameters = self.algorithm.grid_search(
            X_train, y_train)
        return best_model, selected_hyperparameters
