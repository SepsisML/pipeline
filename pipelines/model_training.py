from models import GradientBoostedDecisionTrees

# Orchestrate model training


class ModelTrainingStep:
    def __init__(self, train_data, model, params=None):
        self.train_data = train_data
        self.params = params
        self.model = model

    def execute(self):
        X_train, y_train = self.train_data
        self.model.grid_search(X_train, y_train)
        # self.model.save_model()
        return self.model
