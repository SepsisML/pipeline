from models import GradientBoostedDecisionTrees

# Orchestrate model training


class ModelTrainingStep:
    def __init__(self, train_data, params=None):
        self.train_data = train_data
        self.params = params

    def execute(self):
        model = GradientBoostedDecisionTrees()
        X_train, y_train = self.train_data
        model.grid_search(X_train, y_train)
        model.predict()
        model.save_model()
        return model
