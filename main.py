from pipelines import DataPreprocessingStep
from pipelines import ModelTrainingStep
from pipelines import Metrics
from models import GradientBoostedDecisionTrees

import mlflow
import mlflow.sklearn

# Process and divide initial dataset
mlflow.set_experiment("Sepsis Prediction")
with mlflow.start_run():
    data_processor = DataPreprocessingStep(
        input_path='data/raw/SepsisTraining.DataPacientes6-8.csv')
    X_train, X_test, y_train, y_test, cross_validation = data_processor.preprocess_data()

    # Train specific model
    algorithm = GradientBoostedDecisionTrees(
        cross_validation=cross_validation)
    model_trainer = ModelTrainingStep(
        train_data=(X_train, y_train), algorithm=algorithm)
    model, selected_hyperparameters = model_trainer.train()
    mlflow.log_params(selected_hyperparameters)
    mlflow.log_metrics({'accuracy': model.best_score_})

    mlflow.sklearn.log_model(model, "GradientBoostedDecisionTrees")

    # Prediction and evaluation with best model.
    y_pred = model.predict(X_test)
    model_validator = Metrics(
        y_pred,  X_train, y_train, X_test, y_test)
    f1_score = model_validator.plot_f1_score()

    mlflow.log_metrics({'f1_score': f1_score})

    # model_validator.plot_confusion_matrix()
