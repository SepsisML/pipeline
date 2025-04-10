from pipelines import DataPreprocessingStep
from pipelines import ModelTrainingStep
from pipelines import Metrics
from models import GradientBoostedDecisionTrees

import mlflow
import mlflow.sklearn
import yaml
import joblib

# Process and divide initial dataset


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cargar el csv original.
    # Llamar al stratifiedShuffleSplit (70-30)
    # Guardar las del shuffle particiones cuando no existan (training-evaluation dataset).
    # Se selecciona el training.
    ##

    mlflow.set_experiment(config["experiment"]["name"])
    with mlflow.start_run():
        imputation_strategy = config["imputation"]["strategy"]
        data_processor = DataPreprocessingStep(
            input_path='data/raw/SepsisTraining.Day6-8.csv',
            imputation_strategy=imputation_strategy)
        mlflow.log_param("imputation_strategy", imputation_strategy)
        X_train, X_test, y_train, y_test, cross_validation = data_processor.preprocess_data()

        # Train specific model
        training_algorithm = config["algorithm"]["training_algorithm"]
        if (training_algorithm == "GBDT"):
            algorithm = GradientBoostedDecisionTrees(
                cross_validation=cross_validation)

        model_trainer = ModelTrainingStep(
            train_data=(X_train, y_train), algorithm=algorithm)
        model, selected_hyperparameters = model_trainer.train()
        mlflow.log_params(selected_hyperparameters)
        mlflow.log_metrics({'accuracy': model.best_score_})

        mlflow.sklearn.log_model(model, "GradientBoostedDecisionTrees")

        # Model saving
        joblib.dump(model, "gbdt_model.pkl")

        # Prediction and evaluation with best model.
        y_pred = model.predict(X_test)
        model_validator = Metrics(
            y_pred,  X_train, y_train, X_test, y_test)
        f1_score = model_validator.plot_f1_score()

        mlflow.log_metrics({'f1_score': f1_score})

        # model_validator.plot_confusion_matrix()


if __name__ == "__main__":
    # Preprocessing test
    # data_processor = DataPreprocessingStep(
    #     input_path='data/raw/SepsisTraining.Day6-8.csv', imputation_strategy="custom-mean")
    # X_train, X_test, y_train, y_test, cross_validation = data_processor.preprocess_data()
    main()
