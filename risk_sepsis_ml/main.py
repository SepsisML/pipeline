from pipeline import DataPreprocessingStep
from pipeline import ModelTrainingStep
from pipeline import Metrics
from models import GradientBoostedDecisionTrees

import yaml
import mlflow
import mlflow.sklearn
import joblib

from pipelines import DataPreprocessingStep, ModelTrainingStep, Metrics
from models import GradientBoostedDecisionTrees


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config):
    data_processor = DataPreprocessingStep(
        input_path='data/raw/SepsisTraining.Day6-8.csv',
        imputation_strategy=config["imputation"]["strategy"]
    )
    mlflow.log_param("imputation_strategy", config["imputation"]["strategy"])
    return data_processor.preprocess_data()


def select_model(config, cross_validation):
    algo_name = config["algorithm"]["training_algorithm"]
    if algo_name == "GBDT":
        return GradientBoostedDecisionTrees(cross_validation=cross_validation)
    raise ValueError(f"Unsupported training algorithm: {algo_name}")


def train_and_log_model(X_train, y_train, algorithm):
    trainer = ModelTrainingStep(train_data=(
        X_train, y_train), algorithm=algorithm)
    model, selected_params = trainer.train()
    mlflow.log_params(selected_params)
    mlflow.log_metrics({'accuracy': model.best_score_})
    mlflow.sklearn.log_model(model, "GradientBoostedDecisionTrees")
    joblib.dump(model, "gbdt_model.pkl")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = Metrics(y_pred, X_train, y_train, X_test, y_test)
    f1 = metrics.plot_f1_score()
    mlflow.log_metrics({'f1_score': f1})
    # Optional: metrics.plot_confusion_matrix()


def main():
    config = load_config()
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test, cv = prepare_data(config)
        model_class = select_model(config, cross_validation=cv)
        model = train_and_log_model(X_train, y_train, model_class)
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
