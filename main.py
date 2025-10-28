import yaml
import mlflow
import mlflow.sklearn
import joblib
import subprocess

# Pipeline steps
from pipeline.data_management import DataManagementStep
from pipeline.model_training import ModelTrainingStep
from pipeline.metrics import MetricsStep
# from pipeline.visualization import ImputationPlotter

# Algorithms imports
from pipeline.model_training.models import GradientBoostedDecisionTrees
from pipeline.model_training.models import LightGBMClassifier


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_git_commit_hash():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def prepare_data(config):
    commit_hash = get_git_commit_hash()
    mlflow.set_tag("dvc_git_commit", commit_hash)
    data_processor = DataManagementStep(
        input_path='data/raw/DataPacientes.csv',
        imputation_strategy=config["imputation"]["strategy"]
    )
    mlflow.log_param("imputation_strategy", config["imputation"]["strategy"])
    return data_processor.preprocess_data()


def select_model(config, cross_validation, groups):
    algo_name = config["algorithm"]["training_algorithm"]
    if algo_name == "gbdt":
        return GradientBoostedDecisionTrees(cross_validation=cross_validation, groups=groups, base_params={'random_state': 42})
    if algo_name == "lgbm":
        return LightGBMClassifier(cross_validation=cross_validation,groups=groups, use_gpu=True)
    raise ValueError(f"Unsupported training algorithm: {algo_name}")


def train_and_log_model(X_train, y_train, algorithm, config):
    trainer = ModelTrainingStep(train_data=(
        X_train, y_train), algorithm=algorithm)
    model, selected_params = trainer.train()
    mlflow.log_params(selected_params)
    mlflow.log_metrics({'Validation accuracy': model.best_score_})
    algo_name = config["algorithm"]["training_algorithm"]
    mlflow.sklearn.log_model(model, algo_name)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = MetricsStep(y_pred, X_train, y_train, X_test, y_test)
    f1 = metrics.plot_f1_score()
    mlflow.log_metrics({'f1_score': f1})
    metrics.plot_confusion_matrix()



def main():
    config = load_config()
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["run"]["name"]):
        X_train, X_test, y_train, y_test, cv, groups = prepare_data(config)
        model_class = select_model(config, cross_validation=cv, groups=groups)
        model = train_and_log_model(X_train, y_train, model_class, config)
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
