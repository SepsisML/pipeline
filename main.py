import yaml
import mlflow
import mlflow.sklearn
import joblib
import subprocess
import pandas as pd

# Pasos del pipeline
from sepsismlops.data_management import DataManagementStep
from sepsismlops.model_training import ModelTrainingStep
from sepsismlops.metrics import MetricsStep, ImputationEvaluator
# from sepsismlops.visualization import ImputationPlotter
from sepsismlops.data_management.normalizers import min_max_normalize
from sepsismlops.data_management.normalizers import z_score_normalize
from config import LAB_ATTRIBUTES, VITAL_ATTRIBUTES

# Importación de algoritmos
from sepsismlops.model_training.models import GradientBoostedDecisionTrees
from sepsismlops.model_training.models import LightGBMClassifier


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_git_commit_hash():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def evaluate_imputation_quality(df: pd.DataFrame, strategy: str):
    evaluator = ImputationEvaluator(LAB_ATTRIBUTES, VITAL_ATTRIBUTES)
    mlflow_metrics, per_column = evaluator.evaluate(df, strategy)
    mlflow.log_metrics(mlflow_metrics)
    return per_column


def preprocess_pipeline(config):
    commit_hash = get_git_commit_hash()
    mlflow.set_tag("dvc_git_commit", commit_hash)

    strategy = config["imputation"]["strategy"]
    data_processor = DataManagementStep(imputation_strategy=strategy)

    mlflow.log_param("imputation_strategy", strategy)
    df = data_processor.load_data(config["path"]["input_path"])

    evaluate_imputation_quality(df, strategy)

    imputed_df = data_processor.impute_data(df)
    group_df = data_processor.group_data(imputed_df)
    normalized_df = min_max_normalize(group_df)
    X_train, X_test, y_train, y_test, cv, groups = data_processor.split_data(normalized_df)
    return X_train, X_test, y_train, y_test, cv, groups
    
    
    ## Pipeline desde datos ya imputados
    # mlflow.log_param("imputation_strategy", config["imputation"]["strategy"])
    # imputed_df = data_processor.load_data(config["path"]["input_path"])
    # group_df = data_processor.group_data(imputed_df)
    # normalized_df = min_max_normalize(group_df)
    # X_train, X_test, y_train, y_test, cv, groups = data_processor.split_data(normalized_df)
    # return X_train, X_test, y_train, y_test, cv, groups


    ## Pipeline desde IDs de división previos
    # df = pd.read_csv(config["path"]["input_path"])
    # train_ids = pd.read_csv(config["path"]["train_path"])
    # test_ids = pd.read_csv(config["path"]["test_path"])
    # X_train, X_test, y_train, y_test, cross_validation, groups = data_processor.load_split_data(df, train_ids, test_ids)
    # return X_train, X_test, y_train, y_test, cross_validation, groups


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


    ##########################################
    ## Caso de uso: pipeline sepsischallenge-2019
    ## Flujo del pipeline:
        # paso 1: cargar datos,
        # paso 2: imputar datos,
        # paso 3: normalizar datos,
        # paso 4: creación de variables intermedias -> puntajes SIRS y qSOFA
        # paso 5: agrupar pacientes -> pacientes con más de 1 evento de sepsis son agrupados
        # paso 6: dividir datos,
        # paso 7: entrenar modelo,
        # paso 8: evaluar modelo

    config = load_config()
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["run"]["name"]):
        ## Pasos 1-6: preprocesamiento
        X_train, X_test, y_train, y_test, cv, groups = preprocess_pipeline(config)

        ## Paso 7: entrenar modelo
        model_class = select_model(config, cross_validation=cv, groups=groups)
        model = train_and_log_model(X_train, y_train, model_class, config)

        ## Paso 8: evaluar modelo
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
