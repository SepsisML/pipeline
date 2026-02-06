import yaml
import mlflow
import mlflow.sklearn
import joblib
import subprocess
import pandas as pd

# Pipeline steps
from sepsismlops.data_management import DataManagementStep
from sepsismlops.model_training import ModelTrainingStep
from sepsismlops.metrics import MetricsStep
# from sepsismlops.visualization import ImputationPlotter
from sepsismlops.data_management.normalizers import min_max_normalize 
from sepsismlops.data_management.normalizers import z_score_normalize 

# Algorithms imports
from sepsismlops.model_training.models import GradientBoostedDecisionTrees
from sepsismlops.model_training.models import LightGBMClassifier


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_git_commit_hash():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def preprocess_pipeline(config):
    commit_hash = get_git_commit_hash()
    mlflow.set_tag("dvc_git_commit", commit_hash)


    ## Step 1-6: load data,
        # step 2: impute data,
        # step 3: intermediate variables creation, -> SIRS and QSOFA scores (*)
        # step 4: group patients, -> patients with more than 1 sepsis event are grouped (*)
        # step 5: normalize data, (*)
        # step 6: split data, 
    data_processor = DataManagementStep(
        imputation_strategy=config["imputation"]["strategy"],
    )
    ##Complete pipeline
    # mlflow.log_param("imputation_strategy", config["imputation"]["strategy"])
    # df = data_processor.load_data(config["path"]["input_path"])
    # imputed_df = data_processor.impute_data(df)
    # group_df = data_processor.group_data(imputed_df)
    # normalized_df = min_max_normalize(group_df)
    # X_train, X_test, y_train, y_test, cv, groups = data_processor.split_data(normalized_df)
    # return X_train, X_test, y_train, y_test, cv, groups
    
    
    ##Pipeline from imputed data
    # mlflow.log_param("imputation_strategy", config["imputation"]["strategy"])
    # imputed_df = data_processor.load_data(config["path"]["input_path"])
    # group_df = data_processor.group_data(imputed_df)
    # normalized_df = min_max_normalize(group_df)
    # X_train, X_test, y_train, y_test, cv, groups = data_processor.split_data(normalized_df)
    # return X_train, X_test, y_train, y_test, cv, groups


    ##Pipeline from split data
    df = pd.read_csv(config["path"]["input_path"])
    train_ids = pd.read_csv(config["path"]["train_path"])
    test_ids = pd.read_csv(config["path"]["test_path"])
    X_train, X_test, y_train, y_test, cross_validation, groups = data_processor.load_split_data(df, train_ids, test_ids)
    return X_train, X_test, y_train, y_test, cross_validation, groups


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
    ## Use Case: pipeline sepsischallenge-2019 
    ## Pipeline: 
        # step 1: load data, 
        # step 2: impute data,
        # step 3: normalize data, 
        # step 4: intermediate variables creation, -> SIRS and QSOFA scores
        # step 5: group patients, -> patients with more than 1 sepsis event are grouped 
        # step 6: split data, 
        # step 7: train model, 
        # step 8: evaluate model
        
    config = load_config()
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["run"]["name"]):
        ## Step 1-6: load data,
        #  
        X_train, X_test, y_train, y_test, cv, groups = preprocess_pipeline(config)

        ## Step 7: train model
        model_class = select_model(config, cross_validation=cv, groups=groups)
        model = train_and_log_model(X_train, y_train, model_class, config)

        ## Step 8: evaluate model
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
