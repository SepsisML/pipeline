from pipelines import DataPreprocessingStep
from pipelines import ModelTrainingStep
from pipelines import Metrics
from models import GradientBoostedDecisionTrees
import os
import joblib


def main():
    # Process and divide initial dataset
    data_processor = DataPreprocessingStep(
        input_path='data/raw/SepsisTraining.DataPacientes6-8.csv')
    X_train, X_test, y_train, y_test, cross_validation = data_processor.preprocess_data()

    # Train specific model
    model = GradientBoostedDecisionTrees(cross_validation=cross_validation)
    model_trainer = ModelTrainingStep(
        train_data=(X_train, y_train), model=model)

    model_path = "./sdfsdfsdf.pkl"
    if os.path.exists(model_path):
        print("Cargando el modelo desde el archivo .pkl...")
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
    else:
        model = model_trainer.execute()

    # Se realiza la prediccion con el mejor modelo.
    y_pred = model.predict(X_test)
    model_validator = Metrics(
        y_pred,  X_train, y_train, X_test, y_test)

    model_validator.plot_f1_score()
    model_validator.plot_confusion_matrix()


if __name__ == '__main__':
    main()
