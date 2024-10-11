from pipelines import DataProcessingStep
from pipelines import ModelTrainingStep
from pipelines import ModelValidationStep


def main():
    # Process and divide initial dataset
    data_processor = DataProcessingStep(input_path='data/raw/dataset.csv')
    X_train, X_test, y_train, y_test = data_processor.process_data()

    # Train specific model
    model_trainer = ModelTrainingStep(train_data=(X_train, y_train))
    model = model_trainer.execute()

    # Validate model
    model_validator = ModelValidationStep(
        model, X_train, y_train, X_test, y_test)
    model_validator.plot_confusion_matrix()


if __name__ == '__main__':
    main()
