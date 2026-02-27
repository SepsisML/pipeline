import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize


class GradientBoostedDecisionTrees:
    def __init__(self, cross_validation, groups, base_params={}):
        # Inicializa el modelo con los hiperparámetros por defecto
        self.base_params = base_params or {}
        self.model = GradientBoostingClassifier(**base_params)
        self.best_model = None
        self.best_params = None
        self.cross_validation = cross_validation
        self.groups = groups

    def grid_search(self, X_train, y_train, random_state=1):
        """
        Realiza la búsqueda de hiperparámetros mediante GridSearchCV.
        """

        # Define el espacio de búsqueda
        space = {
            'n_estimators': [ 200],
            'learning_rate': [0.03],
            'max_depth': [3, 5],
            'min_samples_leaf': [5, 10],
            'subsample': [0.8, 1.0],
            'max_features': ['sqrt']
        }

        # Realiza la búsqueda de hiperparámetros
        
        search = GridSearchCV(
            self.model,
            space,
            scoring='f1',
            refit='f1',
            n_jobs=6,
            cv=self.cross_validation,
            verbose=1,
            return_train_score=False,
        )
        
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()

        # Peso proporcional inverso: más peso a la clase positiva
        w_pos = n_neg / n_pos
        w_neg = 1

        # Vector de pesos por muestra
        sample_weights = np.where(y_train == 1, w_pos, w_neg)
        
        result = search.fit(X_train, y_train, sample_weight=sample_weights, groups=self.groups)
        # Almacena el mejor modelo y parámetros
        self.best_model = result
        self.best_params = result.best_params_
        return self.best_model, self.best_params

    def predict(self, X_test):
        """
        Realiza la predicción usando el mejor modelo encontrado.
        """
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        y_pred = self.best_model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo utilizando la curva ROC.
        """
        y_test_hot = label_binarize(y_test, classes=(0, 1))
        y_score = self.best_model.decision_function(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_hot.ravel(), y_score.ravel())

        print("FPR: ", fpr)
        print("TPR: ", tpr)

        return fpr, tpr, thresholds

    def save_model(self, filename='ModeloG1_GBDT.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar.")

        joblib.dump(self.best_model, filename)
        print(f'Modelo guardado en: {filename}')
