# -*- coding: utf-8 -*-
"""Modelo LightGBM con GridSearchCV"""

import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize


class LightGBMClassifier:
    def __init__(self, cross_validation, groups, base_params=None, use_gpu=False):
        """
        Inicializa el modelo LightGBM con parámetros base y configuración de validación cruzada.
        """
        self.base_params = base_params or {}
        if use_gpu:
            self.base_params.update({
                'device_type': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        self.model = lgb.LGBMClassifier(**self.base_params)
        self.best_model = None
        self.best_params = None
        self.cross_validation = cross_validation
        self.groups = groups

    def grid_search(self, X_train, y_train):
        """
        Realiza la búsqueda de hiperparámetros mediante GridSearchCV.
        """
        space = {
            'n_estimators': [200, 400],
            'learning_rate': [0.03, 0.05],
            'max_depth': [3, 5],
            'num_leaves': [15, 31],
            'min_child_samples': [10, 20],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }


        search = GridSearchCV(
            self.model,
            space,
            scoring='f1',
            refit='f1',
            n_jobs=6,
            cv=self.cross_validation,
            verbose=1,
            return_train_score=False
        )

        # Calcular pesos por clase (inversamente proporcionales)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        w_pos = n_neg / n_pos
        w_neg = 1
        sample_weights = np.where(y_train == 1, w_pos, w_neg)

        # Entrenar búsqueda de hiperparámetros
        result = search.fit(X_train, y_train, sample_weight=sample_weights, groups=self.groups)

        self.best_model = result
        self.best_params = result.best_params_
        return self.best_model, self.best_params

    def predict(self, X_test):
        """
        Realiza la predicción usando el mejor modelo encontrado.
        """
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        return self.best_model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo utilizando la curva ROC.
        """
        y_test_hot = label_binarize(y_test, classes=(0, 1))
        y_score = self.best_model.decision_function(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_hot.ravel(), y_score.ravel()
        )

        print("FPR:", fpr)
        print("TPR:", tpr)

        return fpr, tpr, thresholds

    def save_model(self, filename='Modelo_LGBM.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        joblib.dump(self.best_model, filename)
        print(f'Modelo guardado en: {filename}')