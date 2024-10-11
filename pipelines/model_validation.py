from sklearn import metrics
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class ModelValidationStep:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        """
        Inicializa la clase con el modelo entrenado y los datos para validación.
        """
        self.model = model  # Modelo entrenado (con búsqueda de hiperparámetros)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.fpr = None
        self.tpr = None

    def validate(self):
        """
        Ejecuta todas las métricas de validación y muestra los resultados.
        """
        self.y_pred = self.model.predict(self.X_test)
        print(
            f"Accuracy (train): {self.model.score(self.X_train, self.y_train):.4f}")
        print(
            f"Accuracy (test): {self.model.score(self.X_test, self.y_test):.4f}")

        f1 = metrics.f1_score(self.y_test, self.y_pred)
        print(f"F1 Score: {f1:.4f}")

        self.fpr, self.tpr, _ = metrics.roc_curve(
            self.y_test, self.model.decision_function(self.X_test))
        auc_value = metrics.auc(self.fpr, self.tpr)
        print(f"AUC: {auc_value:.4f}")

    def plot_confusion_matrix(self):
        """
        Genera y muestra la matriz de confusión.
        """

        cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        print(f"Confusion Matrix:\n{cm}")

        plot_confusion_matrix(conf_mat=cm, figsize=(
            5, 5), show_normed=False, cmap='Set2')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self):
        """
        Dibuja la curva ROC para el modelo.
        """
        plt.plot(self.fpr, self.tpr, label="ROC curve (area = {:.4f})".format(
            metrics.auc(self.fpr, self.tpr)))
        plt.plot([0, 1], [0, 1], "r--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()
