from sklearn import metrics
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class MetricsStep:
    def __init__(self, y_pred, X_train, y_train, X_test, y_test, y_proba=None):
        """
        Inicializa la clase con el modelo entrenado y los datos para validación.
        """
        self.y_pred = y_pred
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_proba = y_proba
        self.fpr = None
        self.tpr = None

    def validate(self):
        """
        Ejecuta todas las métricas de validación y muestra los resultados.
        """
        acc = metrics.accuracy_score(self.y_test, self.y_pred)
        f1 = metrics.f1_score(self.y_test, self.y_pred)
        print(f"Accuracy (test): {acc:.4f}")
        print(f"F1 Score (threshold=0.5): {f1:.4f}")
        
        # ROC-AUC usando probabilidades si están disponibles
        if self.y_proba is not None:
            self.fpr, self.tpr, _ = metrics.roc_curve(self.y_test, self.y_proba)
            auc_value = metrics.auc(self.fpr, self.tpr)
            print(f"AUC: {auc_value:.4f}")

    def plot_confusion_matrix(self):
        """
        Genera y muestra la matriz de confusión.
        """
        cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        print(f"Confusion Matrix (counts):\n{cm}")
        plot_confusion_matrix(conf_mat=cm, figsize=(5, 5), show_normed=False, cmap='Set2')
        plt.tight_layout()
        plt.show()

        # Matriz normalizada
        cm_norm = metrics.confusion_matrix(self.y_test, self.y_pred, normalize='true')
        print(f"Confusion Matrix (normalized by true labels):\n{cm_norm}")
        plot_confusion_matrix(conf_mat=cm_norm, figsize=(5, 5), show_normed=True, cmap='Blues')
        plt.tight_layout()
        plt.show()

    def plot_f1_score(self):
        f1_score = metrics.f1_score(self.y_test, self.y_pred)
        print("El F1_Score es: ", f1_score)
        return f1_score

    def plot_roc_curve(self):
        """
        Dibuja la curva ROC para el modelo.
        """
        if self.y_proba is not None:
            plt.plot(self.fpr, self.tpr, label="ROC curve (area = {:.4f})".format(
                metrics.auc(self.fpr, self.tpr)))
            plt.plot([0, 1], [0, 1], "r--")
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.show()

    def optimize_threshold_for_f1(self):
        """
        Encuentra el umbral que maximiza F1 usando y_proba. Retorna (best_threshold, best_f1).
        """
        if self.y_proba is None:
            raise ValueError("y_proba no está definido. Provee probabilidades para optimizar el umbral.")
        precisions, recalls, thresholds = metrics.precision_recall_curve(self.y_test, self.y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        best_idx = f1_scores.argmax()
        # precision_recall_curve devuelve len(thresholds) = len(precisions)-1
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        print(f"Best threshold by F1: {best_threshold:.4f} | F1: {best_f1:.4f}")
        return best_threshold, best_f1

    def plot_precision_recall_curve(self):
        if self.y_proba is None:
            raise ValueError("y_proba no está definido. Provee probabilidades para graficar la curva PR.")
        precisions, recalls, _ = metrics.precision_recall_curve(self.y_test, self.y_proba)
        ap = metrics.average_precision_score(self.y_test, self.y_proba)
        plt.plot(recalls, precisions, label=f"PR curve (AP = {ap:.4f})")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.show()
