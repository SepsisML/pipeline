import mlflow
import numpy as np
import pandas as pd
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
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix/raw.json")
        print(f"Confusion Matrix (counts):\n{cm}")
        plot_confusion_matrix(conf_mat=cm, figsize=(5, 5), show_normed=False, cmap='Set2')
        plt.tight_layout()
        plt.show()

        # Matriz normalizada
        # cm_norm = metrics.confusion_matrix(self.y_test, self.y_pred, normalize='true')
        # print(f"Confusion Matrix (normalized by true labels):\n{cm_norm}")
        # plot_confusion_matrix(conf_mat=cm_norm, figsize=(5, 5), show_normed=True, cmap='Blues')
        # plt.tight_layout()
        # plt.show()

    def plot_f1_score(self):
        f1_score = metrics.f1_score(self.y_test, self.y_pred)
        print("El F1_Score es: ", f1_score)
        return f1_score

    
    def f1_por_paciente(df):
        result = pd.DataFrame({
            "Paciente": X_test["Paciente"],  # si no está: pásalo desde self.df
            "y_test": y_test,
            "y_pred": y_pred
        })

        # df tiene columnas: Paciente, y_test, y_pred
        pacientes = result.groupby("Paciente")

        f1_scores = []

        for paciente, grupo in pacientes:
            y_true = grupo["y_test"]
            y_hat = grupo["y_pred"]

            # estrategia simple: mayoría de las predicciones del paciente
            true_label = 1 if y_true.mean() >= 0.5 else 0
            pred_label = 1 if y_hat.mean() >= 0.5 else 0

            f1 = f1_score([true_label], [pred_label])
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)

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

    # def optimize_threshold_for_f1(self):
    #     """
    #     Encuentra el umbral que maximiza F1 usando y_proba. Retorna (best_threshold, best_f1).
    #     """
    #     if self.y_proba is None:
    #         raise ValueError("y_proba no está definido. Provee probabilidades para optimizar el umbral.")
    #     precisions, recalls, thresholds = metrics.precision_recall_curve(self.y_test, self.y_proba)
    #     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    #     best_idx = f1_scores.argmax()
    #     # precision_recall_curve devuelve len(thresholds) = len(precisions)-1
    #     best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    #     best_f1 = f1_scores[best_idx]
    #     print(f"Best threshold by F1: {best_threshold:.4f} | F1: {best_f1:.4f}")
    #     return best_threshold, best_f1

    def u_tp(self, delta_t: float) -> float:
        """U_TP: utilidad de predecir 1 en paciente con sepsis. delta_t = t - t_sepsis."""
        dt_early, dt_optimal, dt_late = -12, -6, 3
        if delta_t < dt_early:
            return 0.0
        elif delta_t <= dt_optimal:
            return (delta_t - dt_early) / (dt_optimal - dt_early)
        elif delta_t <= dt_late:
            return 1.0 - (delta_t - dt_optimal) / (dt_late - dt_optimal)
        else:
            return -2.0


    def u_fn(self, delta_t: float) -> float:
        """U_FN: penalización por predecir 0 en paciente con sepsis. delta_t = t - t_sepsis."""
        dt_optimal, dt_late = -6, 3
        if delta_t <= dt_optimal:
            return 0.0
        elif delta_t <= dt_late:
            return -2.0 * (delta_t - dt_optimal) / (dt_late - dt_optimal)
        else:
            return -2.0


    def u_fp(self) -> float:
        """U_FP: penalización por predecir 1 en paciente sin sepsis (por hora)."""
        return -0.05

    def compute_utility_score(self, patient_ids: pd.Series) -> float:
        df = self.X_test[["ICULOS"]].copy().reset_index(drop=True)
        df["Paciente"] = patient_ids.values
        df["y_pred"] = self.y_pred
        df["y_test"] = self.y_test.reset_index(drop=True).values

        patient_scores = []
        for _, grupo in df.groupby("Paciente"):
            grupo = grupo.sort_values("ICULOS")
            y_true = grupo["y_test"].values
            y_hat  = grupo["y_pred"].values
            iculos = grupo["ICULOS"].values

            has_sepsis = y_true.max() == 1
            score = 0.0

            if has_sepsis:
                t_sepsis = iculos[np.where(y_true == 1)[0][0]]
                for t, pred in zip(iculos, y_hat):
                    delta_t = t - t_sepsis
                    if pred == 1:
                        score += self.u_tp(delta_t)  # U_TP
                    else:
                        score += self.u_fn(delta_t)            # U_FN
            else:
                for pred in y_hat:
                    if pred == 1:
                        score += self.u_fp()                   # U_FP
                    # U_TN = 0, no contribuye

            patient_scores.append(score)

        return float(np.mean(patient_scores)) if patient_scores else 0.0

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
