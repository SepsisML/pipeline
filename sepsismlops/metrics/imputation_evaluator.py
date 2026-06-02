import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ImputationEvaluator:
    """
    Evalúa la calidad de imputación mediante enmascaramiento holdout.

    Una fracción de los valores observados (no NaN) es enmascarada, la estrategia
    de imputación elegida se ejecuta sobre los datos enmascarados, y el error de
    reconstrucción (RMSE, MAE) y la cobertura (fracción de posiciones enmascaradas
    correctamente imputadas) se miden por columna y se agregan por tipo de atributo.

    Diseñado para datasets con alta tasa de valores faltantes: se omiten columnas
    con menos de `min_observed` valores válidos para evitar enmascarar datos ya escasos.
    """

    def __init__(
        self,
        lab_attributes: list,
        vital_attributes: list,
        mask_fraction: float = 0.15,
        min_observed: int = 10,
        random_state: int = 42,
    ):
        self.lab_attributes = lab_attributes
        self.vital_attributes = vital_attributes
        self.all_attributes = vital_attributes + lab_attributes
        self.mask_fraction = mask_fraction
        self.min_observed = min_observed
        self.random_state = random_state

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df.replace(-9999, np.nan, inplace=True)
        return df

    def _mask_observed_values(self, df: pd.DataFrame):
        """
        Enmascara aleatoriamente mask_fraction de los valores observados por columna.
        Omite columnas con menos de min_observed valores observados.
        Retorna (masked_df, diccionario true_values, diccionario mask_indices).
        """
        rng = np.random.default_rng(self.random_state)
        masked_df = df.copy()
        true_values = {}
        mask_indices = {}

        for col in self.all_attributes:
            if col not in df.columns:
                continue
            observed_idx = df.index[df[col].notna()].to_numpy()
            if len(observed_idx) < self.min_observed:
                continue
            
            n_mask = max(1, int(len(observed_idx) * self.mask_fraction))
            chosen = rng.choice(observed_idx, size=n_mask, replace=False)

            true_values[col] = df.loc[chosen, col].copy()
            mask_indices[col] = chosen
            masked_df.loc[chosen, col] = np.nan

        return masked_df, true_values, mask_indices

    def _instantiate_strategy(self, strategy_name: str, df: pd.DataFrame):
        from sepsismlops.data_management.imputers import (
            KNNImputerStrategy,
            MiceForestImputationStrategy,
            MeanImputationStrategy,
            CustomMeanImputationStrategy,
        )
        strategies = {
            "knn": lambda: KNNImputerStrategy(df, self.lab_attributes, self.vital_attributes),
            "miceforest": lambda: MiceForestImputationStrategy(df, self.lab_attributes, self.vital_attributes),
            "mean": lambda: MeanImputationStrategy(df, self.lab_attributes, self.vital_attributes),
            "custom-mean": lambda: CustomMeanImputationStrategy(df, self.lab_attributes, self.vital_attributes),
        }
        if strategy_name not in strategies:
            raise ValueError(f"Unknown imputation strategy: {strategy_name!r}")
        return strategies[strategy_name]()

    def _compute_metrics(
        self,
        imputed_df: pd.DataFrame,
        true_values: dict,
        mask_indices: dict,
    ):
        """
        Calcula cobertura, RMSE y MAE por columna, y agrega en métricas
        medias para vitales, laboratorio y el total.

        Cobertura = fracción de posiciones enmascaradas que fueron imputadas
        correctamente (es decir, no siguen siendo NaN tras la imputación).
        Cobertura de 0 significa que la estrategia no pudo imputar esas posiciones.
        """
        vital_rmse, vital_mae = [], []
        lab_rmse, lab_mae = [], []
        per_column = {}

        for col, true_vals in true_values.items():
            idx = mask_indices[col]
            pred_vals = imputed_df.loc[idx, col]

            valid = true_vals.notna() & pred_vals.notna()
            coverage = float(valid.mean())

            if valid.sum() < 2:
                per_column[col] = {"coverage": coverage, "rmse": None, "mae": None}
                continue

            rmse = float(np.sqrt(mean_squared_error(true_vals[valid], pred_vals[valid])))
            mae = float(mean_absolute_error(true_vals[valid], pred_vals[valid]))
            per_column[col] = {"coverage": coverage, "rmse": rmse, "mae": mae}

            if col in self.vital_attributes:
                vital_rmse.append(rmse)
                vital_mae.append(mae)
            else:
                lab_rmse.append(rmse)
                lab_mae.append(mae)

        summary = {}
        if vital_rmse:
            summary["rmse_vitals"] = float(np.mean(vital_rmse))
            summary["mae_vitals"] = float(np.mean(vital_mae))
        if lab_rmse:
            summary["rmse_labs"] = float(np.mean(lab_rmse))
            summary["mae_labs"] = float(np.mean(lab_mae))
        all_rmse = vital_rmse + lab_rmse
        if all_rmse:
            summary["rmse_overall"] = float(np.mean(all_rmse))
            summary["mae_overall"] = float(np.mean(vital_mae + lab_mae))

        return summary, per_column

    def evaluate(self, df: pd.DataFrame, strategy_name: str):
        """
        Ejecuta la evaluación holdout para la estrategia indicada.

        Args:
            df: DataFrame crudo (puede contener valores centinela -9999).
            strategy_name: Una de 'knn', 'miceforest', 'mean', 'custom-mean'.

        Returns:
            mlflow_metrics (dict): Métricas resumen con prefijo 'imputation_eval/'
                listas para pasar a mlflow.log_metrics().
                Claves: rmse_overall, mae_overall, rmse_vitals, mae_vitals,
                        rmse_labs, mae_labs (solo presentes cuando son calculables).
            per_column (dict): Diccionario por columna con claves 'coverage', 'rmse',
                'mae' para inspección detallada o logging de artefactos.
        """
        prepared = self._prepare_df(df)
        masked_df, true_values, mask_indices = self._mask_observed_values(prepared)

        imputer = self._instantiate_strategy(strategy_name, masked_df)
        imputed_df = imputer.impute()

        summary, per_column = self._compute_metrics(imputed_df, true_values, mask_indices)
        mlflow_metrics = {f"imputation_eval/{k}": v for k, v in summary.items()}

        return mlflow_metrics, per_column
