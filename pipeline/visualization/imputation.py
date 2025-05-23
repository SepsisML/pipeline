import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ImputationPlotter:
    def __init__(self, imputed_dfs: dict[str, pd.DataFrame], original_df: pd.DataFrame):
        """
        Inicializa el objeto con los DataFrames imputados y el original.

        :param imputed_dfs: Diccionario de DataFrames imputados. Ej: {'mean': df1, 'knn': df2}
        :param original_df: DataFrame original con datos faltantes.
        """
        self.imputed_dfs = imputed_dfs
        self.original_df = original_df

    def plot_kde_comparison(self, column: str):
        """Dibuja la comparación KDE para una columna específica."""
        plt.figure(figsize=(10, 6))

        # Original (sin nulos)
        sns.kdeplot(self.original_df[column].dropna(),
                    label='Original (sin nulos)', linewidth=2)

        # Imputaciones
        for method_name, df in self.imputed_dfs.items():
            sns.kdeplot(df[column], label=f'{method_name} imputado')

        plt.title(f'Distribución KDE para columna: {column}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_boxplot_comparison(self, column: str):
        """Boxplot comparativo para una columna específica."""
        data = []
        labels = []

        for method, df in self.imputed_dfs.items():
            data.append(df[column])
            labels.append(method)

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels)
        plt.title(f'Boxplot comparativo - {column}')
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

    def plot_correlation_differences(self, base_method: str):
        """Heatmap con la diferencia de correlaciones entre imputaciones."""
        base_corr = self.imputed_dfs[base_method].corr()

        for method, df in self.imputed_dfs.items():
            if method == base_method:
                continue
            diff = df.corr() - base_corr
            plt.figure(figsize=(10, 8))
            sns.heatmap(diff, cmap='coolwarm', center=0, annot=False)
            plt.title(
                f'Diferencia de correlaciones: {method} vs {base_method}')
            plt.tight_layout()
            plt.show()
