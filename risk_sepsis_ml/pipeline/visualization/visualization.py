import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_kde_comparison(imputed_dfs: dict[str, pd.DataFrame], original_df: pd.DataFrame, column: str):
    """
    Dibuja KDE de una columna específica comparando múltiples imputaciones.

    :param imputed_dfs: Diccionario de DataFrames imputados. Ej: {'mean': df1, 'knn': df2}
    :param original_df: DataFrame original con datos faltantes.
    :param column: Nombre de la columna a graficar.
    """
    plt.figure(figsize=(10, 6))

    # Original (sin nulos)
    sns.kdeplot(original_df[column].dropna(),
                label='Original (sin nulos)', linewidth=2)

    # Imputaciones
    for method_name, df in imputed_dfs.items():
        sns.kdeplot(df[column], label=f'{method_name} imputado')

    plt.title(f'Distribución KDE para columna: {column}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot_comparison(imputed_dfs: dict[str, pd.DataFrame], column: str):
    """
    Boxplot comparativo para una columna específica.

    :param imputed_dfs: Diccionario de DataFrames imputados. Ej: {'mean': df1, 'knn': df2}
    :param column: Nombre de la columna a comparar.
    """
    data = []
    labels = []

    for method, df in imputed_dfs.items():
        data.append(df[column])
        labels.append(method)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.title(f'Boxplot comparativo - {column}')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


def plot_correlation_differences(imputed_dfs: dict[str, pd.DataFrame], base_method: str):
    """
    Muestra un heatmap con la diferencia de correlaciones entre imputaciones.

    :param imputed_dfs: Diccionario de DataFrames imputados.
    :param base_method: Nombre del método de imputación base (clave del dict).
    """
    base_corr = imputed_dfs[base_method].corr()

    for method, df in imputed_dfs.items():
        if method == base_method:
            continue
        diff = df.corr() - base_corr
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap='coolwarm', center=0, annot=False)
        plt.title(f'Diferencia de correlaciones: {method} vs {base_method}')
        plt.tight_layout()
        plt.show()
