import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.plot_feature import plot_feature

def visualize_y(df: pd.DataFrame, y_col_name: str) -> None:
    """
    Exibe a distribuição da variável alvo e gera sua visualização gráfica.

    Parâmetros
    ----------
    df : pandas.DataFrame
        - DataFrame contendo a variável alvo.
    y_col_name : str
        - Nome da coluna correspondente à variável alvo.

    Retorna
    -------
    None
    """
    # 📊 Estatísticas básicas
    print(f'{df[y_col_name].value_counts()}\n')
    print(df[y_col_name].value_counts(normalize=True))

    # 📈 Distribuição (histograma ou barplot automático)
    plot_feature(df, y_col_name)

    # 📦 Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[y_col_name])

    plt.title(f'Boxplot: {y_col_name}')
    plt.ylabel(y_col_name)

    plt.tight_layout()
    plt.show()
    plt.close()
