import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_matrix(df):
    corr = df.corr()

    # 🔺 Máscara para esconder o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 10))  # 🔼 maior área

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8}  # 🔽 fonte menor
    )

    plt.title("Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_target_correlation(df, target):
    corr = df.corr()[target].drop(target).sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=corr.values,
        y=corr.index,
        hue=corr.index,
        legend=False
    )

    plt.title(f"Correlation with {target}")
    plt.xlabel("Correlation")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_linear_relationships(df, target):
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    for col in numeric_df.columns:
        if col == target:
            continue

        plt.figure(figsize=(5, 4))

        sns.regplot(
            x=df[col],
            y=df[target],
            scatter_kws={"alpha": 0.6}
        )

        plt.title(f"{col} vs {target}")
        plt.xlabel(col)
        plt.ylabel(target)

        plt.tight_layout()
        plt.show()
        plt.close()

def plot_relationships(df, target):
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plot_correlation_matrix(numeric_df)
    plot_target_correlation(numeric_df, target)
    plot_linear_relationships(df, target)