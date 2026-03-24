import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Avalia um modelo de regressão utilizando métricas apropriadas
    e visualizações para análise de performance.
    """

    # 📈 Predição
    y_pred = estimator.predict(X_test)

    # 📊 Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("=> Métricas de Regressão:\n")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R²   (Coeficiente de Determinação): {r2:.4f}\n")

    # =========================
    # 📊 Gráfico: Real vs Predito
    # =========================
    plt.figure(figsize=(6, 5))

    plt.scatter(y_test, y_pred, alpha=0.6)

    # Linha ideal (y = x)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Valor Real (kg)")
    plt.ylabel("Valor Predito (kg)")
    plt.title("Real vs Predito")

    plt.tight_layout()
    plt.show()
    plt.close()

    # =========================
    # 📉 Distribuição dos erros
    # =========================
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20)

    plt.title("Distribuição dos Erros (Resíduos)")
    plt.xlabel("Erro (Real - Predito)")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.show()
    plt.close()