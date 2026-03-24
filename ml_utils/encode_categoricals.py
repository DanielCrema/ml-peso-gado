import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: list):
    """
    Aplica One-Hot Encoding nas variáveis categóricas,
    garantindo consistência entre treino e teste.
    """

    # Fit no treino
    encoded_train = encoder.fit_transform(train[categorical_cols])
    encoded_test = encoder.transform(test[categorical_cols])

    # DataFrames com nomes das colunas
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=train.index)
    test_encoded_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=test.index)

    # Drop colunas originais e juntar
    train_final = pd.concat([train.drop(columns=categorical_cols), train_encoded_df], axis=1)
    test_final = pd.concat([test.drop(columns=categorical_cols), test_encoded_df], axis=1)

    return train_final, test_final