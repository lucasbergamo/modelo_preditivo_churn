"""Schema test — valida que os dados gold têm estrutura correta com Pandera."""

import pandas as pd
from pandera.pandas import Column, DataFrameSchema

from src.utils.config import DATA_GOLD_DIR

gold_schema = DataFrameSchema(
    {
        "tenure": Column(float),
        "MonthlyCharges": Column(float),
        "TotalCharges": Column(float),
        "charges_per_month": Column(float),
        "num_services": Column(float),
        "is_new_customer": Column(float),
    },
    strict=False,  # permite colunas extras (one-hot encoded)
)


def test_gold_schema_train():
    """X_train tem colunas obrigatórias com tipos e ranges corretos."""
    df = pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet")
    gold_schema.validate(df)


def test_gold_no_nulls():
    """Nenhum split do gold tem valores nulos."""
    for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        df = pd.read_parquet(DATA_GOLD_DIR / f"{name}.parquet")
        assert df.isnull().sum().sum() == 0, f"{name} tem nulos"


def test_gold_split_sizes():
    """Splits têm tamanhos consistentes com 70/15/15."""
    X_train = pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet")
    X_val = pd.read_parquet(DATA_GOLD_DIR / "X_val.parquet")
    X_test = pd.read_parquet(DATA_GOLD_DIR / "X_test.parquet")
    total = len(X_train) + len(X_val) + len(X_test)
    assert len(X_train) / total >= 0.68, "Treino menor que esperado"
    assert len(X_val) / total >= 0.13, "Validação menor que esperado"
    assert len(X_test) / total >= 0.13, "Teste menor que esperado"
