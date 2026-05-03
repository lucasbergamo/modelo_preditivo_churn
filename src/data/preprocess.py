import pandas as pd

from src.utils.config import DATA_SILVER_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)

BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]


def bronze_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges vem como string com espaços em branco para clientes novos (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    df = df.drop(columns=["customerID"])
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)

    for col in BINARY_YES_NO:
        df[col] = (df[col] == "Yes").astype(int)

    logger.info("silver_ready", rows=len(df), nulls=int(df.isnull().sum().sum()))
    return df


def silver_to_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=False, dtype=int)
    logger.info("features_ready", cols=len(df.columns))
    return df


def save_silver(df: pd.DataFrame) -> None:
    path = DATA_SILVER_DIR / "telco_silver.parquet"
    df.to_parquet(path, index=False)
    logger.info("silver_saved", path=str(path))
