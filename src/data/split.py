import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import DATA_GOLD_DIR, MODELS_DIR, SCALER_PATH, SEED, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)

VAL_SIZE = 0.15
TEST_SIZE = 0.15


def split_and_scale(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_SIZE + TEST_SIZE), stratify=y, random_state=SEED
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    logger.info(
        "split_done",
        train=len(X_train_s),
        val=len(X_val_s),
        test=len(X_test_s),
        churn_rate_train=round(float(y_train.mean()), 3),
    )
    return {
        "X_train": X_train_s,
        "y_train": y_train.reset_index(drop=True),
        "X_val": X_val_s,
        "y_val": y_val.reset_index(drop=True),
        "X_test": X_test_s,
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
    }


def save_scaler(scaler: StandardScaler) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("scaler_saved", path=str(SCALER_PATH))


def save_gold(splits: dict[str, pd.DataFrame]) -> None:
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        path = DATA_GOLD_DIR / f"{name}.parquet"
        splits[name].to_frame().to_parquet(path, index=False) if splits[name].ndim == 1 else splits[
            name
        ].to_parquet(path, index=False)
        logger.info("gold_saved", file=name)
