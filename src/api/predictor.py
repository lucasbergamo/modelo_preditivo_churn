import pickle

import pandas as pd
import torch

from src.api.schemas import CustomerInput
from src.models.mlp import MLP
from src.utils.config import DATA_GOLD_DIR, MODELS_DIR, SCALER_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def _get_train_columns() -> list[str]:
    return list(pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet").columns)


def _build_feature_row(customer: CustomerInput, train_cols: list[str]) -> pd.DataFrame:
    """Converte CustomerInput em DataFrame com as mesmas features do treino."""
    row = customer.model_dump()

    # features derivadas — mesmo cálculo do src/features/engineering.py
    row["charges_per_month"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["num_services"] = sum(row[col] for col in SERVICE_COLS)
    row["is_new_customer"] = int(row["tenure"] <= 12)

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=False, dtype=int)

    # garante colunas ausentes (categoria que não aparece no request)
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0

    return df[train_cols]


class ChurnPredictor:
    def __init__(self) -> None:
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self.train_cols = _get_train_columns()
        self.model = MLP(input_dim=len(self.train_cols))
        self.model.load_state_dict(torch.load(MODELS_DIR / "mlp.pt", weights_only=True))
        self.model.eval()
        logger.info("predictor_loaded", features=len(self.train_cols))

    def predict(self, customer: CustomerInput) -> tuple[float, bool]:
        df = _build_feature_row(customer, self.train_cols)
        scaled = self.scaler.transform(df)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(tensor)).item())
        return round(prob, 4), prob >= 0.5
