import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    recall_score,
    roc_auc_score,
)

from src.models.mlp import MLP
from src.utils.config import DATA_GOLD_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "fbeta2": round(float(fbeta_score(y_true, y_pred, beta=2)), 4),
    }


def evaluate_all() -> pd.DataFrame:
    import numpy as np
    from src.models.baselines import get_baselines

    X_train = pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_GOLD_DIR / "y_train.parquet").squeeze()
    X_test = pd.read_parquet(DATA_GOLD_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_GOLD_DIR / "y_test.parquet").squeeze()

    results = {}

    # Baselines sklearn — avaliados no test set
    for name, model in get_baselines().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test.values, proba)
        logger.info("evaluated", model=name, auc_roc=results[name]["auc_roc"])

    # MLP PyTorch — carrega pesos salvos
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    mlp = MLP(input_dim=X_test.shape[1])
    mlp.load_state_dict(torch.load(MODELS_DIR / "mlp.pt", weights_only=True))
    mlp.eval()
    with torch.no_grad():
        proba_mlp = torch.sigmoid(mlp(X_tensor)).numpy()
    results["mlp_pytorch"] = compute_metrics(y_test.values, proba_mlp)
    logger.info("evaluated", model="mlp_pytorch", auc_roc=results["mlp_pytorch"]["auc_roc"])

    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    df = df.sort_values("auc_roc", ascending=False).reset_index(drop=True)

    print("\n=== COMPARATIVO FINAL — TEST SET ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    evaluate_all()
