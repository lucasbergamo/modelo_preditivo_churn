import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    recall_score,
    roc_auc_score,
)

from src.models.baselines import get_baselines
from src.utils.config import DATA_GOLD_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


def load_gold() -> tuple[pd.DataFrame, ...]:
    X_train = pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_GOLD_DIR / "y_train.parquet").squeeze()
    X_val = pd.read_parquet(DATA_GOLD_DIR / "X_val.parquet")
    y_val = pd.read_parquet(DATA_GOLD_DIR / "y_val.parquet").squeeze()
    return X_train, y_train, X_val, y_val


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    return {
        "auc_roc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "fbeta2": round(fbeta_score(y_true, y_pred, beta=2), 4),
    }


def train_and_log(name: str, model, X_train, y_train, X_val, y_val) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)

        mlflow.log_param("model", name)
        mlflow.log_metrics(metrics)

        logger.info("baseline_trained", model=name, **metrics)
        return metrics


def run() -> None:
    set_global_seed()
    X_train, y_train, X_val, y_val = load_gold()
    baselines = get_baselines()

    results = {}
    for name, model in baselines.items():
        results[name] = train_and_log(name, model, X_train, y_train, X_val, y_val)

    logger.info("baselines_summary", **{k: v["auc_roc"] for k, v in results.items()})


if __name__ == "__main__":
    run()
