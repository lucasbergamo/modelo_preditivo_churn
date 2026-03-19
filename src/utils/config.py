from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
DATA_SILVER_DIR = PROJECT_ROOT / "data" / "silver"
DATA_GOLD_DIR = PROJECT_ROOT / "data" / "gold"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COL = "Churn"

MLFLOW_EXPERIMENT_NAME = "churn-prediction"
MLFLOW_TRACKING_URI = "mlruns"
