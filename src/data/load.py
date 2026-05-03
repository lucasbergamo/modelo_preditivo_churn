import pandas as pd

from src.utils.config import DATA_BRONZE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

BRONZE_FILE = DATA_BRONZE_DIR / "telco_customer_churn.csv"


def load_bronze() -> pd.DataFrame:
    df = pd.read_csv(BRONZE_FILE)
    logger.info("bronze_loaded", rows=len(df), cols=len(df.columns))
    return df
