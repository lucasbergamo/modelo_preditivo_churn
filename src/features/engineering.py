import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas antes do one-hot encoding.

    Aplicado sobre o silver (colunas numéricas e binárias já limpas).

    Features criadas:
    - charges_per_month: TotalCharges / (tenure+1) — detecta se o cliente paga
      proporcionalmente mais ou menos que o esperado pelo tempo de casa.
      Clientes novos com cobrança alta tendem a churnar mais.
    - num_services: soma de serviços adicionais contratados. Clientes com mais
      serviços têm maior custo de troca (switching cost) e churnavam menos.
    - is_new_customer: primeiros 12 meses têm taxa de churn historicamente maior
      (~40% vs ~15% para clientes com mais de 1 ano). Feature binária explícita
      facilita que o modelo aprenda esse limiar.

    Alternativas consideradas e descartadas:
    - monthly_to_total_ratio: redundante com charges_per_month
    - log(tenure): transformação útil para distribuições skewed, mas StandardScaler
      já normaliza — ganho marginal com custo de interpretabilidade
    """
    df = df.copy()

    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["num_services"] = df[SERVICE_COLS].sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)

    logger.info("features_engineered", new_features=3, total_cols=len(df.columns))
    return df
