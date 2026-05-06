import pandas as pd

from src.api.predictor import SERVICE_COLS as API_SERVICE_COLS
from src.api.predictor import _build_feature_row, _get_train_columns
from src.api.schemas import CustomerInput
from src.data.preprocess import ONEHOT_COLS as TRAIN_ONEHOT_COLS
from src.data.preprocess import bronze_to_silver, silver_to_features
from src.features.engineering import SERVICE_COLS as TRAIN_SERVICE_COLS
from src.features.engineering import add_features
from src.utils.config import TARGET_COL

# Cliente sintético representando uma linha do CSV bronze (formato original do dataset)
RAW_ROW = {
    "customerID": "test-parity-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": "421.5",  # string — como vem do CSV original
    "Churn": "No",
}

# Mesmo cliente no formato esperado pela API (pós-encoding binário)
API_INPUT = CustomerInput(
    gender=1,
    SeniorCitizen=0,
    Partner=1,
    Dependents=0,
    tenure=6,
    PhoneService=1,
    MultipleLines=0,
    InternetService="Fiber optic",
    OnlineSecurity=0,
    OnlineBackup=1,
    DeviceProtection=0,
    TechSupport=0,
    StreamingTV=1,
    StreamingMovies=0,
    Contract="Month-to-month",
    PaperlessBilling=1,
    PaymentMethod="Electronic check",
    MonthlyCharges=70.35,
    TotalCharges=421.5,
)


def test_service_cols_identical():
    """SERVICE_COLS em engineering.py e predictor.py devem ser idênticas.

    Essas listas estão duplicadas — se divergirem, num_services fica errado
    em inferência sem nenhum erro explícito.
    """
    assert TRAIN_SERVICE_COLS == API_SERVICE_COLS, (
        f"SERVICE_COLS divergiu entre treino e API.\n"
        f"  Só no treino : {set(TRAIN_SERVICE_COLS) - set(API_SERVICE_COLS)}\n"
        f"  Só na API    : {set(API_SERVICE_COLS) - set(TRAIN_SERVICE_COLS)}"
    )


def test_onehot_cols_covered_by_api():
    """Todas as colunas que recebem one-hot no treino devem estar presentes no schema da API."""
    api_fields = set(CustomerInput.model_fields.keys())
    missing = [col for col in TRAIN_ONEHOT_COLS if col not in api_fields]
    assert not missing, (
        f"Colunas com one-hot no treino ausentes no CustomerInput: {missing}"
    )


def test_feature_parity_train_vs_api():
    """O mesmo cliente processado pelo pipeline de treino e pela API deve gerar vetores idênticos.

    Cobre: features derivadas (charges_per_month, num_services, is_new_customer),
    one-hot encoding (InternetService, Contract, PaymentMethod) e alinhamento de colunas.
    """
    train_cols = _get_train_columns()

    # Caminho do treino: bronze → silver → features derivadas → one-hot
    df_bronze = pd.DataFrame([RAW_ROW])
    df_silver = bronze_to_silver(df_bronze).drop(columns=[TARGET_COL])
    df_features = add_features(df_silver)
    df_train_path = silver_to_features(df_features).reindex(columns=train_cols, fill_value=0)

    # Caminho da API: CustomerInput → _build_feature_row
    df_api_path = _build_feature_row(API_INPUT, train_cols)

    pd.testing.assert_frame_equal(
        df_train_path.reset_index(drop=True),
        df_api_path.reset_index(drop=True),
        check_dtype=False,
        obj="Feature parity treino vs API",
    )
