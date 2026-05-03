"""API test — valida endpoints /health e /predict com TestClient."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

HIGH_RISK_CLIENT = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 2,
    "PhoneService": 1,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 70.0,
    "TotalCharges": 140.0,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}

LOW_RISK_CLIENT = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 1,
    "tenure": 60,
    "PhoneService": 1,
    "MultipleLines": 1,
    "OnlineSecurity": 1,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "TechSupport": 1,
    "StreamingTV": 1,
    "StreamingMovies": 1,
    "PaperlessBilling": 0,
    "MonthlyCharges": 95.0,
    "TotalCharges": 5700.0,
    "InternetService": "DSL",
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_schema():
    response = client.post("/predict", json=HIGH_RISK_CLIENT)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


def test_predict_high_risk_client():
    """Cliente de alto risco deve ter probabilidade acima de 0.5."""
    response = client.post("/predict", json=HIGH_RISK_CLIENT)
    assert response.json()["churn_prediction"] is True


def test_predict_low_risk_client():
    """Cliente fidelizado deve ter probabilidade abaixo de 0.5."""
    response = client.post("/predict", json=LOW_RISK_CLIENT)
    assert response.json()["churn_prediction"] is False


def test_predict_missing_field_returns_422():
    """Request sem campo obrigatório deve retornar 422 (Unprocessable Entity)."""
    incomplete = {k: v for k, v in HIGH_RISK_CLIENT.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_type_returns_422():
    """Campo com tipo errado deve retornar 422."""
    invalid = {**HIGH_RISK_CLIENT, "tenure": "dois meses"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
