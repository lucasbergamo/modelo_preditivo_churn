"""Smoke test — verifica que o modelo carrega e produz previsão válida."""

import pandas as pd
import torch

from src.models.mlp import MLP
from src.utils.config import DATA_GOLD_DIR, MODELS_DIR


def test_mlp_loads_and_predicts():
    """Modelo salvo carrega sem erro e produz probabilidade entre 0 e 1."""
    X = pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet")
    input_dim = X.shape[1]

    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load(MODELS_DIR / "mlp.pt", weights_only=True))
    model.eval()

    dummy_input = torch.zeros(1, input_dim)
    with torch.no_grad():
        prob = float(torch.sigmoid(model(dummy_input)).item())

    assert 0.0 <= prob <= 1.0, f"Probabilidade fora do range: {prob}"
