"""Loop de treinamento do MLP PyTorch com early stopping e MLflow tracking.

Fluxo:
1. Carrega splits gold (parquet) → converte para tensores PyTorch
2. Monta DataLoader com mini-batches (batching)
3. Treina por até MAX_EPOCHS, avaliando no val set a cada época
4. Early stopping: interrompe se val_loss não melhorar por PATIENCE épocas
5. Restaura os pesos da melhor época antes de encerrar
6. Loga parâmetros, métricas por época e métricas finais no MLflow
"""

import copy

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP
from src.utils.config import (
    DATA_GOLD_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    SEED,
)
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)

# Hiperparâmetros — valores alinhados com aula04-mlp-pytorch-keras do curso
HIDDEN_DIMS = (64, 32)   # funil progressivo: suficiente para 27 features
DROPOUT = 0.3            # regularização conservadora para dataset pequeno (~5k amostras)
LEARNING_RATE = 1e-3     # padrão do Adam em problemas tabulares
BATCH_SIZE = 64          # mini-batch: balanço entre estabilidade e velocidade
MAX_EPOCHS = 200         # teto; early stopping vai interromper antes
PATIENCE = 15            # épocas sem melhora antes de parar
POS_WEIGHT_FACTOR = 2.0  # penaliza falsos negativos (churn não detectado é mais caro)


def load_tensors() -> tuple[torch.Tensor, ...]:
    X_train = torch.tensor(pd.read_parquet(DATA_GOLD_DIR / "X_train.parquet").values, dtype=torch.float32)
    y_train = torch.tensor(pd.read_parquet(DATA_GOLD_DIR / "y_train.parquet").values.squeeze(), dtype=torch.float32)
    X_val = torch.tensor(pd.read_parquet(DATA_GOLD_DIR / "X_val.parquet").values, dtype=torch.float32)
    y_val = torch.tensor(pd.read_parquet(DATA_GOLD_DIR / "y_val.parquet").values.squeeze(), dtype=torch.float32)
    return X_train, y_train, X_val, y_val


def compute_metrics(model: MLP, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        proba = torch.sigmoid(logits).numpy()  # sigmoid só na inferência
    preds = (proba >= 0.5).astype(int)
    y_np = y.numpy().astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(y_np, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_np, proba)), 4),
        "f1": round(float(f1_score(y_np, preds)), 4),
        "recall": round(float(recall_score(y_np, preds)), 4),
        "fbeta2": round(float(fbeta_score(y_np, preds, beta=2)), 4),
    }


def run() -> None:
    set_global_seed()
    torch.manual_seed(SEED)

    X_train, y_train, X_val, y_val = load_tensors()
    input_dim = X_train.shape[1]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)

    # pos_weight penaliza falsos negativos — churn não detectado custa mais que alarme falso
    pos_weight = torch.tensor([POS_WEIGHT_FACTOR])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Adam: otimizador padrão para MLPs, adaptativo por parâmetro
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mlp_pytorch"):
        mlflow.log_params({
            "model": "MLP",
            "hidden_dims": str(HIDDEN_DIMS),
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "pos_weight": POS_WEIGHT_FACTOR,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        })

        best_val_loss = np.inf
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        stopped_at = MAX_EPOCHS

        for epoch in range(1, MAX_EPOCHS + 1):
            # --- treino ---
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_train)

            # --- validação ---
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                stopped_at = epoch
                logger.info("early_stopping", epoch=epoch, best_val_loss=round(best_val_loss, 4))
                break

        # restaura melhor época antes de avaliar
        model.load_state_dict(best_weights)
        mlflow.log_param("stopped_at_epoch", stopped_at)

        metrics = compute_metrics(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        logger.info("mlp_trained", **metrics)

        # salva artefato do modelo
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "mlp.pt"
        torch.save(best_weights, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("model_saved", path=str(model_path))


if __name__ == "__main__":
    run()
