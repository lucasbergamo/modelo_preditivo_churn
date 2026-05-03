# Jornada do Projeto — Churn Predictor Telecom

Documento vivo. Atualizado a cada checkpoint concluído.

---

## Contexto de Negócio

Operadora de telecom com ~7 mil clientes. Taxa de churn de ~26%. Custo de reter um cliente é menor que o de adquirir um novo — falso negativo (não identificar quem vai sair) é mais caro que falso positivo. Por isso usamos F-beta com β=2 como métrica secundária (penaliza mais o recall).

---

## Stack

Python 3.12 · PyTorch · scikit-learn · MLflow · FastAPI · Pandera · structlog · ruff · pytest

---

## Checkpoints

### ✅ Checkpoint 1 — Pipeline de Dados (commit `0d3e4f5`)

**O que foi feito:**
- `src/data/load.py` — carrega o CSV bruto (`data/bronze/`)
- `src/data/preprocess.py` — bronze → silver: corrige `TotalCharges` (string com espaços → float), remove `customerID`, converte binárias Yes/No → 0/1, salva parquet em `data/silver/`
- `src/data/split.py` — silver → gold: one-hot encoding, StandardScaler (fit só no treino), split estratificado 70/15/15, salva 6 parquets em `data/gold/`
- `src/data/pipeline.py` — orquestra tudo em sequência

**Resultado ao rodar `python -m src.data.pipeline`:**
```
bronze_loaded   rows=7043  cols=21
silver_ready    rows=7043  nulls=0
features_ready  cols=27
split_done      train=4930  val=1056  test=1057  churn_rate_train=0.265
gold_saved      (6 parquets)
```

**Decisões técnicas:**
- Medallion Architecture (bronze/silver/gold) para rastreabilidade dos dados
- Split estratificado garante proporção de churn igual nos 3 conjuntos
- Scaler fitado só no treino evita data leakage

---

### ✅ Checkpoint 2 — Baselines sklearn (em andamento)

**O que está sendo feito:**
- `src/models/baselines.py` — define 3 modelos: DummyClassifier, LogisticRegression, RandomForest
- `src/training/train_baselines.py` — treina os 3, avalia no val set, loga métricas no MLflow

**Métricas alvo:** AUC-ROC, PR-AUC, F1, Recall, F-beta(β=2)

---

### ⬜ Checkpoint 3 — MLP PyTorch

`src/models/mlp.py` + `src/training/train.py`: rede MLP com early stopping e batching.

---

### ⬜ Checkpoint 4 — Avaliação comparativa

`src/evaluation/`: tabela comparando todos os modelos nas 5 métricas.

---

### ⬜ Checkpoint 5 — FastAPI

`src/api/app.py`: endpoints `/predict` e `/health`, Pydantic, middleware de latência.

---

### ⬜ Checkpoint 6 — Testes

`tests/`: smoke test, schema Pandera, API test. `make test` verde.

---

### ⬜ Checkpoint 7 — Docs finais

`docs/ml_canvas.md`, `docs/model_card.md`, `docs/monitoring_plan.md`, README final.
