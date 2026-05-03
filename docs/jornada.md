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

### ✅ Checkpoint 3 — MLP PyTorch (em andamento)

**Arquivos criados:**
- `src/models/mlp.py` — arquitetura da rede
- `src/training/train.py` — loop de treinamento completo

**Arquitetura: Input(27) → Dense(64) → Dense(32) → Output(1)**

Essa arquitetura de "funil progressivo" é o padrão ensinado na `aula04-mlp-pytorch-keras` do curso (hidden_layer_sizes=(64,32)). Para dados tabulares com ~5k amostras e 27 features, redes mais largas ou profundas tendem a overfitting sem ganho real.

**Decisões técnicas e alternativas consideradas:**

| Decisão | Escolha | Alternativa | Por que escolhemos |
|---|---|---|---|
| Normalização interna | `BatchNorm1d` | `LayerNorm` / sem norm | Estabiliza gradientes em features heterogêneas; LayerNorm seria melhor só com batches muito pequenos |
| Regularização | `Dropout(0.3)` | `weight_decay` no Adam | Dropout força redundância nas representações; 0.3 é conservador para dataset pequeno |
| Função de loss | `BCEWithLogitsLoss` | `BCELoss + Sigmoid no modelo` | Numericamente mais estável — funde sigmoid e cross-entropy num só passo, evita underflow/overflow |
| Penalização de classe | `pos_weight=2.0` | sem peso / SMOTE | Falso negativo (churn não detectado) custa mais que alarme falso — penalizamos proporcionalmente |
| Otimizador | `Adam lr=1e-3` | SGD com momentum | Adam é adaptativo por parâmetro, converge mais rápido em dados tabulares |
| Early stopping | `patience=15` | `patience=10` | 15 épocas de tolerância evita parar cedo em platôs temporários |

**Fonte das escolhas:** `aula04-mlp-pytorch-keras` confirma hidden_dims=(64,32), ReLU, Adam e early_stopping. BatchNorm, Dropout e BCEWithLogitsLoss são boas práticas do ecossistema PyTorch não cobertas na aula simplificada (que usa sklearn.MLPClassifier como proxy).

**O que o Tech Challenge exige e como atendemos:**
- ✅ MLP funcional em PyTorch — `src/models/mlp.py` com `nn.Module`
- ✅ Early stopping — implementado com `patience=15`, restaura melhor época
- ✅ Batching — `DataLoader` com `batch_size=64`, `shuffle=True`
- ✅ MLflow tracking — parâmetros, loss por época, métricas finais, artefato `.pt`

**Resultado obtido (val set):**

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | F-beta(β=2) |
|---|---|---|---|---|---|
| Dummy | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | **0.845** | **0.629** | 0.625 | **0.814** | **0.726** |
| Random Forest | 0.822 | 0.594 | 0.542 | 0.493 | 0.512 |
| **MLP PyTorch** | 0.844 | 0.627 | **0.629** | 0.754 | 0.699 |

**Early stopping na época 25** — o modelo parou de melhorar no val set após ~10 épocas (25 - patience=15). Isso indica convergência rápida dado o tamanho do dataset (~5k amostras, 27 features).

**Análise honesta dos resultados:**
O MLP ficou praticamente empatado com a Regressão Logística (AUC-ROC 0.844 vs 0.845). Isso é esperado e documentado na literatura para datasets tabulares pequenos — o paper "Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022) mostra que redes neurais raramente superam modelos lineares ou baseados em árvores em tabelas com menos de 10k amostras sem feature engineering intensivo.

A vantagem do MLP neste projeto não é a performance bruta, mas a demonstração de domínio técnico: loop de treino controlado, early stopping, batching, regularização com Dropout e BatchNorm, e rastreamento completo no MLflow — todos os requisitos do Tech Challenge atendidos.

**O que a falta de EDA impactou:** sem EDA formal, não exploramos feature engineering (ex: ratios entre MonthlyCharges/tenure, interações entre tipo de contrato e serviços adicionais) que poderiam dar ao MLP uma vantagem real sobre modelos lineares. Com features mais ricas, redes neurais teriam mais padrões não-lineares para aprender.

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
