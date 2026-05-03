# Model Card — Churn Predictor MLP

> Formato baseado no padrão Model Cards for Model Reporting (Mitchell et al., 2019, Google).
> Objetivo: documentar o modelo de forma transparente para avaliadores, usuários e stakeholders.

---

## 1. Detalhes do Modelo

| Atributo | Valor |
|---|---|
| Nome | Churn Predictor MLP |
| Versão | 1.0.0 |
| Tipo | Rede Neural MLP — Classificação Binária |
| Framework | PyTorch 2.x |
| Arquivo | `models/mlp.pt` |
| Desenvolvido por | Projeto Tech Challenge Fase 01 — FIAP Pós-Tech MLET |
| Data | Maio 2026 |

### Arquitetura

```
Input (29 features)
    → Linear(29, 64) → BatchNorm1d(64) → ReLU → Dropout(0.3)
    → Linear(64, 32) → BatchNorm1d(32) → ReLU → Dropout(0.3)
    → Linear(32, 1)
    → [Sigmoid aplicado apenas na inferência]
```

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| hidden_dims | (64, 32) | Funil progressivo — padrão para dados tabulares com ~30 features |
| dropout | 0.3 | Regularização conservadora para dataset ~5k amostras |
| optimizer | Adam, lr=1e-3 | Adaptativo por parâmetro, converge bem em tabulares |
| loss | BCEWithLogitsLoss | Numericamente estável vs BCELoss + Sigmoid no modelo |
| pos_weight | 2.0 | Penaliza falsos negativos — custo assimétrico do negócio |
| batch_size | 64 | Balanço entre estabilidade e velocidade |
| early_stopping | patience=15 | Para quando val_loss não melhora por 15 épocas |
| seed | 42 | Reprodutibilidade total |

---

## 2. Uso Pretendido

### Uso adequado
- Priorizar clientes para ações de retenção proativa
- Apoiar decisões da equipe comercial — modelo é suporte, não substituto humano
- Consulta pontual de risco via API durante atendimento ao cliente

### Uso inadequado
- ❌ Negar serviços ou aplicar penalidades a clientes baseado no score
- ❌ Tomar decisões automáticas sem revisão humana
- ❌ Aplicar em contextos muito distintos do treino (ex: clientes B2B, outros países)
- ❌ Operar sem retreinamento após 6 meses de mudança de mercado

---

## 3. Dados de Treinamento

| Atributo | Detalhe |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Total de registros | 7.043 clientes |
| Split treino | 4.930 (70%) — estratificado por Churn |
| Split validação | 1.056 (15%) — usado para early stopping |
| Split teste | 1.057 (15%) — usado apenas para avaliação final |
| Período | Snapshot único, ~2017 |
| Balanceamento | 26,5% churn / 73,5% não-churn |

### Pipeline de pré-processamento

1. **Bronze → Silver:** correção de `TotalCharges` (string → float), remoção de `customerID`, encoding binário de Yes/No → 0/1
2. **Feature Engineering:** `charges_per_month`, `num_services`, `is_new_customer`
3. **Silver → Gold:** one-hot encoding de variáveis categóricas, `StandardScaler` fitado no treino, split estratificado 70/15/15

---

## 4. Resultados de Avaliação

### Métricas no test set (dados nunca vistos durante treino)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | F-beta(β=2) |
|---|---|---|---|---|---|
| **MLP PyTorch** | 0.844 | 0.652 | 0.624 | 0.701 | 0.668 |
| Logistic Regression | 0.845 | 0.669 | 0.620 | 0.765 | 0.699 |
| Random Forest | 0.823 | 0.633 | 0.538 | 0.466 | 0.493 |
| Dummy (baseline) | 0.500 | 0.266 | 0.000 | 0.000 | 0.000 |

### Interpretação das métricas

**AUC-ROC 0.844:** em 84,4% das comparações par-a-par (churner vs não-churner), o modelo ranqueia corretamente o churner com score maior.

**Recall 0.701:** o modelo identifica 70,1% dos clientes que realmente vão cancelar. Os 29,9% restantes são falsos negativos — clientes que passam despercebidos.

**F-beta(β=2) 0.668:** métrica primária do negócio. Penaliza 4× mais o falso negativo (não detectar churn) que o falso positivo (alarmar cliente que ficaria).

### Por que MLP e LogReg estão empatados

A diferença de 0.001 no AUC-ROC é estatisticamente insignificante. Para datasets tabulares com ~5k amostras e features já bem normalizadas, modelos lineares tendem a competir com redes neurais. Em produção com histórico maior de clientes, o MLP tenderia a superar progressivamente.

---

## 5. Análise por Subgrupos

| Subgrupo | Observação |
|---|---|
| SeniorCitizen (9,7% do dataset) | Sub-representado — métricas podem ser menos confiáveis para esse grupo |
| Clientes < 12 meses (is_new_customer) | Taxa de churn ~40% — modelo mais agressivo nesse grupo |
| Contrato mensal | 43% de churn — principal driver do modelo |
| Fiber optic sem suporte | Combinação de alto risco — interação capturada pelo MLP |

---

## 6. Limitações Conhecidas

**Temporais:**
- Dataset de ~2017 — comportamentos pós-pandemia (trabalho remoto, streaming) podem não estar representados
- Snapshot único sem séries temporais — não captura tendência de uso ao longo do tempo

**De dados:**
- Sem histórico de reclamações ou NPS — satisfação do cliente não está nos dados
- Sem dados geográficos — variações regionais de concorrência não são capturadas
- Sem segmentação de valor do cliente — um churner de R$ 200/mês recebe mesmo tratamento que um de R$ 50/mês

**Do modelo:**
- Threshold padrão de 0.5 pode não ser ótimo para todos os segmentos — ajuste por faixa de receita é recomendado
- Não é explicável por padrão — para justificar decisões individuais, SHAP values devem ser implementados

---

## 7. Considerações Éticas

**Variáveis sensíveis presentes:**
- `gender` — incluído pois análise prévia indica correlação com churn; deve ser monitorado para não gerar discriminação
- `SeniorCitizen` — grupo vulnerável, ações de retenção devem ser revisadas para não resultar em práticas abusivas

**Recomendação:** antes do deploy em produção, realizar auditoria de fairness comparando métricas (Recall, FPR) entre subgrupos de gênero e faixa etária.

---

## 8. Melhorias para Produção

Funcionalidades identificadas durante o desenvolvimento que não foram implementadas por estarem fora do escopo do Tech Challenge, mas representam o próximo nível para um deploy real.

### MLflow — Dataset Logging
Registrar formalmente o dataset usado em cada run via `mlflow.log_input()`. Hoje os runs do MLflow não têm a coluna "Dataset" preenchida. Em produção, onde o dataset muda periodicamente, isso permite rastrear "esse modelo foi treinado com os dados de qual versão/semana".

```python
# Adicionar em src/training/train.py após carregar os dados
dataset = mlflow.data.from_pandas(df, source="data/gold/X_train.parquet", name="telco-churn-train")
mlflow.log_input(dataset, context="training")
```

### MLflow — Model Registry
Promover modelos formalmente para "Staging" ou "Production" em vez de sobrescrever o `mlp.pt` diretamente. Permite rollback imediato se um novo modelo for pior: basta rebaixar a versão nova e promover a anterior, sem redeployar código.

Requer migrar o backend do MLflow de filesystem para SQLite (hoje gera o `FutureWarning` no terminal):

```bash
# mlflow.db no lugar de ./mlruns
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

### SHAP Values — Explicabilidade
O modelo atual não é explicável por padrão — não é possível justificar individualmente por que um cliente recebeu score 0.82. SHAP values calculam a contribuição de cada feature para cada previsão, permitindo que o atendente saiba: "esse cliente tem score alto porque tem contrato mensal + fibra + 3 meses de casa".

### Threshold por segmento de receita
O threshold padrão de 0.5 trata igual um cliente de R$120/mês e um de R$30/mês. Em produção, clientes de alto valor justificam threshold menor (mais agressivo em detectar churn) e ações de retenção mais custosas.

---

## 9. Como Reproduzir

```bash
# 1. Clonar e instalar
git clone https://github.com/lucasbergamo/modelo_preditivo_churn.git
cd modelo_preditivo_churn
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Rodar pipeline completo
python -m src.data.pipeline          # bronze → silver → gold + scaler.pkl

# 3. Treinar modelos
python -m src.training.train_baselines   # baselines sklearn
make train                               # MLP PyTorch

# 4. Avaliar
python -m src.evaluation.metrics        # comparativo no test set

# 5. Subir API
make run                                 # FastAPI em localhost:8000

# 6. Rodar testes
make test                                # 10 testes — deve passar em ~2 min
```
