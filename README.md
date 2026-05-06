# Modelo Preditivo de Churn — Telecom

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.15+-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ruff](https://img.shields.io/badge/linting-ruff-FCC21B?style=flat-square)
![Testes](https://img.shields.io/badge/testes-13%20passando-brightgreen?style=flat-square)

Rede Neural (MLP) para previsão de churn em operadora de telecomunicações.

## Sumário

- [Contexto](#contexto)
- [Stack Tecnológica](#stack-tecnológica)
- [Dataset](#dataset)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura de Dados — Medalhão](#arquitetura-de-dados--medalhão-bronze--silver--gold)
- [Setup e Instalação](#setup)
- [Reproduzir o pipeline completo](#reproduzir-o-pipeline-completo)
- [Configuração do Projeto](#configuração-do-projeto)
- [Boas Práticas](#boas-práticas)
- [Critérios de Avaliação](#critérios-de-avaliação)

## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa de um modelo preditivo de churn que classifique clientes com risco de cancelamento. O projeto abrange desde a análise exploratória até o modelo servido via API, aplicando boas práticas de engenharia de ML.

O modelo central é uma **rede neural (MLP)** treinada com **PyTorch**, comparada com baselines (**Scikit-Learn**) e rastreada com **MLflow**.

## Stack Tecnológica

| Biblioteca | Função |
|---|---|
| **PyTorch** | Construção e treinamento da rede neural (MLP) |
| **Scikit-Learn** | Pipelines de pré-processamento e modelos baseline |
| **MLflow** | Tracking de experimentos (parâmetros, métricas, artefatos) |
| **FastAPI** | API de inferência do modelo |
| **Pandera** | Validação de schema dos dados |
| **Pydantic** | Validação de request/response da API |
| **Ruff** | Linting e formatação de código |
| **Pytest** | Testes automatizados |

## Dataset

**Telco Customer Churn (IBM)** — dataset público de telecomunicações com variáveis tabulares.

| Propriedade | Valor |
|---|---|
| Fonte | IBM Sample Datasets |
| Registros | 7.043 |
| Features | 21 (19 features + 1 ID + 1 target) |
| Target | `Churn` (Yes/No — classificação binária) |
| Localização | `data/bronze/telco_customer_churn.csv` |

**Variáveis do dataset:**

| Variável | Tipo | Descrição |
|---|---|---|
| `customerID` | string | Identificador único do cliente |
| `gender` | categórica | Gênero (Male/Female) |
| `SeniorCitizen` | binária | Cliente idoso (0/1) |
| `Partner` | categórica | Possui parceiro (Yes/No) |
| `Dependents` | categórica | Possui dependentes (Yes/No) |
| `tenure` | numérica | Meses como cliente |
| `PhoneService` | categórica | Serviço de telefone (Yes/No) |
| `MultipleLines` | categórica | Múltiplas linhas (Yes/No/No phone service) |
| `InternetService` | categórica | Tipo de internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | categórica | Segurança online (Yes/No/No internet service) |
| `OnlineBackup` | categórica | Backup online (Yes/No/No internet service) |
| `DeviceProtection` | categórica | Proteção de dispositivo (Yes/No/No internet service) |
| `TechSupport` | categórica | Suporte técnico (Yes/No/No internet service) |
| `StreamingTV` | categórica | Streaming de TV (Yes/No/No internet service) |
| `StreamingMovies` | categórica | Streaming de filmes (Yes/No/No internet service) |
| `Contract` | categórica | Tipo de contrato (Month-to-month/One year/Two year) |
| `PaperlessBilling` | categórica | Fatura digital (Yes/No) |
| `PaymentMethod` | categórica | Método de pagamento |
| `MonthlyCharges` | numérica | Valor mensal |
| `TotalCharges` | numérica | Valor total acumulado |
| `Churn` | target | Cancelou o serviço (Yes/No) |

## Resultados

Comparação de modelos no conjunto de teste (split estratificado 70/15/15, seed=42):

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | F-beta(β=2) |
|---|---|---|---|---|---|
| **MLP PyTorch** | **0.844** | 0.652 | **0.624** | 0.701 | 0.668 |
| Logistic Regression | 0.845 | **0.669** | 0.620 | **0.765** | **0.699** |
| Random Forest | 0.823 | 0.633 | 0.538 | 0.466 | 0.493 |
| Dummy (baseline) | 0.500 | 0.266 | 0.000 | 0.000 | 0.000 |

O MLP é competitivo com a Regressão Logística (diferença de 0.001 em AUC-ROC), resultado esperado em dados tabulares com ~5k amostras. A vantagem da rede neural está na demonstração de domínio técnico: early stopping, batching, regularização com Dropout e BatchNorm, rastreamento de loss por época no MLflow.

Detalhes completos em [`docs/model_card.md`](docs/model_card.md).

## Critérios de Avaliação

| Critério | Peso | Descrição |
|---|---|---|
| Qualidade do código e estrutura | 20% | Organização, modularidade, SOLID, linting sem erros |
| Rede neural (PyTorch) | 25% | MLP funcional, early stopping, comparação com baselines |
| Pipeline e reprodutibilidade | 15% | Pipeline sklearn, seeds, pyproject.toml, instala do zero |
| API de inferência | 15% | FastAPI funcional, Pydantic, logging, testes passando |
| Documentação e Model Card | 10% | Model Card completa, README claro, plano de monitoramento |
| Vídeo STAR | 10% | Clareza, cobertura dos 4 elementos STAR, dentro de 5 minutos |
| Bônus: deploy em nuvem | 5% | API acessível via URL pública |

## Plano de Desenvolvimento

### Etapa 1 — Entendimento e Preparação

> **Foco:** formulação do problema, exploração de dados e construção de baselines.

- [x] Montar estrutura do projeto (`src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`)
- [x] Configurar `pyproject.toml` (dependências, ruff, pytest)
- [x] Criar `Makefile` com comandos padrão (lint, test, run)
- [x] Download do dataset Telco Customer Churn (IBM) — 7.043 registros, 21 colunas
- [x] Configurar módulo de utilidades (`src/utils/`): config, logger (structlog), reproducibility (seeds)
- [x] Configurar `.gitignore` para projeto de ML (dados, modelos, mlruns, ambientes virtuais)
- [x] Notebook de EDA completa (volume, qualidade, distribuição, data readiness)
- [x] Preencher ML Canvas (stakeholders, métricas de negócio, SLOs)
- [x] Definir métricas técnicas: AUC-ROC, PR-AUC, F1, Recall, F-beta(β=2)
- [x] Pipeline de dados bronze→silver→gold (`src/data/`) com split estratificado 70/15/15
- [x] Treinar baselines: `DummyClassifier` + Regressão Logística + Random Forest (Scikit-Learn)
- [x] Registrar experimentos no MLflow (parâmetros, métricas por modelo)

**Entregável:** notebook de EDA + baselines registrados no MLflow.

### Etapa 2 — Modelagem com Redes Neurais

> **Foco:** construção, treinamento e avaliação de MLP com PyTorch.

- [x] Construir MLP em PyTorch — arquitetura 64→32, ReLU, BatchNorm, Dropout(0.3)
- [x] Implementar training loop com early stopping (patience=15) e batching (batch_size=64)
- [x] Registrar experimento MLP no MLflow (parâmetros, loss por época, métricas, artefato .pt)
- [x] Comparar MLP vs. baselines usando ≥ 4 métricas — tabela comparativa
- [x] Analisar trade-off de custo (falso positivo vs. falso negativo)

**Entregável:** tabela comparativa de modelos + MLP treinado + artefatos no MLflow.

### Etapa 3 — Engenharia e API

> **Foco:** refatoração profissional, API de inferência e pacote reutilizável.

- [x] Estrutura modular em `src/` com separação clara de responsabilidades
- [x] Logging estruturado com structlog (sem print())
- [x] Linting e formatação com ruff sem erros (`make lint` verde)
- [x] 10 testes passando: smoke test, schema Pandera (3), API (6) — `make test` verde
- [x] API FastAPI: `/predict`, `/health`, validação Pydantic, middleware de latência
- [x] Feature engineering: `charges_per_month`, `num_services`, `is_new_customer`

**Entregável:** repositório refatorado + API funcional + testes passando.

### Etapa 4 — Documentação e Entrega Final

> **Foco:** consolidação, documentação e vídeo de apresentação.

- [x] Model Card completo (arquitetura, métricas, limitações, ética, reprodução)
- [x] Plano de monitoramento (métricas, PSI, thresholds, playbook de incidentes)
- [x] ML Canvas (proposta de valor, stakeholders, features, riscos)
- [x] jornada.md — storytelling vivo com todas as decisões técnicas documentadas
- [ ] Gravar vídeo de 5 min (método STAR) demonstrando o projeto
- [ ] (Opcional) Deploy da API em nuvem (AWS/Azure/GCP) com endpoint público

**Entregável:** repositório final + vídeo STAR + (opcional) URL do deploy em nuvem.

## Estrutura do Projeto

```
modelo_preditivo_churn/
├── src/                        # código-fonte principal
│   ├── __init__.py
│   ├── data/                   # carregamento, split, pré-processamento
│   ├── features/               # feature engineering
│   ├── models/                 # definição MLP PyTorch + baselines sklearn
│   ├── training/               # loop de treinamento + MLflow tracking
│   ├── evaluation/             # métricas, comparações entre modelos
│   ├── api/                    # FastAPI (endpoints /predict, /health)
│   └── utils/
│       ├── config.py           # constantes: SEED, paths, nomes MLflow
│       ├── logger.py           # logging estruturado com structlog
│       └── reproducibility.py  # set_global_seed (random, numpy, torch)
├── data/                          # arquitetura medalhão
│   ├── bronze/                 # dado bruto — CSV original, sem alteração
│   ├── silver/                 # dado limpo — tipagem, nulos tratados, encoding
│   └── gold/                   # dado model-ready — features, splits train/val/test
├── models/                     # artefatos salvos (.pt, .pkl)
├── notebooks/                  # EDA, experimentos exploratórios
├── tests/                      # testes pytest (smoke, schema, API)
├── docs/                       # ML Canvas, Model Card
├── pyproject.toml              # single source of truth (deps, ruff, pytest)
├── Makefile                    # comandos: install, lint, format, test, run, train, mlflow
├── .gitignore                  # ignora dados, modelos, mlruns, venvs, caches
└── README.md
```

## Arquitetura de Dados — Medalhão (Bronze / Silver / Gold)

O pipeline de dados segue a **arquitetura medalhão**, garantindo rastreabilidade e separação de responsabilidades em cada camada:

| Camada | Diretório | Descrição | Exemplo |
|---|---|---|---|
| **Bronze** | `data/bronze/` | Dado bruto, exatamente como veio da fonte. Nenhuma transformação aplicada. | `telco_customer_churn.csv` (CSV original da IBM) |
| **Silver** | `data/silver/` | Dado limpo e padronizado. Tipagem corrigida, nulos tratados, encoding de categóricas, remoção de duplicatas. | Dados com `TotalCharges` convertido para float, colunas categóricas codificadas |
| **Gold** | `data/gold/` | Dado pronto para modelagem. Features engineered, normalização aplicada, splits train/val/test gerados. | Arrays numpy ou tensors PyTorch prontos para o DataLoader |

```mermaid
flowchart LR
    A["📄 Bronze\nCSV original"] -->|"bronze_to_silver\nadd_features"| B["🥈 Silver\nlimpo + tipado"]
    B -->|"silver_to_features\nget_dummies"| C["🥇 Gold\ntrain / val / test"]
    C -->|"StandardScaler\nMLP PyTorch"| D["🧠 Modelo\nmlp.pt + scaler.pkl"]
    D -->|"FastAPI\n/predict"| E["🌐 API\nJSON response"]

    style A fill:#cd7f32,color:#fff
    style B fill:#c0c0c0,color:#000
    style C fill:#ffd700,color:#000
    style D fill:#6366f1,color:#fff
    style E fill:#059669,color:#fff
```

## Configuração do Projeto

### pyproject.toml

Single source of truth do projeto. Gerencia:

- **Dependências de produção:** torch, scikit-learn, mlflow, fastapi, uvicorn, pandas, numpy, pandera, pydantic, httpx, structlog
- **Dependências de desenvolvimento:** pytest, pytest-cov, ruff, ipykernel, matplotlib, seaborn
- **Ruff (linter):** target Python 3.11, line-length 100, regras: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, flake8-simplify
- **Pytest:** testpaths `tests/`, pythonpath `.`, verbose com traceback curto

### Makefile

| Comando | Descrição |
|---|---|
| `make install` | Instala o projeto em modo editável com deps de dev |
| `make lint` | Verifica linting e formatação com ruff |
| `make format` | Corrige linting e formata código automaticamente |
| `make test` | Executa testes com pytest |
| `make run` | Sobe a API FastAPI em `localhost:8000` |
| `make data` | Executa o pipeline de dados (bronze → gold) |
| `make train` | Executa `data` + `train-baselines` + `train-mlp` em sequência |
| `make train-mlp` | Treina apenas o MLP (requer `make data` antes) |
| `make train-baselines` | Treina apenas os baselines sklearn no MLflow |
| `make mlflow` | Abre a UI do MLflow em `localhost:5000` |
| `make clean` | Remove caches (\_\_pycache\_\_, .pytest_cache, .ruff_cache) |

### Módulo de Utilidades (src/utils/)

| Arquivo | Função |
|---|---|
| `config.py` | Constantes globais: `SEED=42`, paths medalhão (`DATA_BRONZE_DIR`, `DATA_SILVER_DIR`, `DATA_GOLD_DIR`, `MODELS_DIR`), `TARGET_COL`, configuração MLflow |
| `logger.py` | Factory de logger estruturado usando `structlog` — substitui `print()` por logs com timestamp ISO, nível e contexto |
| `reproducibility.py` | `set_global_seed()` — fixa seeds de `random`, `numpy` e `torch` (CPU + CUDA) para garantir reprodutibilidade |

## Boas Práticas

- Seeds fixados para reprodutibilidade
- Validação cruzada estratificada
- Model Card documentando limitações e vieses
- Testes automatizados (≥ 3: smoke test, schema, API)
- Logging estruturado (sem `print()`)
- Linting com ruff sem erros
- Histórico de commits limpo e significativo

## Setup

```bash
# Clonar repositório
git clone https://github.com/lucasbergamo/modelo_preditivo_churn.git
cd modelo_preditivo_churn

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -e ".[dev]"
```

### Dataset

O dataset **Telco Customer Churn (IBM)** está incluído no repositório em `data/bronze/telco_customer_churn.csv`.

Decisão intencional: o arquivo é público (sem PII), tem 1.4MB e sua presença garante reprodutibilidade completa independente de disponibilidade de serviços externos. A seed fixada em 42 assegura que qualquer pessoa que rodar `make retrain` obterá exatamente os mesmos splits e métricas.

Fonte original: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Reproduzir o pipeline completo

> **Atenção:** `make train` é pré-requisito obrigatório antes de `make test` e `make run`.
> Os artefatos de modelo (`models/mlp.pt`, `models/scaler.pkl`) e os splits de dados
> (`data/gold/`) são gitignored e precisam ser gerados localmente.

```bash
# 1. Gera gold data + treina baselines + treina MLP (~3-5 min)
make train

# 2. Verifica que todos os 13 testes passam
make test

# 3. Verifica linting
make lint

# 4. Sobe a API em localhost:8000
make run
```
