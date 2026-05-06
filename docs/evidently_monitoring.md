# Plano de Implementação — Monitoramento com Evidently

Documento de trabalho futuro. O `monitoring_plan.md` define O QUÊ monitorar;
este documento define COMO implementar usando Evidently.

## Contexto

A API `/predict` hoje loga latência e scores via structlog (console), mas não
persiste os vetores de features em disco. O Evidently precisa comparar duas
janelas de dados — referência (treino) vs produção (requests recentes) — para
detectar data drift. A parte que falta é a persistência dos requests.

## O que o Evidently entrega

- Relatório HTML interativo com drift por feature
- PSI (Population Stability Index) para features contínuas
- Chi-quadrado para features categóricas
- Alertas configuráveis por threshold
- Dataset de referência: `data/gold/X_train.parquet` (já existe)

## Implementação em duas etapas

---

### Etapa 1 — Persistir requests da API

**Arquivo:** `src/api/app.py`

Adicionar chamada ao preditor que salva o vetor de features usado na inferência.
Não salvar o `CustomerInput` bruto — salvar o DataFrame pós-feature-engineering
(o mesmo que entra no modelo), para que a comparação com `X_train.parquet` seja
direta.

**Abordagem recomendada:** append em CSV com lock de arquivo. Simples, sem
dependências extras, suficiente para volume baixo/médio.

```python
# src/api/logger_requests.py
import threading
from pathlib import Path
import pandas as pd

PRODUCTION_LOG = Path("data/production/requests.csv")
_lock = threading.Lock()

def log_request(feature_row: pd.DataFrame) -> None:
    PRODUCTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        feature_row.to_csv(
            PRODUCTION_LOG,
            mode="a",
            header=not PRODUCTION_LOG.exists(),
            index=False,
        )
```

Em `predictor.py`, o método `predict()` já constrói o DataFrame via
`_build_feature_row()` — basta retorná-lo também e logar antes de escalar:

```python
# src/api/predictor.py — predict() modificado
def predict(self, customer: CustomerInput) -> tuple[float, bool]:
    df = _build_feature_row(customer, self.train_cols)
    log_request(df)                          # ← adicionar esta linha
    scaled = self.scaler.transform(df)
    ...
```

**Atenção:** adicionar `data/production/` ao `.gitignore`.

---

### Etapa 2 — Script de relatório Evidently

**Novo arquivo:** `src/monitoring/drift_report.py`

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from pathlib import Path

REFERENCE = Path("data/gold/X_train.parquet")
PRODUCTION_LOG = Path("data/production/requests.csv")
OUTPUT_DIR = Path("monitoring")


def run_drift_report() -> None:
    reference = pd.read_parquet(REFERENCE)
    current = pd.read_csv(PRODUCTION_LOG)

    OUTPUT_DIR.mkdir(exist_ok=True)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(OUTPUT_DIR / "drift_report.html")
    print(f"Relatório salvo em {OUTPUT_DIR / 'drift_report.html'}")


if __name__ == "__main__":
    run_drift_report()
```

**Dependência a adicionar no `pyproject.toml`:**

```toml
[project.optional-dependencies]
dev = [
    ...
    "evidently>=0.4",
]
```

**Comando a adicionar no `Makefile`:**

```makefile
monitor:
	python -m src.monitoring.drift_report
```

---

### Etapa 3 — Thresholds e alertas (opcional)

Após o relatório funcionar, adicionar verificação programática dos thresholds
definidos no `monitoring_plan.md`:

- PSI > 0.20 em qualquer feature → log de warning + e-mail/Slack
- Dataset drift detected (Evidently flag) → acionar `make retrain`

Isso pode ser feito com `evidently.test_suite.TestSuite` e
`evidently.tests.TestNumberOfDriftedColumns`.

---

## Ordem de execução recomendada

1. `make install` (após adicionar `evidently` ao pyproject.toml)
2. `make run` — subir a API e fazer alguns requests manuais para popular o CSV
3. `make monitor` — gerar o primeiro relatório
4. Abrir `monitoring/drift_report.html` no browser
5. Validar que as distribuições de referência batem com o esperado

## Estimativa de esforço

| Etapa | Tempo |
|---|---|
| Etapa 1 — persistência de requests | ~1h |
| Etapa 2 — script Evidently | ~30min |
| Etapa 3 — thresholds programáticos | ~1h |
| Testes + ajustes | ~30min |
| **Total** | **~3h** |
