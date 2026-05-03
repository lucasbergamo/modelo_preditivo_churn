# Plano de Monitoramento — Churn Predictor MLP

> Documento que define como monitorar o modelo em produção, detectar degradação
> e responder a incidentes. Baseado em práticas de MLOps para modelos de classificação.

---

## 1. Contexto

O modelo é servido via FastAPI em modo batch diário (scores calculados para toda a base) com API disponível para consultas pontuais. O monitoramento opera em duas dimensões:

- **Operacional:** a API está respondendo corretamente?
- **De modelo:** as previsões ainda são confiáveis?

---

## 2. Métricas Operacionais

### O que monitorar

| Métrica | Descrição | Coleta |
|---|---|---|
| Latência p50 | Tempo mediano de resposta do `/predict` | Middleware de latência (structlog) |
| Latência p95 | Percentil 95 — detecta lentidão em casos extremos | Middleware de latência |
| Latência p99 | Percentil 99 — detecta outliers críticos | Middleware de latência |
| Taxa de erro (5xx) | Proporção de requests com erro interno | Logs da API |
| Taxa de erro (4xx) | Proporção de requests com schema inválido | Logs da API |
| Disponibilidade | % de tempo com API respondendo | Health check externo |

### Thresholds e alertas

| Métrica | Verde | Amarelo | Vermelho |
|---|---|---|---|
| Latência p50 | < 50ms | 50–200ms | > 200ms |
| Latência p95 | < 200ms | 200–500ms | > 500ms |
| Taxa de erro 5xx | < 0.1% | 0.1–1% | > 1% |
| Disponibilidade | > 99.9% | 99–99.9% | < 99% |

---

## 3. Métricas de Modelo

### 3.1 Distribuição de scores (monitoramento sem label)

O principal sinal de degradação disponível imediatamente (sem esperar labels reais) é a **mudança na distribuição dos scores de churn previstos**.

| Métrica | Baseline (treino) | Ação se desviar |
|---|---|---|
| Média dos scores | ~0.27 | Alerta se sair do intervalo [0.20, 0.35] |
| % classificados como churn (score ≥ 0.5) | ~26% | Alerta se sair do intervalo [20%, 35%] |
| Desvio padrão dos scores | Calcular no deploy | Alerta se aumentar > 30% |

### 3.2 Métricas de performance (com label — retroativas)

Após o ciclo de retenção (30–90 dias), os labels reais ficam disponíveis e permitem medir performance real:

| Métrica | Threshold mínimo aceitável | Ação se abaixo |
|---|---|---|
| AUC-ROC | 0.80 | Investigar + retreinar |
| Recall | 0.65 | Investigar + retreinar |
| F-beta(β=2) | 0.60 | Investigar + retreinar |

---

## 4. Data Drift — Monitoramento de Features

Data drift ocorre quando a distribuição das features de entrada muda ao longo do tempo — ex: crise econômica eleva `MonthlyCharges`, aquisição de concorrente muda perfil de contratos.

### Features prioritárias para monitorar

| Feature | Por que monitorar | Método |
|---|---|---|
| `tenure` | Mudança no perfil de novos clientes | PSI (Population Stability Index) |
| `MonthlyCharges` | Sensível a reajustes de preço | PSI + KS Test |
| `Contract` | Mudança em campanhas de fidelização | Chi-quadrado em frequências |
| `InternetService` | Expansão de fibra ou 5G | Chi-quadrado em frequências |
| `charges_per_month` | Feature derivada — amplifica drifts | PSI |

### Critério de alerta (PSI)

| PSI | Interpretação | Ação |
|---|---|---|
| < 0.10 | Sem drift significativo | Nenhuma |
| 0.10 – 0.20 | Drift moderado | Investigar causa |
| > 0.20 | Drift severo | Retreinar modelo |

---

## 5. Frequência de Monitoramento

| Tipo | Frequência | Responsável |
|---|---|---|
| Health check da API | A cada 5 minutos | Sistema automatizado |
| Latência e erros | Tempo real (streaming de logs) | Dashboard operacional |
| Distribuição de scores | Diária (junto ao batch) | Pipeline automatizado |
| PSI de features | Semanal | Engenheiro de ML |
| Performance com labels reais | Mensal (após fechar ciclo) | Engenheiro de ML |
| Auditoria de fairness (gênero/idade) | Trimestral | Engenheiro de ML + Compliance |

---

## 6. Triggers de Retreinamento

O modelo deve ser retreinado quando **qualquer uma** das condições abaixo for atendida:

1. AUC-ROC no período atual < 0.80
2. Recall no período atual < 0.65
3. PSI > 0.20 em qualquer feature prioritária
4. Mais de 6 meses desde o último treino
5. Mudança estrutural no negócio (novo produto, mudança de preços, fusão)

### Processo de retreinamento

```
1. Coletar novos dados rotulados (mínimo 3 meses)
2. Executar python -m src.data.pipeline (regenera gold)
3. Executar python -m src.training.train_baselines
4. Executar make train
5. Comparar métricas novo modelo vs modelo em produção no mesmo test set
6. Se novo modelo ≥ modelo atual em AUC-ROC e F-beta(β=2): promover
7. Manter modelo antigo como fallback por 30 dias
8. Atualizar Model Card com novos resultados
```

---

## 7. Playbook de Incidentes

### Cenário 1 — API retornando 5xx

```
1. Verificar logs: make run / logs do uvicorn
2. Verificar se mlp.pt e scaler.pkl existem em models/
3. Verificar se data/gold/*.parquet existem
4. Se arquivos faltando: rodar python -m src.data.pipeline + make train
5. Reiniciar API: make run
```

### Cenário 2 — Scores anômalos (todos próximos de 0 ou 1)

```
1. Verificar se scaler.pkl é o mesmo gerado pelo último pipeline
2. Verificar se as colunas do request batem com X_train.parquet
3. Rodar python -m src.evaluation.metrics para comparar com baseline
4. Se degradação confirmada: retreinar seguindo processo acima
```

### Cenário 3 — Recall caindo abaixo de 0.65

```
1. Verificar data drift nas features prioritárias (PSI)
2. Analisar se perfil de novos clientes mudou
3. Avaliar se threshold de 0.5 deve ser ajustado para 0.4 (aumenta recall)
4. Planejar retreinamento com dados recentes
```

---

## 8. Ferramentas Recomendadas para Produção

| Necessidade | Ferramenta Recomendada | Alternativa |
|---|---|---|
| Tracking de experimentos | MLflow (já implementado) | Weights & Biases |
| Monitoramento operacional | Prometheus + Grafana | Datadog |
| Data drift | Evidently AI | NannyML |
| Feature store | Feast | Hopsworks |
| Orquestração de retreino | Apache Airflow | Prefect |
