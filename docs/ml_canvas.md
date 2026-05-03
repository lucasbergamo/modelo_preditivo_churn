# ML Canvas — Churn Predictor Telecom

> Documento de alinhamento estratégico entre negócio e machine learning.
> Preenchido retrospectivamente após conclusão do projeto — decisões refletem
> o raciocínio que guiou cada etapa de desenvolvimento.

---

## 1. Proposta de Valor

**Problema de negócio:**
Uma operadora de telecomunicações enfrenta taxa de churn de ~26,5% — aproximadamente 1 em cada 4 clientes cancela o serviço. O custo de adquirir um novo cliente é 5 a 7× maior que o custo de reter um existente. Sem um sistema preditivo, a equipe de retenção age de forma reativa (após o cancelamento) ou aleatória (sem priorização por risco).

**Solução ML:**
Modelo preditivo que classifica clientes por probabilidade de cancelamento, permitindo à equipe de retenção agir preventivamente com ações direcionadas (oferta de desconto, upgrade de plano, contato proativo).

**Valor gerado:**
- Redução da taxa de churn via retenção direcionada
- Priorização da equipe de retenção para clientes de maior risco e maior valor
- Substituição de campanhas genéricas por intervenções personalizadas

---

## 2. Stakeholders

| Papel | Interesse |
|---|---|
| Diretoria comercial | Redução do churn, aumento do LTV (Lifetime Value) |
| Equipe de retenção | Lista priorizada de clientes para contato |
| TI / Engenharia | API confiável, baixa latência, fácil manutenção |
| Compliance | Uso ético dos dados, sem discriminação por gênero ou idade |

---

## 3. Decisão que o Modelo Apoia

> **"Este cliente deve receber uma ação de retenção proativa?"**

O modelo não decide a ação — ele prioriza quem recebe atenção. A decisão final é humana. O modelo é um **sistema de apoio à decisão**, não um sistema autônomo.

**Fluxo de uso:**
```
Batch diário → Score de churn por cliente → Fila priorizada → Equipe de retenção → Ação
```

---

## 4. Dados

| Atributo | Detalhe |
|---|---|
| Fonte | IBM Telco Customer Churn Dataset (público) |
| Volume | 7.043 clientes, 21 variáveis |
| Janela temporal | Snapshot único — sem séries temporais |
| Target | `Churn` — cancelou (Yes/No) nos últimos meses |
| Balanceamento | 26,5% positivos (churn) / 73,5% negativos |

**Riscos de qualidade identificados:**
- `TotalCharges` veio como string com espaços em branco para clientes novos (tenure=0) — tratado no pipeline silver
- Dataset de 2017 — pode não refletir comportamentos pós-pandemia
- Sem informações geográficas, histórico de reclamações ou dados de uso

---

## 5. Features Principais

| Feature | Tipo | Importância | Raciocínio |
|---|---|---|---|
| `Contract` | Categórica | Alta | Mensal → 43% churn; 2 anos → 3% churn |
| `tenure` | Numérica | Alta | Primeiros 12 meses: risco crítico |
| `InternetService` | Categórica | Alta | Fiber optic correlaciona com churn maior |
| `num_services` | Derivada | Média | Switching cost: mais serviços = menos churn |
| `charges_per_month` | Derivada | Média | Detecta cobrança desproporcional ao tempo |
| `TechSupport` | Binária | Média | Sem suporte aumenta insatisfação |
| `MonthlyCharges` | Numérica | Média | Clientes com cobranças altas são mais sensíveis |
| `is_new_customer` | Derivada | Média | Primeiros 12 meses têm churn ~40% vs ~15% depois |

---

## 6. Métricas de Sucesso

### Métricas técnicas (test set)

| Métrica | Threshold mínimo | Resultado obtido |
|---|---|---|
| AUC-ROC | > 0.80 | **0.844** ✅ |
| Recall | > 0.70 | **0.701** ✅ |
| F-beta(β=2) | > 0.60 | **0.668** ✅ |
| PR-AUC | > 0.55 | **0.652** ✅ |

### Métrica de negócio (estimada)

Assumindo:
- 7.000 clientes ativos, 26,5% em risco = ~1.855 churners potenciais
- Recall 0.70 → modelo identifica ~1.300 antes de cancelarem
- Taxa de sucesso de retenção de 30% com ação proativa
- Ticket médio mensal de R$ 65

**Valor estimado por ciclo:** ~1.300 × 30% × R$ 65 × 12 meses = **~R$ 3 milhões/ano**

---

## 7. Arquitetura de Inferência

**Escolha: Batch (diária) — não real-time**

**Justificativa:** ações de retenção têm latência natural de horas a dias (ligação, e-mail, oferta). Processar scores em batch diário é suficiente e elimina complexidade operacional de streaming.

**Alternativa real-time:** FastAPI `/predict` implementada para uso pontual (consulta de risco de um cliente específico durante atendimento) e para demonstração do Tech Challenge.

---

## 8. Riscos e Limitações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Data drift (comportamento muda com o tempo) | Alta | Alto | Monitoramento mensal de distribuição de features |
| Viés em SeniorCitizen | Média | Médio | Analisar métricas por subgrupo antes do deploy |
| Overfitting ao dataset de 2017 | Média | Alto | Retreinar com dados recentes a cada 6 meses |
| Falsos negativos em clientes de alto valor | Média | Alto | Threshold ajustável por segmento de receita |

---

## 9. O que NÃO fazer com este modelo

- ❌ Não usar para negar serviços a clientes
- ❌ Não usar como única fonte de decisão sem revisão humana
- ❌ Não aplicar em segmentos muito diferentes do dataset de treino (ex: clientes corporativos)
- ❌ Não operar por mais de 6 meses sem retreinamento com dados recentes
