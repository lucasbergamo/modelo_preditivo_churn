import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP para classificação binária de churn.

    Arquitetura: Input → [Linear → BatchNorm → ReLU → Dropout] x N → Linear (logit)

    O modelo retorna logits brutos (sem Sigmoid). Use BCEWithLogitsLoss no treino
    e torch.sigmoid() na inferência para obter probabilidades. Essa separação é
    numericamente mais estável que aplicar Sigmoid dentro do modelo e usar BCELoss.

    Escolhas de arquitetura:
    - hidden_dims=(64, 32): funil progressivo — suficiente para 27 features, evita
      overfitting em dataset pequeno (~5k amostras). Alternativas: mais largo (256→128)
      para datasets maiores; mais profundo (64→32→16) raramente ajuda em dados tabulares.
    - BatchNorm1d: estabiliza gradientes e acelera convergência. Alternativa: LayerNorm
      (melhor para batches muito pequenos). Sem normalização: treino instável dado que
      as features têm escalas heterogêneas mesmo após StandardScaler.
    - Dropout(0.3): regularização — desativa 30% dos neurônios por batch, força
      redundância nas representações. Valor conservador para dataset pequeno.
      Alternativa: weight_decay no otimizador (L2), que também regulariza mas de forma
      global. Os dois podem ser usados juntos.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retorna logits — shape (batch,). Sigmoid aplicado só na inferência.
        return self.network(x).squeeze(1)
