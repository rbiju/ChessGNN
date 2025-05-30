from abc import ABC, abstractmethod
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn


class ChessEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class ChessEngineEncoder(ChessEncoder):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class ChessBackbone(ABC, pl.LightningModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_encoder(self) -> ChessEncoder:
        raise NotImplementedError


class ChessEngine(ChessBackbone):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_encoder(self) -> ChessEngineEncoder:
        raise NotImplementedError


@dataclass
class EngineLossWeights:
    from_loss: float = 1.0
    to_loss: float = 1.0
    win_prediction_loss: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.from_loss < 0 or self.to_loss < 0 or self.win_prediction_loss < 0:
            raise ValueError(
                f"Loss proportions must be positive: {self.from_loss, self.to_loss, self.win_prediction_loss}.")


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: list[int], out_dim: int, activation: str = 'gelu',
                 dropout_p: float = 0.1, skip: bool = False) -> None:
        super().__init__()
        activation_mapping = {'gelu': nn.GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_p = dropout_p
        try:
            activation = activation_mapping[activation.lower()]
            self.activation = activation()
        except KeyError:
            raise RuntimeError(
                f'Unsupported activation type: {activation}. Supported types are: {activation_mapping.keys()}')
        self.skip = skip
        if skip:
            self.skip_connector = nn.Linear(self.in_dim, self.hidden_dim[-1])
            self.norm = nn.LayerNorm(self.hidden_dim[-1])

        modules = []
        for dim_in, dim_out in zip([in_dim, *hidden_dim[:-1]], [*hidden_dim]):
            modules.append(nn.Linear(dim_in, dim_out))
            modules.append(self.activation)
            modules.append(nn.Dropout(p=self.dropout_p))

        self.mlp = nn.Sequential(*modules)
        self.last_layer = nn.Linear(self.hidden_dim[-1], self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip:
            residual = self.skip_connector(x)
            x = self.mlp(x)
            x = x + residual
            x = self.norm(x)
            x = self.last_layer(x)
            return x
        else:
            x = self.mlp(x)
            x = self.last_layer(x)
            return x
