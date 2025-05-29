from abc import ABC, abstractmethod
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
