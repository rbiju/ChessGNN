from abc import ABC, abstractmethod
import torch.nn as nn

from .dyt import DyTNorm


class NormFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def norm(self, size: int) -> nn.Module:
        raise NotImplementedError


class LayerNormFactory(NormFactory):
    def __init__(self):
        super().__init__()

    def norm(self, size: int) -> nn.Module:
        return nn.LayerNorm(size)


class DyTNormFactory(NormFactory):
    def __init__(self):
        super().__init__()

    def norm(self, size: int) -> nn.Module:
        return DyTNorm(size)
