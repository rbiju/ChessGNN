from abc import ABC, abstractmethod
import torch.nn as nn

from .dyt import DyTNorm


class NormFactory(ABC):
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def norm(self) -> nn.Module:
        raise NotImplementedError


class LayerNormFactory(NormFactory):
    def __init__(self, size):
        super().__init__(size)

    def norm(self) -> nn.Module:
        return nn.LayerNorm(self.size)


class DyTNormFactory(NormFactory):
    def __init__(self, size):
        super().__init__(size)

    def norm(self) -> nn.Module:
        return DyTNorm(self.size)
