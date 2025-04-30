from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, List

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

from .lamb import Lamb


class OptimizerFactory(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def optimizer(self, params: Iterable[Tuple[str, Parameter]]) -> Optimizer:
        raise NotImplementedError

    @staticmethod
    def named_parameters_to_parameters(params: Iterable[Tuple[str, Parameter]]) -> Iterable[Parameter]:
        params_ = []

        for n, p in params:
            params_.append(p)
        return params_


class AdamWFactory(OptimizerFactory):
    def __init__(self, learning_rate: float, weight_decay: float, betas: Optional[List[float]] = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if betas is None:
            self.betas: Tuple[float, float] = (0.9, 0.95)
        else:
            if len(betas) != 2:
                raise ValueError("`betas` must have length 2.")
            self.betas = tuple(betas)

    def optimizer(self, params: Iterable[Tuple[str, Parameter]]) -> Optimizer:
        params = self.named_parameters_to_parameters(params)
        return torch.optim.AdamW(params=params,
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay,
                                 betas=self.betas)


class LambFactory(OptimizerFactory):
    def __init__(self, learning_rate: float = 1e-3,
                 weight_decay: float = 0,
                 eps: float = 1e-6,
                 betas: Optional[List[float]] = None,
                 adam: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.adam = adam
        if betas is None:
            self.betas: Tuple[float, float] = (0.9, 0.999)
        else:
            if len(betas) != 2:
                raise ValueError("`betas` must have length 2.")
            self.betas = tuple(betas)

    def optimizer(self, params: Iterable[Tuple[str, Parameter]]) -> Optimizer:
        params = self.named_parameters_to_parameters(params)
        return Lamb(params=params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=self.betas,
                    eps=self.eps,
                    adam=self.adam)
