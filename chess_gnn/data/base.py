from abc import ABC, abstractmethod
from typing import Dict

from torch.utils.data import Dataset
from torch import Tensor


class BaseBERTDataset(Dataset, ABC):
    def __init__(self, file):
        self.file = file

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> Dict[str, Tensor]:
        raise NotImplementedError
