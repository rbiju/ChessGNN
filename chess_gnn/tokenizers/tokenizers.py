from typing import Optional

import torch
from torch import Tensor

from .base import ChessTokenizer


class SimpleChessTokenizer(ChessTokenizer):
    def __init__(self, board_str: Optional[str] = None):
        super().__init__(board_str)

    def tokenize(self, board_str: str) -> Tensor:
        seq_idxs = [self.inverse_vocab[token] for token in board_str]
        return torch.tensor(seq_idxs, dtype=torch.long)
