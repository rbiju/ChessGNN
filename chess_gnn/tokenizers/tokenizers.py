from .base import ChessTokenizer
from torch import Tensor
import torch.nn.functional as F


class OneHotChessTokenizer(ChessTokenizer):
    def __init__(self, board_str: str):
        super().__init__(board_str)

    def tokenize(self, board_str: str) -> Tensor:
        seq_idxs = [self.inverse_vocab[token] for token in board_str]
        return F.one_hot(Tensor(seq_idxs), self.vocab_size)
