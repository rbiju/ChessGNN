from abc import abstractmethod, ABC
from typing import Optional
from torch import Tensor
from chess_gnn.utils.constants import STARTING_BOARD


class ChessTokenizer(ABC):
    def __init__(self, board_str: Optional[str] = None):
        if board_str is None:
            board_str = STARTING_BOARD
        self.vocab = sorted(list(set(board_str.replace(" ", ""))))
        self.inverse_vocab = dict(zip(self.vocab, range(len(self.vocab))))
        self.vocab_size = len(self.vocab)

    @abstractmethod
    def tokenize(self, board_str: str) -> Tensor:
        raise NotImplementedError
