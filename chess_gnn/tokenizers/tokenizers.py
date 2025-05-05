from typing import Optional

from .base import ChessTokenizer


class SimpleChessTokenizer(ChessTokenizer):
    def __init__(self, board_str: Optional[str] = None):
        super().__init__(board_str)

    def tokenize(self, board_str: str) -> list[float]:
        return [self.inverse_vocab[token] for token in board_str]
