from typing import Optional

import chess
import torch

from chess_gnn.utils import process_board_string
from .base import ChessTokenizer


class SimpleChessTokenizer(ChessTokenizer):
    def __init__(self, board_str: Optional[str] = None):
        super().__init__(board_str)

    def tokenize(self, board_str: str) -> list[float]:
        return [self.inverse_vocab[token] for token in board_str]

    def untokenize(self, tokens: list[int]) -> list[str]:
        board_str = [self.vocab[token] for token in tokens]
        return board_str

    def tokenize_board(self, chess_board: chess.Board):
        board = process_board_string(str(chess_board))
        board_tokens = torch.Tensor(self.tokenize(board)).long()
        whose_move = torch.Tensor([int(not chess_board.turn)]).long()

        return board_tokens, whose_move
