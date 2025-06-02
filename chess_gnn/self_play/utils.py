import numpy as np
import torch
import torch.nn.functional as F
from chess import Move


class ChessEngineMoveHandler:
    def __init__(self, predictions: dict[str, torch.Tensor]):
        self.from_logits = F.softmax(predictions['from'], dim=-1)
        self.to_logits = F.softmax(predictions['to'], dim=-1)

    def select_move(self, legal_moves: list[Move]) -> Move:
        move_weights = torch.outer(self.from_logits, self.to_logits).flipud()

        legal_move_weights = []
        for move in legal_moves:
            legal_move_weights.append(move_weights[move.from_square][move.to_square])

        legal_move_weights = np.array(legal_move_weights)
        move = legal_moves[np.random.choice(np.arange(len(legal_move_weights)), p=legal_move_weights)]

        return move
