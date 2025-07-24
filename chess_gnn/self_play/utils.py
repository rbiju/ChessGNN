import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from chess import Move


@dataclass
class ChessAction:
    from_square: int
    to_square: int

    def to_1d(self):
        return self.from_square * 64 + self.to_square

    @classmethod
    def from_1d(cls, idx: int) -> "ChessAction":
        return ChessAction(from_square=idx//64, to_square=idx % 64)


@dataclass
class PPOData:
    board: torch.Tensor
    whose_move: torch.Tensor
    reward: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor = None
    value: torch.Tensor = None
    mask: torch.Tensor = None

    def cat(self, other: "PPOData") -> "PPOData":
        return PPOData(board=torch.cat((self.board, other.board), dim=0),
                       whose_move=torch.cat((self.whose_move, other.whose_move), dim=0),
                       reward=torch.cat((self.reward, other.reward), dim=0),
                       action=torch.cat((self.action, other.action), dim=0),
                       log_prob=torch.cat((self.log_prob, other.log_prob), dim=0),
                       value=torch.cat((self.value, other.value), dim=0),
                       mask=torch.cat((self.mask, other.mask), dim=0)
                       )
