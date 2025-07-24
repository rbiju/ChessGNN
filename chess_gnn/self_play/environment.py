from typing import Any, SupportsFloat

import chess
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ObsType

from chess_gnn.tokenizers import SimpleChessTokenizer
from .utils import ChessAction


class ChessEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleChessTokenizer()
        self.board = chess.Board()
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 13), dtype=np.float32)
        self.action_space = spaces.Discrete(4096)

    def _is_promotion(self, move):
        return (
                self.board.piece_type_at(move.from_square) == chess.PAWN and
                chess.square_rank(move.to_square) in [0, 7]
        )

    def get_obs(self):
        board = torch.tensor(self.tokenizer.tokenize(str(self.board)), dtype=torch.long)
        whose_move = torch.tensor([int(not self.board.turn)], dtype=torch.long)
        return board, whose_move

    def get_legal_moves_mask(self) -> torch.Tensor:
        legal_moves = [move for move in self.board.legal_moves]
        legal_moves = torch.tensor([ChessAction(move.from_square, move.to_square).to_1d() for move in legal_moves], dtype=torch.long)
        legal_move_mask = torch.zeros(4096, dtype=torch.bool)
        legal_move_mask[legal_moves] = True

        return legal_move_mask

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.board.reset()
        return self.get_obs(), {}

    def decode_action(self, action: torch.Tensor):
        pass

    def step(self, action: torch.Tensor,
             ) -> tuple[torch.Tensor, bool]:
        action = ChessAction.from_1d(int(action))
        move = chess.Move(from_square=action.from_square, to_square=action.to_square)

        if move is not None and self._is_promotion(move):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        self.board.push(move)

        done = self.board.is_game_over()
        reward = 0
        if done:
            result = self.board.result()
            reward = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.}[result]
            if not self.board.turn:
                reward = -reward

        return torch.tensor(reward), done
