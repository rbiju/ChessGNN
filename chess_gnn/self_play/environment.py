from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType

import numpy as np
import torch
import chess

from chess_gnn.tokenizers import SimpleChessTokenizer
from .utils import ChessEngineMoveHandler


class ChessEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleChessTokenizer()
        self.board = chess.Board()
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 13), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)

    def _is_promotion(self, move):
        return (
                self.board.piece_type_at(move.from_square) == chess.PAWN and
                chess.square_rank(move.to_square) in [0, 7]
        )

    def _get_obs(self):
        return torch.tensor(self.tokenizer.tokenize(str(self.board)), dtype=torch.long)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.board.reset()
        return self._get_obs(), {}

    def step(self, action: dict[str, torch.Tensor],
             ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        move_selector = ChessEngineMoveHandler(action)
        move = move_selector.select_move(list(self.board.legal_moves))

        if move is not None and self._is_promotion(move):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        self.board.push(move)

        done = self.board.is_game_over()
        reward = 0
        if done:
            result = self.board.result()
            reward = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]

        return self._get_obs(), reward, done, False, {}
