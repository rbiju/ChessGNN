from abc import ABC, abstractmethod
from typing import Generator
from pathlib import Path

import chess.pgn
from chess import Board

SITE_PREFIXES = {"lichess": 'https://lichess.org/'}


class ChessBoardGetter(ABC):
    def __init__(self, file: Path):
        self.file = str(file)

    @abstractmethod
    def get_board_strings(self) -> Generator[str, None, None]:
        raise NotImplementedError

    @abstractmethod
    def get_unique_game_identifier(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> int:
        raise NotImplementedError


class LichessChessBoardGetter(ChessBoardGetter):
    def __init__(self, file: Path, mode: str = 'lichess'):
        super().__init__(file)
        if mode not in SITE_PREFIXES.keys():
            raise ValueError(f'Invalid mode: {mode}')

        self.prefix = SITE_PREFIXES[mode]
        self.pgn = open(self.file, encoding="utf-8")
        self.game = None
        self.eof = False

        self.result_mapping = {'1-0': 0,
                               '0-1': 1,
                               '1/2-1/2': 2}

        self.game_states = list(self.result_mapping.values())

    def get_game(self):
        self.game = chess.pgn.read_game(self.pgn)
        if self.game is None:
            self.eof = True

    def get_unique_game_identifier(self) -> str:
        site: str = self.game.headers["Site"]
        return site.removeprefix(self.prefix)

    @staticmethod
    def process_board_string(board: str) -> str:
        return str(board).replace('\n', ' ').replace(" ", "")

    def get_board_strings(self) -> Generator[str, None, None]:
        board = self.game.board()
        for move in self.game.mainline_moves():
            board.push(move)
            yield self.process_board_string(str(board))

    def result(self) -> int:
        return self.result_mapping[self.game.headers['Result']]
