from abc import ABC, abstractmethod
from typing import Optional, Generator
from pathlib import Path
from uuid import uuid4

import chess.pgn


class ChessBoardGetter(ABC):
    def __init__(self, file: Path):
        self.file = str(file)

    @abstractmethod
    def process_game_at_offset(self, offset: int) -> Optional[tuple[int, str, list[str]]]:
        raise NotImplementedError


class LichessChessBoardGetter(ChessBoardGetter):
    def __init__(self, pgn_file: Path):
        super().__init__(pgn_file)
        self.result_mapping = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}

    @staticmethod
    def process_board_string(board: str) -> str:
        return str(board).replace('\n', ' ').replace(" ", "")

    def process_game_at_offset(self, offset: int) -> Optional[tuple[int, str, list[str]]]:
        with open(self.file, encoding='utf-8') as f:
            f.seek(offset)
            game = chess.pgn.read_game(f)
            if game is None or 'Result' not in game.headers or 'Site' not in game.headers:
                return None

            identifier = str(uuid4())
            result = self.result_mapping.get(game.headers['Result'])
            if result is None:
                return None

            board = game.board()
            positions = []
            whose_move = False
            for move in game.mainline_moves():
                move_from = move.from_square
                move_to = move.to_square
                positions.append(f'{self.process_board_string(str(board))}/{move_from}/{move_to}/{int(whose_move)}')
                board.push(move)
                whose_move = not whose_move

            return result, identifier, positions


class TransformerChessBoardGetter(ChessBoardGetter):
    def __init__(self, pgn_file: Path):
        super().__init__(pgn_file)
        self.result_mapping = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}

    @staticmethod
    def process_board_string(board: str) -> str:
        return str(board).replace('\n', ' ').replace(" ", "")

    def process_game_at_offset(self, offset: int) -> Optional[tuple[int, str, list[str]]]:
        with open(self.file, encoding='utf-8') as f:
            f.seek(offset)
            game = chess.pgn.read_game(f)
            if game is None or 'Result' not in game.headers or 'Site' not in game.headers:
                return None

            identifier = str(uuid4())
            result = self.result_mapping.get(game.headers['Result'])
            if result is None:
                return None

            board = game.board()
            positions = []
            whose_move = False
            for move in game.mainline_moves():
                move_from = move.from_square
                move_to = move.to_square
                current_board = self.process_board_string(str(board))
                board.push(move)
                next_board = self.process_board_string(str(board))
                positions.append(f'{current_board}/{next_board}/{move_from}/{move_to}/{int(whose_move)}')
                whose_move = not whose_move

            return result, identifier, positions


class PGNBoardHelper:
    def __init__(self, file: Path):
        self.file = file
        self.pgn = open(self.file, encoding="utf-8")
        self.game: Optional[chess.pgn.Game] = None
        self.eof = False

        self.result_mapping = {'1-0': 0,
                               '0-1': 1,
                               '1/2-1/2': 2}

        self.game_states = list(self.result_mapping.values())

    def get_game(self):
        self.game = chess.pgn.read_game(self.pgn)
        if self.game is None:
            self.eof = True

    @staticmethod
    def process_board_string(board: str) -> str:
        return str(board).replace('\n', ' ').replace(" ", "")

    def get_board_fens(self) -> list[str]:
        board = self.game.board()
        boards = []
        for move in self.game.mainline_moves():
            boards.append(board.fen())
            board.push(move)

        return boards

    def result(self) -> int:
        return self.result_mapping[self.game.headers['Result']]


def process_board_string(board: str) -> str:
    return str(board).replace('\n', ' ').replace(" ", "")
