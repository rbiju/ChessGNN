import os
from pathlib import Path

from chess_gnn.utils import LichessChessBoardGetter


class BERTLichessDatasetCreator:
    def __init__(self, pgn_file: Path, data_directory: Path):
        self.pgn_file = pgn_file
        self.data_directory = data_directory
        self.board_getter = LichessChessBoardGetter(file=pgn_file)

        os.makedirs(data_directory, exist_ok=True)
        for game_state in self.board_getter.game_states:
            os.makedirs(data_directory / str(game_state), exist_ok=True)

    def create_dataset(self):
        while True:
            self.board_getter.get_game()
            if self.board_getter.eof:
                break
            file_dir = Path(self.data_directory) / Path(str(int(self.board_getter.result())))
            file_path = file_dir / Path(self.board_getter.get_unique_game_identifier()).with_suffix('.txt')
            if file_path.exists():
                continue

            with open(file_path, "w", encoding="utf-8") as f:
                for text in self.board_getter.get_board_strings():
                    f.write(text + "\n")
