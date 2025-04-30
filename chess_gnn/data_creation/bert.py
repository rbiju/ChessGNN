import os
import random
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


class BERTLichessDataAggregator:
    def __init__(self, data_location: Path):
        self.data_location = data_location
        self.file_name = Path('aggregated_data.txt')
        self.label_folders = os.listdir(self.data_location)

        self.file_paths = []
        for dir_name in self.label_folders:
            full_dir = os.path.abspath(dir_name)
            for fname in os.listdir(full_dir):
                fpath = os.path.join(full_dir, fname)
                if os.path.isfile(fpath) and fname.endswith('.txt'):
                    self.file_paths.append((fpath, dir_name))

        random.shuffle(self.file_paths)

    def aggregate_data(self) -> None:
        with open(self.data_location / self.file_name, 'w', encoding='utf-8') as out_f:
            for fpath, label in self.file_paths:
                with open(fpath, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.rstrip('\n')
                        out_f.write(f"{line} {label}\n")
