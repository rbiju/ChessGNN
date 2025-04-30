import os
from pathlib import Path
import random

import multiprocessing
from threading import Thread
from queue import Queue

import chess.pgn

from chess_gnn.utils import LichessChessBoardGetter


class BERTLichessDatasetCreator:
    def __init__(self, pgn_file: Path, data_directory: Path, num_workers: int = None):
        self.pgn_file = pgn_file
        self.data_directory = data_directory
        self.board_getter = LichessChessBoardGetter(pgn_file)
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

        os.makedirs(data_directory, exist_ok=True)
        for state in self.board_getter.result_mapping.values():
            os.makedirs(data_directory / str(state), exist_ok=True)

    def _get_game_offsets(self) -> list[int]:
        offsets = []
        with open(self.pgn_file, encoding="utf-8") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("[Event "):  # Game start
                    offsets.append(offset)
                offset = f.tell()
        return offsets

    def _write_game(self, result: int, identifier: str, positions: list[str]):
        file_path = self.data_directory / str(result) / f"{identifier}.txt"
        if file_path.exists():
            return
        with open(file_path, "w", encoding="utf-8") as f:
            for board in positions:
                f.write(board + "\n")

    def _process_offset(self, offset: int):
        parsed = self.board_getter.process_game_at_offset(offset)
        if parsed:
            self._write_game(*parsed)

    def create_dataset(self):
        offsets = self._get_game_offsets()
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.map(self._process_offset, offsets)


class BERTLichessDataAggregator:
    def __init__(self, data_location: str):
        self.data_location = Path(data_location)
        self.output_path = self.data_location / 'aggregated_data.txt'
        self.file_paths = self.get_file_paths()

        self.write_queue = Queue()
        self.sentinel = None
        self.writer_thread = Thread(target=self._writer)
        self.reader_threads = []

    def get_file_paths(self) -> list[tuple[Path, int]]:
        label_folders = [d for d in self.data_location.iterdir() if d.is_dir() and not d.name.startswith('.')]

        file_paths = []
        for folder in label_folders:
            label = folder.name
            for file_path in folder.glob('*.txt'):
                if file_path.is_file():
                    file_paths.append((file_path, label))

        random.shuffle(file_paths)
        return file_paths

    def _writer(self):
        """Continuously write lines from the queue to the output file."""
        with self.output_path.open('w', encoding='utf-8') as f:
            while True:
                line = self.write_queue.get()
                if line is self.sentinel:
                    break
                f.write(line)
                self.write_queue.task_done()

    def _reader(self, file_path: Path, label: str):
        """Read a file line-by-line and enqueue labeled lines."""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    self.write_queue.put(f'{clean_line}{label}\n')
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

    def aggregate(self):
        """Run the full process: gather files, start threads, and combine them."""
        files = self.file_paths

        # Start writer
        self.writer_thread.start()

        # Launch readers
        for file_path, label in files:
            t = Thread(target=self._reader, args=(file_path, label))
            t.start()
            self.reader_threads.append(t)

        # Wait for readers
        for t in self.reader_threads:
            t.join()

        # Signal writer to finish
        self.write_queue.put(self.sentinel)
        self.writer_thread.join()
