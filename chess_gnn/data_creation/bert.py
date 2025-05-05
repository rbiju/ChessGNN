from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import random

import multiprocessing
from threading import Thread
from queue import Queue
from tqdm import tqdm

import torch

from chess_gnn.utils import LichessChessBoardGetter
from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer

from .utils import Split


class BERTLichessDatasetCreator:
    def __init__(self, pgn_file: str,
                 data_directory: str = None,
                 split: Split = Split(),
                 seed: int = 42, num_workers: int = None):
        self.pgn_file = Path(pgn_file)
        if data_directory is None:
            data_directory = self.pgn_file.with_suffix('')
        self.data_directory = Path(data_directory)
        self.board_getter = LichessChessBoardGetter(self.pgn_file)
        self.random_seed = seed
        self.split = split
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

        os.makedirs(data_directory, exist_ok=True)
        for split in asdict(self.split).keys():
            for state in self.board_getter.result_mapping.values():
                os.makedirs(self.data_directory / split / str(state), exist_ok=True)

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

    def _choose_split(self, offset: int) -> str:
        rng = random.Random(f"{self.random_seed}_{offset}")
        val = rng.random()
        if val < self.split.train:
            return 'train'
        elif val < self.split.train + self.split.val:
            return 'val'
        else:
            return 'test'

    def _write_game(self, result: int, identifier: str, positions: list[str], split: str):
        file_path = self.data_directory / split / str(result) / f"{identifier}.txt"
        if file_path.exists():
            return
        with open(file_path, "w", encoding="utf-8") as f:
            for board in positions:
                f.write(board + "\n")

    def _process_offset(self, offset: int):
        parsed = self.board_getter.process_game_at_offset(offset)
        split = self._choose_split(offset)
        if parsed:
            self._write_game(*parsed, split=split)

    def create_dataset(self):
        offsets = self._get_game_offsets()
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self._process_offset, offsets), total=len(offsets),
                          desc="Processing games"):
                pass

        return [self.data_directory / split for split in asdict(self.split).keys()]


class BERTLichessDataAggregator:
    def __init__(self, data_location: str, include_draw: bool = False, max_open_files: int = 256):
        self.data_location = Path(data_location)
        self.include_draw = include_draw
        self.max_open_files = max_open_files  # Limit on number of open files at once

        if include_draw:
            self.output_path = self.data_location / 'aggregated_data_with_draws.txt'
        else:
            self.output_path = self.data_location / 'aggregated_data.txt'
        self.file_paths = self.get_file_paths()

        self.write_queue = Queue()
        self.sentinel = None
        self.writer_thread = Thread(target=self._writer)
        self.reader_threads = []

    def get_file_paths(self) -> list[tuple[Path, str]]:
        if not self.include_draw:
            label_folders = [d for d in self.data_location.iterdir() if d.is_dir()
                             and not d.name.startswith('.')
                             and not d.name.endswith('2')]
        else:
            label_folders = [d for d in self.data_location.iterdir() if d.is_dir()
                             and not d.name.startswith('.')]

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

    def _reader(self, file_path: Path, label: str, pbar):
        """Read a file line-by-line and enqueue labeled lines."""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    self.write_queue.put(f'{clean_line}{label}\n')
                pbar.update(1)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

    def aggregate(self):
        files = self.file_paths

        # Start writer
        self.writer_thread.start()

        futures = []
        # Use a ThreadPoolExecutor to limit the number of reader threads
        with tqdm(total=len(files), desc="Processing Files") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_open_files) as executor:
                for file_path, label in files:
                    executor.submit(self._reader, file_path, label, pbar)

        for future in futures:
            future.result()

        self.write_queue.put(self.sentinel)

        # Wait for readers to finish
        self.writer_thread.join()

        return self.output_path


class BERTLichessDataShuffler:
    def __init__(self, input_file: str, buffer_size=100000):
        self.input_file = Path(input_file)
        self.output_file = self.input_file.parent / 'shuffled.txt'

        self.buffer_size = buffer_size

    def shuffle(self):
        buffer = []
        with open(self.input_file, 'r', encoding='utf-8') as infile, \
                open(self.output_file, 'w', encoding='utf-8') as outfile:

            for line in infile:
                if len(buffer) < self.buffer_size:
                    buffer.append(line)
                else:
                    idx = random.randint(0, len(buffer))
                    if idx < self.buffer_size:
                        outfile.write(buffer[idx])
                        buffer[idx] = line
                    else:
                        outfile.write(line)

            # Write the remaining buffer shuffled
            random.shuffle(buffer)
            outfile.writelines(buffer)
