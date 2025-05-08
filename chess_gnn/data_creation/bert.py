from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import os
from pathlib import Path
import random
import shutil
from uuid import uuid4

import multiprocessing

from tqdm import tqdm

from chess_gnn.utils import LichessChessBoardGetter

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
        print("Getting game offsets...")
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

        print("Game offsets retrieved.")
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
    def __init__(self, data_location: str, batch_size: int = 750, include_draw: bool = False):
        self.data_location = Path(data_location)
        self.include_draw = include_draw
        self.batch_size = batch_size

        self.output_path = self.data_location / (
            'aggregated_data_with_draws.txt' if include_draw else 'aggregated_data.txt'
        )
        self.temp_dir = self.data_location / ".temp_agg_parts"
        self.temp_dir.mkdir(exist_ok=True)

        self.file_paths = self.get_file_paths()

    def get_file_paths(self) -> list[tuple[Path, str]]:
        if not self.include_draw:
            label_folders = [
                d for d in self.data_location.iterdir()
                if d.is_dir() and not d.name.startswith('.') and not d.name.endswith('2')
            ]
        else:
            label_folders = [
                d for d in self.data_location.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]

        file_paths = []
        for folder in label_folders:
            label = folder.name
            for file_path in folder.glob('*.txt'):
                if file_path.is_file():
                    file_paths.append((file_path, label))

        random.shuffle(file_paths)
        return file_paths

    @staticmethod
    def process_file_batch(batch: list[tuple[Path, str]], temp_dir: Path) -> str:
        out_path = temp_dir / f"{uuid4().hex}.txt"
        buffer = []

        for file_path, label in batch:
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    buffer.extend(f"{line.rstrip()}/{label}\n" for line in f)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

        with out_path.open('w', encoding='utf-8') as fout:
            fout.writelines(buffer)

        return str(out_path)

    @staticmethod
    def chunked(iterable, n):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def aggregate(self):
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        batches = list(self.chunked(self.file_paths, self.batch_size))

        print(f"Aggregating {len(self.file_paths)} files in {len(batches)} batches using {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            process_fn = partial(self.process_file_batch, temp_dir=self.temp_dir)
            results = executor.map(process_fn, batches, chunksize=1)
            part_files = list(tqdm(results, total=len(batches), desc="Processing Batches"))

        with self.output_path.open('w', encoding='utf-8') as fout:
            for part_file in part_files:
                with open(part_file, 'r', encoding='utf-8') as pf:
                    shutil.copyfileobj(pf, fout)

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"Aggregation complete: {self.output_path}")

        return self.output_path


class BERTLichessDataShuffler:
    def __init__(self, input_file: str, buffer_size: int = 2500):
        self.input_file = Path(input_file)
        self.output_file = self.input_file.parent / 'shuffled.txt'
        self.buffer_size = buffer_size

    def shuffle(self):
        buffer = []

        with open(self.input_file, 'r', encoding='utf-8') as infile, \
             open(self.output_file, 'w', encoding='utf-8') as outfile:

            total_lines = sum(1 for _ in open(self.input_file, 'r', encoding='utf-8'))
            infile.seek(0)  # Reset after counting

            with tqdm(total=total_lines, desc="Windowed shuffling", unit="lines") as pbar:
                for line in infile:
                    buffer.append(line)
                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        outfile.writelines(buffer)
                        buffer.clear()
                        pbar.update(self.buffer_size)

                # Write remaining lines
                if buffer:
                    random.shuffle(buffer)
                    outfile.writelines(buffer)
                    pbar.update(len(buffer))
