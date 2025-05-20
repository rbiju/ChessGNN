import multiprocessing
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer


class HDF5DatasetBuilder:
    def __init__(self, chunk_size: int, tokenizer: ChessTokenizer = SimpleChessTokenizer(), max_len: int = 64,
                 num_workers: int = None):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_workers = num_workers or multiprocessing.cpu_count()

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return None

        try:
            board_str, move_from, move_to, whose_move, label = line.split('/')
            tokenized = self.tokenizer.tokenize(board_str)
            return np.array(tokenized, dtype=np.int64), float(move_from), float(move_to), whose_move, float(label)
        except Exception:
            return None

    def write_dataset(self, input_file: str):
        input_file = Path(input_file)
        output_path = input_file.parent / "data.h5"

        if output_path.exists():
            output_path.unlink()

        with open(input_file, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)

        with h5py.File(output_path, 'w') as h5f:
            board_ds = h5f.create_dataset('board', shape=(total_samples, self.max_len), dtype='i8',
                                          chunks=(self.chunk_size, self.max_len))
            label_ds = h5f.create_dataset('label', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            from_ds = h5f.create_dataset('from', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            to_ds = h5f.create_dataset('to', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            whose_move_ds = h5f.create_dataset('whose_move', shape=(total_samples,), dtype='i8',
                                               chunks=(self.chunk_size,))

            idx = 0
            with open(input_file, 'r', encoding='utf-8') as f, Pool(self.num_workers) as pool:
                buffer = []
                for line in tqdm(f, total=total_samples, desc=f"Writing {output_path.parent.name}"):
                    buffer.append(line)
                    if len(buffer) >= self.chunk_size:
                        results = pool.map(self._process_line, buffer)
                        results = [r for r in results if r is not None]
                        if results:
                            boards, move_from, move_to, whose_move, labels = zip(*results)
                            length = len(boards)
                            board_ds[idx:idx + length] = np.stack(boards)
                            label_ds[idx:idx + length] = np.array(labels, dtype=np.float32)
                            from_ds[idx:idx + length] = np.array(move_from, dtype=np.float32)
                            to_ds[idx:idx + length] = np.array(move_to, dtype=np.float32)
                            whose_move_ds[idx:idx + length] = np.array(whose_move, dtype=np.int64)
                            idx += length
                        buffer = []

                if buffer:
                    results = pool.map(self._process_line, buffer)
                    results = [r for r in results if r is not None]
                    if results:
                        boards, move_from, move_to, whose_move, labels = zip(*results)
                        length = len(boards)
                        board_ds[idx:idx + length] = np.stack(boards)
                        label_ds[idx:idx + length] = np.array(labels, dtype=np.float32)
                        from_ds[idx:idx + length] = np.array(move_from, dtype=np.float32)
                        to_ds[idx:idx + length] = np.array(move_to, dtype=np.float32)
                        whose_move_ds[idx:idx + length] = np.array(whose_move, dtype=np.int64)
                        idx += length

            h5f.attrs['total_length'] = idx


class TransformerHDF5DatasetBuilder(HDF5DatasetBuilder):
    def __init__(self, chunk_size: int, tokenizer: ChessTokenizer = SimpleChessTokenizer(), max_len: int = 64,
                 num_workers: int = None):
        super().__init__(chunk_size, tokenizer, max_len, num_workers)

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return None

        try:
            board_str, next_board, move_from, move_to, whose_move, label = line.split('/')
            tokenized = self.tokenizer.tokenize(board_str)
            tokenized_next = self.tokenizer.tokenize(next_board)
            return np.array(tokenized, dtype=np.int64), np.array(tokenized_next, dtype=np.int64), float(move_from), float(move_to), whose_move, float(label)
        except Exception:
            return None

    def write_dataset(self, input_file: str):
        input_file = Path(input_file)
        output_path = input_file.parent / "data.h5"

        if output_path.exists():
            output_path.unlink()

        with open(input_file, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)

        with h5py.File(output_path, 'w') as h5f:
            board_ds = h5f.create_dataset('board', shape=(total_samples, self.max_len), dtype='i8',
                                          chunks=(self.chunk_size, self.max_len))
            next_board_ds = h5f.create_dataset('next_board', shape=(total_samples, self.max_len), dtype='i8',
                                               chunks=(self.chunk_size, self.max_len))
            label_ds = h5f.create_dataset('label', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            from_ds = h5f.create_dataset('from', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            to_ds = h5f.create_dataset('to', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,))
            whose_move_ds = h5f.create_dataset('whose_move', shape=(total_samples,), dtype='i8',
                                               chunks=(self.chunk_size,))

            idx = 0
            with open(input_file, 'r', encoding='utf-8') as f, Pool(self.num_workers) as pool:
                buffer = []
                for line in tqdm(f, total=total_samples, desc=f"Writing {output_path.parent.name}"):
                    buffer.append(line)
                    if len(buffer) >= self.chunk_size:
                        results = pool.map(self._process_line, buffer)
                        results = [r for r in results if r is not None]
                        if results:
                            boards, next_boards, move_from, move_to, whose_move, labels = zip(*results)
                            length = len(boards)
                            board_ds[idx:idx + length] = np.stack(boards)
                            next_board_ds[idx:idx + length] = np.stack(next_boards)
                            label_ds[idx:idx + length] = np.array(labels, dtype=np.float32)
                            from_ds[idx:idx + length] = np.array(move_from, dtype=np.float32)
                            to_ds[idx:idx + length] = np.array(move_to, dtype=np.float32)
                            whose_move_ds[idx:idx + length] = np.array(whose_move, dtype=np.int64)
                            idx += length
                        buffer = []

                if buffer:
                    results = pool.map(self._process_line, buffer)
                    results = [r for r in results if r is not None]
                    if results:
                        boards, next_boards, move_from, move_to, whose_move, labels = zip(*results)
                        length = len(boards)
                        board_ds[idx:idx + length] = np.stack(boards)
                        next_board_ds[idx:idx + length] = np.stack(next_boards)
                        label_ds[idx:idx + length] = np.array(labels, dtype=np.float32)
                        from_ds[idx:idx + length] = np.array(move_from, dtype=np.float32)
                        to_ds[idx:idx + length] = np.array(move_to, dtype=np.float32)
                        whose_move_ds[idx:idx + length] = np.array(whose_move, dtype=np.int64)
                        idx += length

            h5f.attrs['total_length'] = idx
