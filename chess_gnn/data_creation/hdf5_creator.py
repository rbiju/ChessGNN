from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer


class HDF5DatasetBuilder:
    def __init__(self, chunk_size: int, tokenizer: ChessTokenizer = SimpleChessTokenizer(), max_len: int = 64, num_workers: int = 4):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_workers = num_workers

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return None
        board_str = line[:-1]
        label = float(line[-1])
        tokenized = self.tokenizer.tokenize(board_str)
        return np.array(tokenized, dtype=np.int64), label

    def write_dataset(self, input_file: Path):
        input_file = Path(input_file)
        output_path = input_file.parent / "data.h5"

        if output_path.exists():
            output_path.unlink()

        with open(input_file, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)

        with h5py.File(output_path, 'w') as h5f:
            board_ds = h5f.create_dataset(
                'board', shape=(total_samples, self.max_len), dtype='i8',
                chunks=(self.chunk_size, self.max_len)
            )
            label_ds = h5f.create_dataset(
                'label', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,)
            )

            # Process lines in parallel
            with open(input_file, 'r', encoding='utf-8') as f:
                buffer = []
                idx = 0
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for line in tqdm(f, total=total_samples, desc=f"Writing {output_path.parent.name}"):
                        buffer.append(line)
                        if len(buffer) >= self.chunk_size:
                            results = list(executor.map(self._process_line, buffer))
                            results = [r for r in results if r is not None]
                            if results:
                                boards, labels = zip(*results)
                                board_ds[idx:idx+len(boards)] = np.stack(boards)
                                label_ds[idx:idx+len(labels)] = np.array(labels, dtype=np.float32)
                                idx += len(boards)
                            buffer = []

                    # Final flush
                    if buffer:
                        results = list(executor.map(self._process_line, buffer))
                        results = [r for r in results if r is not None]
                        if results:
                            boards, labels = zip(*results)
                            board_ds[idx:idx+len(boards)] = np.stack(boards)
                            label_ds[idx:idx+len(labels)] = np.array(labels, dtype=np.float32)
                            idx += len(boards)

            h5f.attrs['total_length'] = idx
