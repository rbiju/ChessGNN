from abc import ABC, abstractmethod
from pathlib import Path
import mmap
import h5py

import torch
from torch.utils.data import IterableDataset, get_worker_info

from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer


class DataTransform(ABC):
    def __init__(self, tokenizer: ChessTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, line) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class WinPredictionTransform(DataTransform):
    def __init__(self, tokenizer: SimpleChessTokenizer = SimpleChessTokenizer()):
        super().__init__(tokenizer)

    def __call__(self, line) -> dict[str, torch.Tensor]:
        return {'board': self.tokenizer.tokenize(line[:-1]),
                'label': torch.tensor(int(line[-1]), dtype=torch.float32)}


class WinPredictionMapDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: Path, transform: DataTransform = WinPredictionTransform()):
        self.file_path = file_path
        self.transform = transform

        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]  # in-memory load

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.transform(self.lines[idx])


class MMapWinPredictionBERTDataset(IterableDataset):
    def __init__(self, file_path: Path, transform: DataTransform = WinPredictionTransform()):
        self.file_path = file_path
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        with open(str(self.file_path), 'r', encoding='utf-8') as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            file_size = mmapped_file.size()
            num_workers = worker_info.num_workers if worker_info else 1
            worker_id = worker_info.id if worker_info else 0

            chunk_size = file_size // num_workers
            start = worker_id * chunk_size
            end = file_size if worker_id == num_workers - 1 else (worker_id + 1) * chunk_size

            # Skip to the beginning of the next full line
            if start != 0:
                mmapped_file.seek(start)
                mmapped_file.readline()  # discard partial line
                start = mmapped_file.tell()

            mmapped_file.seek(start)

            while mmapped_file.tell() < end:
                line = mmapped_file.readline()
                if not line:
                    break
                line = line.decode('utf-8').strip()
                if line:
                    yield self.transform(line) if self.transform else line

            mmapped_file.close()


class HDF5ChessDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, batch_size: int):
        self.path = path
        self.batch_size = batch_size
        self.h5_file = None

        with h5py.File(self.path, 'r') as f:
            self.total_samples = f.attrs['total_length']

        self.total_batches = (self.total_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.total_batches

    def _ensure_file_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.path, 'r')

    def __getitem__(self, idx):
        self._ensure_file_open()

        start = idx * self.batch_size
        end = min(start + self.batch_size, self.total_samples)

        boards = self.h5_file['board'][start:end]
        labels = self.h5_file['label'][start:end]

        return {
            'board': torch.tensor(boards, dtype=torch.long),
            'label': torch.tensor(labels, dtype=torch.float32)
        }
