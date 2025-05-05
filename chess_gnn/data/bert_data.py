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
    def __init__(self, h5_file_path: Path):
        # Open the HDF5 file for read-only access
        self.h5_file_path = h5_file_path
        self.h5f = h5py.File(h5_file_path, 'r')
        self.board_ds = self.h5f['board']
        self.label_ds = self.h5f['label']
        self.total_length = self.h5f.attrs['total_length']

    def __len__(self):
        # Return the total number of samples (length of the dataset)
        return self.total_length

    def __getitem__(self, idx):
        # Fetch the board and label data at the given index
        board = self.board_ds[idx]  # (max_len,)
        label = self.label_ds[idx]  # Scalar label

        # Convert to torch tensors (could be optimized depending on use)
        board_tensor = torch.tensor(board, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {'board': board_tensor, 'label': label_tensor}

    def __del__(self):
        # Ensure that the HDF5 file is properly closed when the object is deleted
        self.h5f.close()
