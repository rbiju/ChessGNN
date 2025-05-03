from abc import ABC, abstractmethod
from pathlib import Path
import mmap

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
        board_str = line[:-1]
        label = torch.tensor(int(line[-1]), dtype=torch.float32)

        return {'board': self.tokenizer.tokenize(board_str),
                'label': label}


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
