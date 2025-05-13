import h5py

import torch

DATA_MODES = {'ssl': ['board', 'whose_move'],
              'bert': ['board', 'whose_move', 'label'],
              'engine': ['board', 'label', 'from', 'to', 'whose_move'], }
DTYPES = {'board': torch.long,
          'whose_move': torch.long,
          'label': torch.float32,
          'from': torch.long,
          'to': torch.long,}


class HDF5ChessDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, batch_size: int, mode: str = 'bert'):
        self.path = path
        self.batch_size = batch_size
        self.h5_file = None
        self.keys = DATA_MODES[mode]

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

        data_dict = {key: torch.tensor(self.h5_file[key][start:end], dtype=DTYPES[key]) for key in self.keys}

        return data_dict
