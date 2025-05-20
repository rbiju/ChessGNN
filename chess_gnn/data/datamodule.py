from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .h5_data import HDF5ChessDataset


class ChessDataModule(LightningDataModule):
    def __init__(self, data_directory: str, batch_size: int, mode: str, prefetch_factor: int = 4, num_workers: int = 12):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.mode = mode

        self.file_name = "data.h5"

    def train_dataloader(self):
        file = self.data_directory / 'train' / self.file_name
        dataset = HDF5ChessDataset(str(file), self.batch_size, mode=self.mode)

        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=True, persistent_workers=True, pin_memory=True, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        file = self.data_directory / 'val' / self.file_name
        dataset = HDF5ChessDataset(str(file), self.batch_size, mode=self.mode)

        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False, persistent_workers=True, pin_memory=True, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        file = self.data_directory / 'test' / self.file_name
        dataset = HDF5ChessDataset(str(file), self.batch_size, mode=self.mode)

        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
