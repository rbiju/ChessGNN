from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .bert_data import MMapWinPredictionBERTDataset, DataTransform, WinPredictionTransform


class BERTDataModule(LightningDataModule):
    def __init__(self, data_directory: str, batch_size: int, num_workers: int = 8,
                 transform: DataTransform = WinPredictionTransform()):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.file_name = "shuffled.txt"

    def train_dataloader(self):
        file = self.data_directory / 'train' / self.file_name
        dataset = MMapWinPredictionBERTDataset(file, transform=self.transform)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        file = self.data_directory / 'val' / self.file_name
        dataset = MMapWinPredictionBERTDataset(file, transform=self.transform)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        file = self.data_directory / 'test' / self.file_name
        dataset = MMapWinPredictionBERTDataset(file, transform=self.transform)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
