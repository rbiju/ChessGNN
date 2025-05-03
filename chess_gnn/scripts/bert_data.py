from torch.utils.data import DataLoader
from chess_gnn.data.bert_data import MMapWinPredictionBERTDataset
from chess_gnn.data import BERTDataModule

from chess_gnn.models import ChessBERT
from chess_gnn.configuration import LocalHydraConfiguration

from pytorch_lightning.utilities.model_summary import ModelSummary


def test_data():
    file = '/Users/ray/Datasets/txt/test/train/shuffled.txt'
    dataset = MMapWinPredictionBERTDataset(file)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    batch = next(iter(dataloader))

    return batch


def model_test():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/bert.yaml')
    model = ChessBERT.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)


def model_forward_test():
    model = ChessBERT.from_hydra_configuration(LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/bert.yaml'))

    file = '/Users/ray/Datasets/txt/test/train/shuffled.txt'
    dataset = MMapWinPredictionBERTDataset(file)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    batch = next(iter(dataloader))

    ms = ModelSummary(model=model)
    print(ms)

    out = model(batch['board'])

    return out


def test_datamodule():
    dm = BERTDataModule(data_directory='/Users/ray/Datasets/txt/test', batch_size=4, num_workers=2)
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))

    return train_batch, val_batch


if __name__ == '__main__':
    test_datamodule()
