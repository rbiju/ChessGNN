from chess_gnn.models import ChessBERT, ChessELECTRA, ChessXAttnEngine, ChessElectraEncoder, ChessContrastiveBackbone
from chess_gnn.configuration import LocalHydraConfiguration
from chess_gnn.data import HDF5ChessDataset

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_summary import ModelSummary


def model_test():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/bert.yaml')
    model = ChessBERT.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)


def model_dummy_forward():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/bert.yaml')
    model = ChessBERT.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'label': torch.randint(low=0, high=2, size=(4,)).float(),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model.calculate_loss(batch)

    return out


def electra_dummy_forward():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/electra.yaml')
    model = ChessELECTRA.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model(batch)

    return out


def engine_dummy_forward():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/engine.yaml')
    model = ChessXAttnEngine.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'label': torch.randint(low=0, high=2, size=(4,)).float(),
             'from': torch.randint(low=0, high=64, size=(4,)),
             'to': torch.randint(low=0, high=64, size=(4,)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model.calculate_loss(batch)

    return out


def test_engine_data():
    file = '/home/ray/datasets/chess/Carlsen/test/data.h5'
    dataset = HDF5ChessDataset(str(file), 4, mode='engine')

    dl = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    batch = next(iter(dl))

    cfg = LocalHydraConfiguration('/home/ray/pycharm_projects/ChessGNN/configs/bert/training/engine.yaml')
    model = ChessXAttnEngine.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    out = model.calculate_loss(batch)

    return out


def test_electra_encoder():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/electra.yaml')
    model = ChessELECTRA.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    encoder = ChessElectraEncoder(electra=model)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = encoder(batch['board'], batch['whose_move'], get_attn=True)

    return out


def contrastive_dummy_forward():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/contrastive.yaml')
    model = ChessContrastiveBackbone.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model.training_step(batch, 0)

    return out


if __name__ == '__main__':
    contrastive_dummy_forward()
