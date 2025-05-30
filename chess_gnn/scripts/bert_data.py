import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader

from chess_gnn.configuration import LocalHydraConfiguration
from chess_gnn.data import HDF5ChessDataset
from chess_gnn.models import ChessBERT, ChessELECTRA, ChessXAttnEngine, ChessElectraEncoder, ChessContrastiveBackbone, ChessMLPEngine
from chess_gnn.models.chess_transformer import ChessTransformer


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


def transformer_dummy_forward():
    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/transformer.yaml')
    model = ChessTransformer.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'next_board': torch.randint(low=0, high=13, size=(4, 64)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model(batch)
    return out


def test_data():
    file = '/Users/ray/Datasets/chess/test_transformer/test/data.h5'
    dataset = HDF5ChessDataset(str(file), 4, mode='transformer')
    dl = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, persistent_workers=True, pin_memory=True)

    batch = next(iter(dl))

    return batch


def transformer_forward():
    file = '/Users/ray/Datasets/chess/test_transformer/test/data.h5'
    dataset = HDF5ChessDataset(str(file), 4, mode='transformer')
    dl = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, persistent_workers=True, pin_memory=True)

    batch = next(iter(dl))

    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/transformer.yaml')
    model = ChessTransformer.from_hydra_configuration(cfg)

    ms = ModelSummary(model=model)
    print(ms)

    out = model(batch)
    return out


def mlp_engine_dummy_forward():
    ckpt = torch.load('/Users/ray/models/chess/transformer/0eb35bb7-5c81-4f8f-829a-f00cd1686ac7/last.ckpt',
                      map_location="cpu")
    model = ChessTransformer(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'])

    encoder = model.get_encoder()

    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/mlp_engine.yaml')
    model = ChessMLPEngine.from_hydra_configuration(cfg)
    model = model(encoder)

    ms = ModelSummary(model=model)
    print(ms)

    batch = {'board': torch.randint(low=0, high=13, size=(4, 64)),
             'label': torch.randint(low=0, high=3, size=(4,)).float(),
             'from': torch.randint(low=0, high=64, size=(4,)),
             'to': torch.randint(low=0, high=64, size=(4,)),
             'whose_move': torch.randint(low=0, high=2, size=(4,))}

    out = model.calculate_loss(batch)

    return out


if __name__ == '__main__':
    mlp_engine_dummy_forward()
