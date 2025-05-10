from pytorch_lightning import Trainer
import torch

from chess_gnn.configuration import LocalHydraConfiguration
from chess_gnn.models import ChessBERT


def save_model_with_hparams():
    config = LocalHydraConfiguration('/home/ray/pycharm_projects/ChessGNN/configs/bert/training/bert.yaml')
    model: ChessBERT = ChessBERT.from_hydra_configuration(config)
    checkpoint = torch.load('/home/ray/lightning_checkpoints/chess_bert/5b08cbbb-8893-4d6c-865e-a9a08771c824/last.ckpt', weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer = Trainer()
    trainer.strategy.connect(model)
    trainer.save_checkpoint("/home/ray/lightning_checkpoints/chess_bert/5b08cbbb-8893-4d6c-865e-a9a08771c824/final.ckpt")


if __name__ == '__main__':
    save_model_with_hparams()
