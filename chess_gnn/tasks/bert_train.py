import torch
from pytorch_lightning import Trainer, seed_everything

from chess_gnn.models import ChessBERT
from chess_gnn.data import BERTDataModule

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class BERTTrain(Task):
    def __init__(self, model: ChessBERT, datamodule: BERTDataModule, trainer: Trainer):
        super().__init__()
        seed_everything(42)
        torch.set_float32_matmul_precision('medium')

        self.model = torch.compile(model)
        self.datamodule = datamodule
        self.trainer = trainer

    def run(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == '__main__':
    config_path = get_config_path(BERTTrain)
    task = BERTTrain.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
