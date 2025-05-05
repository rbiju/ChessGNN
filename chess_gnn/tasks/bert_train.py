import os

import comet_ml
import torch
from pytorch_lightning import Trainer, seed_everything

from chess_gnn.models import ChessBERT
from chess_gnn.data import BERTDataModule

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path
from chess_gnn.callbacks import CometLoggerCallbackFactory


@HydraConfigurable
class BERTTrain(Task):
    def __init__(self, model: ChessBERT, datamodule: BERTDataModule, trainer: Trainer, logger: CometLoggerCallbackFactory):
        super().__init__()
        seed_everything(42)
        torch.set_float32_matmul_precision('medium')

        self.model = torch.compile(model)
        self.datamodule = datamodule
        self.trainer = trainer
        self.trainer.logger = logger.logger(os.getenv("COMET_API_KEY"))

    def run(self, configuration_path: str):
        artifact = comet_ml.Artifact(name="configuration", artifact_type="ConfigurationFile")
        artifact.add(configuration_path)
        experiment = self.trainer.logger.experiment
        experiment.log_artifact(artifact)

        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == '__main__':
    config_path = get_config_path(BERTTrain)
    task = BERTTrain.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
