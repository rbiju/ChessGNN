from pathlib import Path
import uuid

import comet_ml
import pytorch_lightning
import torch
from pytorch_lightning import Trainer, seed_everything

from chess_gnn.data import ChessDataModule
from chess_gnn.trainer import TrainerFactory

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class BERTTrain(Task):
    def __init__(self, model: pytorch_lightning.LightningModule, datamodule: ChessDataModule, trainer_factory: TrainerFactory):
        super().__init__()
        seed_everything(42)
        torch.set_float32_matmul_precision('medium')

        self.model = model
        self.datamodule = datamodule

        self.uid = str(uuid.uuid4())

        ckpt_dir = Path('/home/ray/lightning_checkpoints/chess_bert') / self.uid
        trainer_factory.resolve_checkpoint_callback(ckpt_dir=str(ckpt_dir))
        trainer_factory.resolve_logger()

        self.trainer: Trainer = trainer_factory.trainer()

    def run(self, configuration_path: str):
        print(f"Saving model in {self.uid}")
        artifact = comet_ml.Artifact(name="configuration", artifact_type="ConfigurationFile")
        artifact.add(configuration_path)
        experiment = self.trainer.logger.experiment
        experiment.log_artifact(artifact)
        experiment.set_name(self.uid)

        model = torch.compile(self.model)

        self.trainer.fit(model=model, datamodule=self.datamodule)

    def debug_run(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == '__main__':
    config_path = get_config_path("BERTTrain")
    task = BERTTrain.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.debug_run()
