from pathlib import Path
import uuid
from typing import Optional

import comet_ml
import torch
from pytorch_lightning import Trainer, seed_everything

from chess_gnn.data import BERTDataModule
from chess_gnn.trainer import TrainerFactory
from chess_gnn.models import ChessELECTRA, ChessBERT

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class ELECTRATrain(Task):
    def __init__(self, model: ChessELECTRA, datamodule: BERTDataModule, trainer_factory: TrainerFactory, bert_checkpoint: Optional[str] = None):
        super().__init__()
        seed_everything(42)
        torch.set_float32_matmul_precision('medium')

        if bert_checkpoint:
            bert = ChessBERT.load_from_checkpoint(checkpoint_path=bert_checkpoint)
            model = model(bert)

        self.model = model
        self.datamodule = datamodule

        self.uid = str(uuid.uuid4())

        ckpt_dir = Path('/home/ray/lightning_checkpoints/chess_electra') / self.uid
        trainer_factory.resolve_checkpoint_callback(ckpt_dir=str(ckpt_dir))
        trainer_factory.resolve_logger(project_name='chess-electra')

        self.trainer: Trainer = trainer_factory.trainer()

    def run(self, configuration_path: str):
        print(f"Saving engine in {self.uid}")
        artifact = comet_ml.Artifact(name="configuration", artifact_type="ConfigurationFile")
        artifact.add(configuration_path)
        experiment = self.trainer.logger.experiment
        experiment.log_artifact(artifact)

        model = torch.compile(self.model)

        self.trainer.fit(model=model, datamodule=self.datamodule)

    def debug_run(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == '__main__':
    config_path = get_config_path("ELECTRATrain")
    task = ELECTRATrain.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.debug_run()
