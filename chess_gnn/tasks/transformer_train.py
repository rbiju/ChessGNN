from pathlib import Path
from functools import partial
import uuid
from typing import Optional, Union

import comet_ml
import torch
from pytorch_lightning import Trainer, seed_everything

from chess_gnn.data import ChessDataModule
from chess_gnn.trainer import TrainerFactory
from chess_gnn.models import ChessTransformer, ChessELECTRA

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class TransformerTrain(Task):
    def __init__(self, model: Union[ChessTransformer, partial], datamodule: ChessDataModule, trainer_factory: TrainerFactory, electra_checkpoint: Optional[str] = None,
                 ckpt_path: Optional[str] = None, compile_model: bool = True):
        super().__init__()
        seed_everything(42)
        torch.set_float32_matmul_precision('medium')
        self.ckpt_path = ckpt_path
        self.datamodule = datamodule
        self.compile_model = compile_model

        if self.ckpt_path is None:
            if electra_checkpoint:
                electra: ChessELECTRA = ChessELECTRA.load_from_checkpoint(checkpoint_path=electra_checkpoint)
                discriminator = electra.discriminator
                model = model(discriminator)

            self.model = model
            self.uid = str(uuid.uuid4())

            ckpt_dir = Path('/home/ray/lightning_checkpoints/chess_transformer') / self.uid
            trainer_factory.resolve_checkpoint_callback(ckpt_dir=str(ckpt_dir))
            trainer_factory.resolve_logger(project_name='chess-transformer')

            self.trainer: Trainer = trainer_factory.trainer()
        else:
            self.ckpt_path = Path(self.ckpt_path)
            self.uid = str(self.ckpt_path.parent.name)

            self.model = ChessTransformer.load_from_checkpoint(checkpoint_path=self.ckpt_path)

            ckpt_dir = Path('/home/ray/lightning_checkpoints/chess_transformer') / self.uid
            trainer_factory.resolve_checkpoint_callback(ckpt_dir=str(ckpt_dir))
            trainer_factory.resolve_logger(project_name='chess-transformer')

            self.trainer: Trainer = trainer_factory.trainer()

    def run(self, configuration_path: str):
        print(f"Saving backbone in {self.uid}")
        artifact = comet_ml.Artifact(name="configuration", artifact_type="ConfigurationFile")
        artifact.add(configuration_path)
        experiment: comet_ml.CometExperiment = self.trainer.logger.experiment
        experiment.log_artifact(artifact)
        experiment.set_name(self.uid)

        if self.ckpt_path is None:
            if self.compile_model:
                self.model = torch.compile(self.model)
            self.trainer.fit(model=self.model, datamodule=self.datamodule)
        else:
            if self.compile_model:
                self.model = torch.compile(self.model)
            self.trainer.fit(model=self.model, datamodule=self.datamodule, ckpt_path=self.ckpt_path)

    def debug_run(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == '__main__':
    config_path = get_config_path("TransformerTrain")
    task = TransformerTrain.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.debug_run()
