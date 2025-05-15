import os
from functools import partial
from typing import Optional, Union
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from chess_gnn.configuration import HydraConfigurable


@HydraConfigurable
class TrainerFactory:
    def __init__(self,
                 accelerator: str = 'gpu',
                 devices: int = -1,
                 precision: Union[str, int] = 32,
                 accumulate_grad_batches: int = 1,
                 strategy: Optional[str] = 'ddp',
                 max_epochs: int = 200,
                 log_every_n_steps: int = 250,
                 num_sanity_val_steps: int = 1,
                 check_val_every_n_epoch=None,
                 **kwargs):
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.strategy = strategy
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.num_sanity_val_steps = num_sanity_val_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.callbacks = kwargs.pop('callbacks', [])
        self.logger = kwargs.pop('logger', None)
        self.kwargs = kwargs

    @property
    def limit_val_batches(self):
        return self.kwargs.get('limit_val_batches', None)

    def resolve_checkpoint_callback(self, ckpt_dir: str):
        for i, callback in enumerate(self.callbacks):
            if isinstance(callback, partial):
                if callback.func == pl.callbacks.model_checkpoint.ModelCheckpoint:
                    self.callbacks[i] = callback(ckpt_dir)

    def resolve_logger(self, project_name: Optional[str] = None):
        if self.logger is None:
            warnings.warn('Logger was not set, setting comet logger automatically')
            if project_name is None:
                project_name = 'chess_bert'
            self.logger = CometLogger(api_key=os.getenv('COMET_API_KEY'), project_name=project_name)

    def trainer(self) -> pl.Trainer:
        return pl.Trainer(accelerator=self.accelerator,
                          callbacks=self.callbacks,
                          logger=self.logger,
                          devices=self.devices,
                          precision=self.precision,
                          accumulate_grad_batches=self.accumulate_grad_batches,
                          strategy=self.strategy,
                          max_epochs=self.max_epochs,
                          log_every_n_steps=self.log_every_n_steps,
                          num_sanity_val_steps=self.num_sanity_val_steps,
                          check_val_every_n_epoch=self.check_val_every_n_epoch,
                          **self.kwargs)
