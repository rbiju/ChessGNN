from typing import NamedTuple
from einops import repeat
import pytorch_lightning as pl

import torch
from torch.nn import Parameter
import torch.nn as nn

from typing import Iterable

from chess_gnn.bert import TransformerBlock, BERTMaskHandler
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.optimizers.lr_schedules import LRSchedulerFactory
from chess_gnn.tokenizers import ChessTokenizer


class BERTLossWeights(NamedTuple):
    masking: float = 1.0
    win_prediction: float = 1.0


@HydraConfigurable
class ChessBERT(pl.LightningModule):
    def __init__(self, num_layers: int,
                 block: TransformerBlock,
                 masking_loss: nn.Module,
                 win_prediction_loss: nn.Module,
                 tokenizer: ChessTokenizer,
                 mask_handler: BERTMaskHandler,
                 loss_weights: BERTLossWeights = BERTLossWeights()):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = nn.ModuleList([block for _ in range(num_layers)])
        self.masking_loss = masking_loss
        self.win_prediction_loss = win_prediction_loss
        self.mask_handler = mask_handler

        self.loss_weights = loss_weights
        self.vocab_size = tokenizer.vocab_size

        self.mask_token = torch.zeros(block.dim)
        self.cls_token = torch.zeros(1, block.dim)

        self.embedding_table = torch.nn.Parameter(torch.rand(tokenizer.vocab_size, 32))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        torch.nn.init.trunc_normal_(self.embedding_table, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x, ids_mask = self.mask_handler.mask_sequence(x,
                                                      vocab_size=self.vocab_size,
                                                      mask_token=self.mask_token)

        cls_token = repeat(self.cls_token, 'l e -> b l e', b=x.shape[0])
        x_ = torch.concat([cls_token, x], dim=1)

        x_ = self.encoder(x_)

        return {'cls': x_[..., :1, ...],
                'tokens': x_[..., :1, ...],
                'ids_mask': ids_mask}

    def training_step(self, batch, batch_idx):
        out = self(batch['board'])

        masked = self.mask_handler.mask_loss(out['tokens'], batch['tokens'], ids_mask=batch['ids_mask'])
        mask_loss = self.mask_loss(masked)

        win_prediction_loss = self.win_prediction_loss(out['cls'], batch['labels'])

        loss = (self.loss_weights.masking * mask_loss +
                self.loss_weights.win_prediction + win_prediction_loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch['board'])

        masked = self.mask_handler.mask_loss(out['tokens'], batch['tokens'], ids_mask=batch['ids_mask'])
        mask_loss = self.mask_loss(masked)

        win_prediction_loss = self.win_prediction_loss(out['cls'], batch['labels'])

        loss = (self.loss_weights.masking * mask_loss +
                self.loss_weights.win_prediction + win_prediction_loss)

        return {'loss': loss}

    @staticmethod
    def configure_optimizer_from_params(params: Iterable[tuple[str, Parameter]],
                                        optimizer_factory: OptimizerFactory,
                                        scheduler_factory: LRSchedulerFactory):
        if optimizer_factory is None or scheduler_factory is None:
            raise RuntimeError('Optimizer and scheduler must be set for nagini training')

        optimizer = optimizer_factory.optimizer(params=params)
        scheduler = scheduler_factory.scheduler(optimizer=optimizer)

        optimizer_config = {"optimizer": optimizer}
        optimizer_config.update(scheduler_factory.scheduler_config(scheduler=scheduler))

        return optimizer_config

    def configure_optimizers(self):
        return self.configure_optimizer_from_params(self.named_parameters(),
                                                    self.optimizer_factory,
                                                    self.scheduler_factory)
