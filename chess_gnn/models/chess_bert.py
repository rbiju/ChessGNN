import copy
from dataclasses import dataclass

import einops
from einops import repeat
import pytorch_lightning as pl

import torch
from torch.nn import Parameter
import torch.nn as nn
from torchmetrics import Accuracy

from typing import Iterable

from chess_gnn.bert import TransformerBlock, BERTMaskHandler, Mlp
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.lr_schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.tokenizers import ChessTokenizer


@dataclass
class BERTLossWeights:
    masking: float = 1.0
    win_prediction: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.masking < 0 or self.win_prediction < 0:
            raise ValueError(f"Loss proportions must be positive: {self.masking, self.win_prediction}")


@HydraConfigurable
class ChessBERT(pl.LightningModule):
    def __init__(self, num_layers: int,
                 block: TransformerBlock,
                 tokenizer: ChessTokenizer,
                 mask_handler: BERTMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 win_prediction_dropout: float = 0.,
                 loss_weights: BERTLossWeights = BERTLossWeights()):
        super().__init__()
        self.has_learned_pos_emb = False
        self.pos_emb = None
        self.dim = block.dim
        self.num_layers = num_layers
        self.encoder = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        self.mask_handler = mask_handler

        self.loss_weights = loss_weights
        self.vocab_size = tokenizer.vocab_size

        self.cls_token = torch.nn.Parameter(torch.rand(1, block.dim))

        self.whose_move_embedding = nn.Parameter(torch.rand(2, block.dim))

        self.embedding_table = torch.nn.Parameter(torch.rand(tokenizer.vocab_size + 1, block.dim))

        self.mlm_head = nn.Linear(block.dim, self.vocab_size)
        self.win_prediction_head = Mlp(in_dim=block.dim, out_dim=1, dropout=win_prediction_dropout, hidden_dim=block.dim)
        self.win_prediction_accuracy = Accuracy(task='binary', threshold=0.5)

        if block.pos_emb_mode == 'learned':
            self.has_learned_pos_emb = True
            self.pos_emb = torch.nn.Parameter(torch.rand(65, block.dim))
            torch.nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(block.dim)

        self.masking_loss = nn.CrossEntropyLoss()
        self.win_prediction_loss = nn.BCEWithLogitsLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.initialize_weights()
        self.save_hyperparameters(ignore=['block', 'mask_handler'])

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.embedding_table, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        x_ = self.embedding_table[x]
        cls_token = repeat(self.cls_token, 'l e -> b l e', b=x.shape[0])
        x_ = torch.concat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move_embedding[whose_move].unsqueeze(1)

        if self.has_learned_pos_emb:
            x_ = x_ + self.pos_emb.unsqueeze(0)

        for block in self.encoder:
            x_ = block(x_)

        x_ = self.norm(x_)

        cls = x_[:, :1, :].squeeze()
        win_prob = self.win_prediction_head(cls).squeeze()

        return {'cls': cls,
                'tokens': x_[:, 1:, :],
                'win_probability': win_prob}

    def forward_mask(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        x, labels = self.mask_handler(x)

        out = self(x, whose_move)

        return {**out, 'masked_token_labels': labels}

    def calculate_loss(self, batch):
        batch = self.squeeze_batch(batch)
        out = self.forward_mask(batch['board'], batch['whose_move'])

        mlm_preds = self.mlm_head(out['tokens'])
        mlm_preds = einops.rearrange(mlm_preds, 'b l c -> b c l')

        mask_loss = self.masking_loss(mlm_preds, out['masked_token_labels'])

        win_prediction_loss = self.win_prediction_loss(out['win_probability'], batch['label'])
        self.win_prediction_accuracy.update(out['win_probability'], batch['label'])

        loss = (self.loss_weights.masking * mask_loss +
                self.loss_weights.win_prediction * win_prediction_loss)

        return {'loss': loss, 'mask_loss': mask_loss, 'win_prediction_loss': win_prediction_loss}

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("train_masking_loss", loss['mask_loss'], on_step=True, sync_dist=True)
        self.log("train_win_prediction_loss", loss['win_prediction_loss'], on_step=True, sync_dist=True)
        self.log("train_win_prediction_accuracy", self.win_prediction_accuracy, on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("val_masking_loss", loss['mask_loss'], sync_dist=True)
        self.log("val_win_prediction_loss", loss['win_prediction_loss'], sync_dist=True)
        self.log("val_win_prediction_accuracy", self.win_prediction_accuracy, sync_dist=True)
        self.log("loss", loss['loss'], sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.calculate_loss(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        out = self(batch['board'])

        return out

    @staticmethod
    def configure_optimizer_from_params(params: Iterable[tuple[str, Parameter]],
                                        optimizer_factory: OptimizerFactory,
                                        scheduler_factory: LRSchedulerFactory):
        if optimizer_factory is None or scheduler_factory is None:
            raise RuntimeError('Optimizer and scheduler must be set for training')

        optimizer = optimizer_factory.optimizer(params=params)
        scheduler = scheduler_factory.scheduler(optimizer=optimizer)

        optimizer_config = {"optimizer": optimizer}
        optimizer_config.update(scheduler_factory.scheduler_config(scheduler=scheduler))

        return optimizer_config

    def configure_optimizers(self):
        return self.configure_optimizer_from_params(self.named_parameters(),
                                                    self.optimizer_factory,
                                                    self.lr_scheduler_factory)
