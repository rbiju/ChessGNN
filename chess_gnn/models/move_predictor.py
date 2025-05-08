from typing import Iterable

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .chess_bert import ChessBERT
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.lr_schedules.lr_schedules import LRSchedulerFactory


class MovePredictionXAttnHead(nn.Module):
    def __init__(self, in_dim: int, decoder_layer: nn.TransformerDecoderLayer, num_layers: int, out_dim: int = 64):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, decoder_layer.linear1.in_features)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(decoder_layer.linear1.in_features, out_dim)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x_ = self.linear_in(x)
        pos_emb_ = self.linear_in(pos_emb)

        x_ = self.decoder(x_, pos_emb_)
        x_ = self.linear_out(x_)

        return x_


class ChessMoveXAttnPredictor(pl.LightningModule):
    def __init__(self,
                 bert: ChessBERT,
                 decoder_layer: nn.TransformerDecoderLayer,
                 n_decoder_layers: int,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 x_attn_tokens: bool = True):
        super().__init__()
        if not bert.has_learned_pos_emb:
            raise ValueError("Currently only configured for BERT models with learned positional embeddings.")
        self.bert = bert
        self.from_head = MovePredictionXAttnHead(in_dim=bert.dim, decoder_layer=decoder_layer,
                                                 num_layers=n_decoder_layers)
        self.to_head = MovePredictionXAttnHead(in_dim=bert.dim, decoder_layer=decoder_layer,
                                               num_layers=n_decoder_layers)

        self.move_loss = nn.CrossEntropyLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.x_attn_tokens = x_attn_tokens

    @staticmethod
    def squeeze_batch(batch):
        return {'board': batch['board'].squeeze(),
                'label': batch['label'].squeeze()}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.bert(x)

        if self.x_attn_tokens:
            from_prediction = self.from_head(out['cls'].unsqueeze(1), out['tokens'])
            to_prediction = self.to_head(out['cls'].unsqueeze(1), out['tokens'])
        else:
            from_prediction = self.from_head(out['cls'].unsqueeze(1), self.bert.pos_emb)
            to_prediction = self.to_head(out['cls'].unsqueeze(1), self.bert.pos_emb)

        return {'cls': out['cls'],
                'from': from_prediction.squeeze(),
                'to': to_prediction.squeeze()}

    def calculate_loss(self, batch):
        batch = self.squeeze_batch(batch)

        out = self(batch['board'])

        from_loss = self.move_loss(out['from'], batch['from'])
        to_loss = self.move_loss(out['to'], batch['to'])

        return {'loss': from_loss + to_loss}

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("train_masking_loss", loss['mask_loss'], on_step=True, sync_dist=True)
        self.log("train_win_prediction_loss", loss['win_prediction_loss'], on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("val_masking_loss", loss['mask_loss'], sync_dist=True)
        self.log("val_win_prediction_loss", loss['win_prediction_loss'], sync_dist=True)
        self.log("loss", loss['loss'], sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.calculate_loss(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        out = self(batch['board'])

        return out

    @staticmethod
    def configure_optimizer_from_params(params: Iterable[tuple[str, nn.Parameter]],
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
