import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from chess_gnn.bert import Mlp
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.schedules.lr_schedules import LRSchedulerFactory

from .base import ChessEncoder, ChessEngineEncoder, ChessEngine


class ChessXAttnEncoder(ChessEngineEncoder):
    def __init__(self, engine: "ChessXAttnEngine"):
        super().__init__()
        self.encoder = engine.encoder
        self.from_head = engine.from_head
        self.to_head = engine.to_head
        self.win_prediction_head = engine.win_prediction_head

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        out = self.encoder(x, whose_move, get_attn)
        from_prediction = self.from_head(out['cls'].unsqueeze(1), out['tokens'])
        to_prediction = self.to_head(out['cls'].unsqueeze(1), out['tokens'])
        win_prediction = self.win_prediction_head(out['cls'])

        return {**out,
                'from': from_prediction.squeeze(),
                'to': to_prediction.squeeze(),
                'win_probability': win_prediction.squeeze()}


@dataclass
class EngineLossWeights:
    from_loss: float = 1.0
    to_loss: float = 1.0
    win_prediction_loss: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.from_loss < 0 or self.to_loss < 0 or self.win_prediction_loss < 0:
            raise ValueError(
                f"Loss proportions must be positive: {self.from_loss, self.to_loss, self.win_prediction_loss}.")


class MovePredictionXAttnHead(nn.Module):
    def __init__(self, in_dim: int, decoder_layer: nn.TransformerDecoderLayer, num_layers: int, out_dim: int = 64):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, decoder_layer.linear1.in_features)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(decoder_layer.linear1.in_features, out_dim)

    def forward(self, x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        x_ = self.linear_in(x)
        other_ = self.linear_in(other)

        x_ = self.decoder(x_, other_)
        x_ = self.linear_out(x_)

        return x_


@HydraConfigurable
class ChessXAttnEngine(ChessEngine):
    def __init__(self,
                 encoder: ChessEncoder,
                 decoder_layer: nn.TransformerDecoderLayer,
                 n_decoder_layers: int,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 loss_weights: EngineLossWeights = EngineLossWeights(), ):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.dim
        self.from_head = MovePredictionXAttnHead(in_dim=self.dim, decoder_layer=copy.deepcopy(decoder_layer),
                                                 num_layers=n_decoder_layers)
        self.to_head = MovePredictionXAttnHead(in_dim=self.dim, decoder_layer=copy.deepcopy(decoder_layer),
                                               num_layers=n_decoder_layers)
        self.win_prediction_head = Mlp(in_dim=self.dim, out_dim=1, dropout=0.1,
                                       hidden_dim=self.dim)

        self.move_loss = nn.CrossEntropyLoss()
        self.win_prediction_loss = nn.BCEWithLogitsLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.loss_weights = loss_weights

        self.save_hyperparameters()

    def get_encoder(self) -> ChessEncoder:
        return ChessXAttnEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(x, whose_move)

        from_prediction = self.from_head(out['cls'].unsqueeze(1), out['tokens'])
        to_prediction = self.to_head(out['cls'].unsqueeze(1), out['tokens'])
        win_prediction = self.win_prediction_head(out['cls'])

        return {'cls': out['cls'],
                'tokens': out['tokens'],
                'from': from_prediction.squeeze(),
                'to': to_prediction.squeeze(),
                'win_probability': win_prediction.squeeze()}

    def calculate_loss(self, batch):
        batch = self.squeeze_batch(batch)

        out = self(batch['board'], batch['whose_move'])

        from_loss = self.move_loss(out['from'], batch['from'])
        to_loss = self.move_loss(out['to'], batch['to'])

        win_prediction_loss = self.win_prediction_loss(out['win_probability'], batch['label'])

        loss = (self.loss_weights.from_loss * from_loss +
                self.loss_weights.to_loss * to_loss +
                self.loss_weights.win_prediction_loss * win_prediction_loss)

        return {'from_loss': from_loss,
                'to_loss': to_loss,
                'win_prediction_loss': win_prediction_loss,
                'loss': loss}

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("train_from_loss", loss['from_loss'], on_step=True, sync_dist=True)
        self.log("train_to_loss", loss['to_loss'], on_step=True, sync_dist=True)
        self.log("train_win_prediction_loss", loss['win_prediction_loss'], on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("val_from_loss", loss['from_loss'], sync_dist=True)
        self.log("val_to_loss", loss['to_loss'], sync_dist=True)
        self.log("val_win_prediction_loss", loss['win_prediction_loss'], sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, sync_dist=True)

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
