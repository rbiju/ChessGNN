from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.schedules.lr_schedules import LRSchedulerFactory

from .base import ChessEncoder, ChessEngineEncoder, ChessEngine, EngineLossWeights, MLPHead


class ChessMLPEngineEncoder(ChessEngineEncoder):
    def __init__(self, engine: "ChessMLPEngine"):
        super().__init__()
        self.encoder = engine.encoder
        self.move_prediction_head = engine.move_prediction_head
        self.win_prediction_head = engine.win_prediction_head

    @property
    def dim(self):
        return self.encoder.dim

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        out = self.encoder(x, whose_move)
        move_predictions = self.move_prediction_head(out['tokens'])
        win_prediction = self.win_prediction_head(out['cls'])

        return {'cls': out['cls'],
                'tokens': out['tokens'],
                'from': F.softmax(move_predictions.squeeze()[..., 0], dim=-1),
                'to': F.softmax(move_predictions.squeeze()[..., 1], dim=-1),
                'win_probability': F.softmax(win_prediction.squeeze(), dim=-1)}


@HydraConfigurable
class ChessMLPEngine(ChessEngine):
    def __init__(self,
                 encoder: ChessEncoder,
                 move_prediction_head: MLPHead,
                 win_prediction_head: MLPHead,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 loss_weights: EngineLossWeights = EngineLossWeights(), ):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.dim

        self.move_prediction_head = move_prediction_head
        self.win_prediction_head = win_prediction_head

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.loss_weights = loss_weights

        self.save_hyperparameters()

    def get_encoder(self) -> ChessEncoder:
        return ChessMLPEngineEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    @staticmethod
    def convert_labels_to_probs(labels: torch.LongTensor) -> torch.Tensor:
        probs = torch.zeros((labels.size(0), 2), dtype=torch.float32, device=labels.device)
        mask_0 = labels == 0
        mask_1 = labels == 1
        mask_2 = labels == 2

        probs[mask_0, 0] = 1.0
        probs[mask_1, 1] = 1.0
        probs[mask_2] = 0.5

        return probs

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(x, whose_move)

        move_predictions = self.move_prediction_head(out['tokens'])
        win_prediction = self.win_prediction_head(out['cls'])

        return {'cls': out['cls'],
                'tokens': out['tokens'],
                'from': move_predictions.squeeze()[..., 0],
                'to': move_predictions.squeeze()[..., 1],
                'win_probability': win_prediction.squeeze()}

    def calculate_loss(self, batch):
        batch = self.squeeze_batch(batch)

        out = self(batch['board'], batch['whose_move'])

        from_loss = self.loss_fn(out['from'], batch['from'])
        to_loss = self.loss_fn(out['to'], batch['to'])

        win_prediction_loss = self.loss_fn(out['win_probability'], self.convert_labels_to_probs(batch['label']))

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
    def configure_optimizer_from_params(params: Iterator[nn.Parameter],
                                        optimizer_factory: OptimizerFactory,
                                        scheduler_factory: LRSchedulerFactory):
        if optimizer_factory is None or scheduler_factory is None:
            raise RuntimeError('Optimizer and scheduler must be set for training')

        optimizer = optimizer_factory.optimizer(filter(lambda p: p.requires_grad, params))
        scheduler = scheduler_factory.scheduler(optimizer=optimizer)

        optimizer_config = {"optimizer": optimizer}
        optimizer_config.update(scheduler_factory.scheduler_config(scheduler=scheduler))

        return optimizer_config

    def configure_optimizers(self):
        return self.configure_optimizer_from_params(self.parameters(),
                                                    self.optimizer_factory,
                                                    self.lr_scheduler_factory)
