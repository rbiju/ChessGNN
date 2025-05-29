import copy
from dataclasses import dataclass

import einops
from einops import repeat

import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from typing import Iterable, Optional

from chess_gnn.bert import TransformerBlock, BERTMaskHandler, Mlp
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.tokenizers import ChessTokenizer

from .base import ChessBackbone, ChessEncoder


class ChessBERTEncoder(ChessEncoder):
    def __init__(self, bert: "ChessBERT"):
        super().__init__()
        self.encoder = bert.encoder
        self.norm = bert.norm
        self._dim = bert.dim

        self.cls_token = bert.cls_token
        self.whose_move_embedding = bert.whose_move_embedding
        self.embedding_table = bert.embedding_table

        self.pos_emb = bert.pos_emb

    @property
    def dim(self):
        return self._dim

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        x_ = self.embedding_table[x]
        cls_token = self.cls_token.unsqueeze(0).expand(x_.size(0), -1, -1)
        x_ = torch.cat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move_embedding[whose_move].unsqueeze(1)
        x_ = x_ + self.pos_emb.unsqueeze(0)

        if get_attn:
            attns = []
            for block in self.encoder:
                x_, attn = block(x_, get_attn)
                attns.append(attn)
            x_ = self.norm(x_)
            cls = x_[:, :1, :].squeeze(1)

            return {'cls': cls,
                    'tokens': x_[:, 1:, :],
                    'attns': attns}
        else:
            for block in self.encoder:
                x_ = block(x_)
            x_ = self.norm(x_)
            cls = x_[:, :1, :].squeeze(1)

            return {'cls': cls,
                    'tokens': x_[:, 1:, :]}


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
class ChessBERT(ChessBackbone):
    def __init__(self, num_layers: int,
                 block: TransformerBlock,
                 tokenizer: ChessTokenizer,
                 mask_handler: BERTMaskHandler,
                 optimizer_factory: Optional[OptimizerFactory] = None,
                 lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
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
        self.win_prediction_head = Mlp(in_dim=block.dim, out_dim=1, dropout=win_prediction_dropout,
                                       hidden_dim=block.dim)
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
        self.save_hyperparameters()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.embedding_table, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_encoder(self):
        return ChessBERTEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def normalized_embedding(self):
        return F.normalize(self.embedding_table, p=2, dim=-1)

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        x_ = self.normalized_embedding()[x]
        cls_token = repeat(self.cls_token, 'l e -> b l e', b=x.shape[0])
        x_ = torch.concat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move_embedding[whose_move].unsqueeze(1)

        if self.has_learned_pos_emb:
            x_ = x_ + self.pos_emb.unsqueeze(0)

        if get_attn:
            attns = []
            for block in self.encoder:
                x_, attn = block(x_, get_attn)
                attns.append(attn)

            x_ = self.norm(x_)

            cls = x_[:, :1, :].squeeze()

            return {'cls': cls,
                    'tokens': x_[:, 1:, :],
                    'attn_weights': attns}

        else:
            for block in self.encoder:
                x_ = block(x_)

            x_ = self.norm(x_)

            cls = x_[:, :1, :].squeeze()

            return {'cls': cls,
                    'tokens': x_[:, 1:, :]}

    def forward_mask(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        x, labels = self.mask_handler(x)

        out = self(x, whose_move)

        win_prob = self.win_prediction_head(out['cls']).squeeze()

        return {**out, 'win_probability': win_prob, 'masked_token_labels': labels}

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
