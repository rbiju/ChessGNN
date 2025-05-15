import copy
from dataclasses import dataclass
import einops

import torch
from torch.nn import Parameter
import torch.nn as nn

from typing import Iterable

from chess_gnn.bert import TransformerBlock, ElectraMaskHandler, Mlp
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.lr_schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer

from .base import ChessBackbone, ChessEncoder
from .chess_bert import ChessBERT


class ChessElectraEncoder(ChessEncoder):
    def __init__(self, electra: "ChessELECTRA"):
        super().__init__()
        self.encoder = electra.discriminator
        self.dim = electra.dim

        self.connector = electra.connector

        self.cls_token = electra.bert.cls_token
        self.whose_move = electra.bert.whose_move_embedding
        self.embeddings = electra.bert.embedding_table
        self.pos_emb = electra.bert.pos_emb

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        x_ = self.embeddings[x]
        cls_token = self.cls_token.unsqueeze(0).expand(x_.size(0), -1, -1)
        x_ = torch.cat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move[whose_move].unsqueeze(1)
        x_ = x_ + self.pos_emb.unsqueeze(0)
        x_ = self.connector(x_)

        out = self.encoder(x_, get_attn=get_attn)

        return out


@dataclass
class ELECTRALossWeights:
    mlm: float = 2.0
    discriminator: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mlm < 0 or self.discriminator < 0:
            raise ValueError(f"Loss proportions must be positive: {self.mlm, self.discriminator}")


class ChessDiscriminator(nn.Module):
    def __init__(self, num_layers: int,
                 block: TransformerBlock,
                 tokenizer: ChessTokenizer = SimpleChessTokenizer(), ):
        super().__init__()
        self.dim = block.dim
        self.num_layers = num_layers
        self.encoder = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        self.vocab_size = tokenizer.vocab_size

        self.norm = nn.LayerNorm(block.dim)

    def forward(self, x, get_attn: bool = False):
        if get_attn:
            attns = []
            for block in self.encoder:
                x, attn = block(x, get_attn)
                attns.append(attn)

            x_ = self.norm(x)

            cls = x_[:, :1, :].squeeze(1)

            return {'cls': cls,
                    'tokens': x_[:, 1:, :],
                    'attn_weights': attns}

        else:
            for block in self.encoder:
                x = block(x)

            x_ = self.norm(x)

            cls = x_[:, :1, :].squeeze(1)

            return {'cls': cls,
                    'tokens': x_[:, 1:, :]}


@HydraConfigurable
class ChessELECTRA(ChessBackbone):
    def __init__(self, bert: ChessBERT,
                 discriminator: ChessDiscriminator,
                 mask_handler: ElectraMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 loss_weights: ELECTRALossWeights = ELECTRALossWeights(),
                 discriminator_dropout: float = 0.1, ):
        super().__init__()
        self.bert = bert
        self.bert.mask_handler = mask_handler
        self.discriminator = discriminator

        self.dim = self.discriminator.dim

        self.connector = nn.Linear(self.bert.dim, self.dim)

        self.discriminator_head = Mlp(self.dim, 1, self.dim, dropout=discriminator_dropout)
        self.discriminator_loss = nn.BCEWithLogitsLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.loss_weights = loss_weights

        self.apply(self._init_weights)
        self.save_hyperparameters()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_encoder(self):
        return ChessElectraEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.bert(x, whose_move)
        mlm_preds = self.bert.mlm_head(out['tokens'])
        x_ = torch.concat([out['cls'].unsqueeze(1), out['tokens']], dim=1).detach()
        x_ = self.connector(x_)

        discriminator_out = self.discriminator(x_)

        return {'cls': discriminator_out['cls'],
                'mlm_preds': mlm_preds,
                'tokens': discriminator_out['tokens']}

    def forward_mask(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        masked_input_ids, mask, mlm_labels = self.bert.mask_handler(x)
        out = self(masked_input_ids, whose_move)

        return {**out,
                'masked_input_ids': masked_input_ids,
                'mask': mask,
                'mlm_labels': mlm_labels}

    def calculate_loss(self, batch):
        batch = self.squeeze_batch(batch)
        b = batch['board'].shape[0]

        out = self.forward_mask(batch['board'], batch['whose_move'])

        sampled_tokens = torch.multinomial(
            nn.functional.softmax(
                einops.rearrange(out['mlm_preds'], 'b l e -> (b l) e'), dim=-1), num_samples=1).squeeze()
        sampled_tokens = einops.rearrange(sampled_tokens, '(b l) -> b l', b=b)  # [B L]
        mlm_preds = einops.rearrange(out['mlm_preds'], 'b l c -> b c l')
        mask_loss = self.bert.masking_loss(mlm_preds, out['mlm_labels'])

        discriminator_preds = self.discriminator_head(out['tokens']).squeeze()  # [B L]
        sampled_tokens[~out['mask']] = batch['board'][~out['mask']]
        discriminator_labels = torch.eq(sampled_tokens, batch['board']).float()
        discriminator_loss = self.discriminator_loss(discriminator_preds, discriminator_labels)

        loss = (self.loss_weights.mlm * mask_loss +
                self.loss_weights.discriminator * discriminator_loss)

        return {'loss': loss, 'mask_loss': mask_loss, 'discriminator_loss': discriminator_loss}

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("train_masking_loss", loss['mask_loss'], on_step=True, sync_dist=True)
        self.log("train_discriminator_loss", loss['discriminator_loss'], on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        self.log("val_masking_loss", loss['mask_loss'], sync_dist=True)
        self.log("val_discriminator_loss", loss['discriminator_loss'], sync_dist=True)
        self.log("loss", loss['loss'], sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.calculate_loss(batch)

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
