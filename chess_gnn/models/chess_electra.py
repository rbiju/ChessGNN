import copy
import einops
import pytorch_lightning as pl

import torch
from torch.nn import Parameter
import torch.nn as nn

from typing import Iterable

from chess_gnn.bert import TransformerBlock, ElectraMaskHandler, Mlp
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.lr_schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer

from .chess_bert import ChessBERT


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

                cls = x_[:, :1, :].squeeze()

                return {'cls': cls,
                        'tokens': x_[:, 1:, :],
                        'attn_weights': attns}

        else:
            for block in self.encoder:
                x = block(x)

            x_ = self.norm(x)

            cls = x_[:, :1, :].squeeze()

            return {'cls': cls,
                    'tokens': x_[:, 1:, :]}


@HydraConfigurable
class ChessELECTRA(pl.LightningModule):
    def __init__(self, bert: ChessBERT,
                 discriminator: ChessDiscriminator,
                 mask_handler: ElectraMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 discriminator_dropout: float = 0.1, ):
        super().__init__()
        self.bert = bert
        self.bert.mask_handler = mask_handler
        self.discriminator = discriminator

        self.dim = self.discriminator.dim

        self.discriminator_head = Mlp(self.dim, 1, self.dim, dropout=discriminator_dropout)
        self.discriminator_loss = nn.BCEWithLogitsLoss()

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.apply(self._init_weights)
        self.save_hyperparameters()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.bert(x, whose_move)
        mlm_preds = self.bert.mlm_head(out['tokens'])
        x_ = torch.concat([out['cls'].unsqueeze(1), out['tokens']], dim=1).detach()

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

        out = self.forward_mask(batch['board'], batch['whose_move'])

        mlm_tokens = torch.argmax(out['mlm_preds'], dim=-1)  # [B L]
        mlm_preds = einops.rearrange(out['mlm_preds'], 'b l c -> b c l')
        mask_loss = self.bert.masking_loss(mlm_preds, out['mlm_labels'])

        discriminator_preds = self.discriminator_head(out['tokens']).squeeze()  # [B L]
        mlm_tokens[~out['mask']] = batch['board'][~out['mask']]
        discriminator_labels = torch.eq(mlm_tokens, batch['board']).float()
        discriminator_loss = self.discriminator_loss(discriminator_preds, discriminator_labels)

        loss = mask_loss + discriminator_loss

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
