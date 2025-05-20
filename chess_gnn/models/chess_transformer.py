from dataclasses import dataclass
from typing import Iterable

import einops
import torch
import torch.nn as nn
from torch.nn import Parameter

from chess_gnn.bert import TransformerMaskHandler
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.lr_schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer
from .base import ChessBackbone, ChessEncoder
from .chess_electra import ChessDiscriminator


class ChessTransformerEncoder(ChessEncoder):
    def __init__(self, transformer: "ChessTransformer"):
        super().__init__()
        self.encoder = transformer.encoder
        self.dim = transformer.dim

        self.cls_token = transformer.cls_token
        self.whose_move = transformer.whose_move_embedding
        self.embeddings = transformer.embedding_table
        self.pos_emb = transformer.pos_embedding

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
class TransformerLossWeights:
    current: float = 1.0
    next: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.current < 0 or self.next < 0:
            raise ValueError(f"Loss proportions must be positive: {self.current, self.next}")


@HydraConfigurable
class ChessTransformer(ChessBackbone):
    def __init__(self, encoder: ChessDiscriminator,
                 decoder: nn.TransformerDecoder,
                 mask_handler: TransformerMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 loss_weights: TransformerLossWeights = TransformerLossWeights(),
                 tokenizer: ChessTokenizer = SimpleChessTokenizer()):
        super().__init__()
        self.dim = encoder.dim
        self.decoder_dim = decoder.layers[0].linear1.in_features

        self.encoder = encoder
        self.decoder = decoder
        self.mask_handler = mask_handler

        self.cls_token = nn.Parameter(torch.rand(1, self.dim))
        self.embedding_table = torch.nn.Parameter(torch.rand(tokenizer.vocab_size + 1, self.dim))
        self.whose_move_embedding = nn.Parameter(torch.rand(2, self.dim))
        self.pos_embedding = nn.Parameter(torch.rand(64, self.dim))

        self.connector = nn.Linear(self.dim, self.decoder_dim)
        self.mlm_head = nn.Linear(self.decoder_dim, tokenizer.vocab_size)

        self.masking_loss = nn.CrossEntropyLoss()
        self.loss_weights = loss_weights

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

        self.apply(self._init_weights)
        self.save_hyperparameters()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_encoder(self):
        return ChessTransformerEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def encode(self, board: torch.Tensor, whose_move: torch.Tensor, ids_shuffle: torch.Tensor, ids_restore: torch.Tensor, len_keep: int):
        ids_keep = ids_shuffle[..., :len_keep]
        ids_mask = ids_shuffle[..., len_keep:]

        x_in = self.mask_handler.shuffle_and_mask(board, ids_shuffle, ids_restore, len_keep)

        x_in = self.embedding_table[x_in] + self.pos_embedding.unsqueeze(0)

        x_in = x_in + self.whose_move_embedding[whose_move].unsqueeze(1)
        decoder_in = self.mask_handler.get_masked_embeddings(x_in, ids_mask)
        encoder_in = self.mask_handler.get_unmasked_embeddings(x_in, ids_keep)

        cls_token = self.cls_token.unsqueeze(0).expand(x_in.size(0), -1, -1) + self.whose_move_embedding[whose_move].unsqueeze(1)
        encoder_in = torch.cat([cls_token, encoder_in], dim=1)
        encoder_out = self.encoder(encoder_in)

        masked_labels = self.mask_handler.get_masked_tokens(board, ids_mask)

        return {'cls': encoder_out['cls'].unsqueeze(1),
                'tokens': encoder_out['tokens'],
                'labels': masked_labels,
                'decoder_in': decoder_in}

    def decode(self, cls_token: torch.Tensor, decoder_in: torch.Tensor):
        decoder_in = self.connector(decoder_in)
        cls_token = self.connector(cls_token)

        decoder_out = self.decoder(decoder_in, cls_token)
        decoder_out = self.mlm_head(decoder_out)
        return einops.rearrange(decoder_out, 'b l c -> b c l')

    def forward(self, batch: dict[str, torch.Tensor]):
        batch = self.squeeze_batch(batch)

        # mask is shared by current and next boards
        ids_shuffle, ids_restore, len_keep = self.mask_handler.get_mask(batch['board'])
        current_board_encoded = self.encode(batch['board'], batch['whose_move'], ids_shuffle, ids_restore, len_keep)
        next_board_encoded = self.encode(batch['next_board'], torch.logical_not(batch['whose_move']).long(), ids_shuffle, ids_restore, len_keep)

        current_board_preds = self.decode(next_board_encoded['cls'], current_board_encoded['decoder_in'])
        next_board_preds = self.decode(current_board_encoded['cls'], next_board_encoded['decoder_in'])

        current_board_loss = self.masking_loss(current_board_preds, current_board_encoded['labels'])
        next_board_loss = self.masking_loss(next_board_preds, next_board_encoded['labels'])

        loss = (self.loss_weights.current * current_board_loss +
                self.loss_weights.next * next_board_loss)

        return {'current_board_loss': current_board_loss, 'next_board_loss': next_board_loss, 'loss': loss}

    def training_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("train_current_board_masking_loss", loss['current_board_loss'], on_step=True, sync_dist=True)
        self.log("train_next_board_loss", loss['next_board_loss'], on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("val_current_board_masking_loss", loss['current_board_loss'], sync_dist=True)
        self.log("val_next_board_loss", loss['next_board_loss'], sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("val_current_board_masking_loss", loss['current_board_loss'], sync_dist=True)
        self.log("val_next_board_loss", loss['next_board_loss'], sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, sync_dist=True)

        return loss

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
