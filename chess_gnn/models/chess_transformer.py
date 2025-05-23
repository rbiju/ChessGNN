from dataclasses import dataclass
from typing import Iterable, TypedDict, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from chess_gnn.bert import TransformerMaskHandler
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.schedules import LRSchedulerFactory, MaskingSchedule
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer
from .base import ChessBackbone, ChessEncoder
from .chess_electra import ChessDiscriminator


class ChessTransformerEncoder(ChessEncoder):
    def __init__(self, transformer: "ChessTransformer"):
        super().__init__()
        self.encoder = transformer.encoder
        self.dim = transformer.dim

        self.cls_token = transformer.current_board_cls_token
        self.whose_move = transformer.whose_move_embedding
        self.embeddings = transformer.embedding_table
        self.pos_emb = transformer.pos_embedding

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        x_ = self.embeddings[x]
        x_ = x_ + self.pos_emb.unsqueeze(0)
        cls_token = self.cls_token.unsqueeze(0).expand(x_.size(0), -1, -1)
        x_ = torch.cat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move[whose_move].unsqueeze(1)

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


PieceWeights = TypedDict('PieceWeights', {
    '.': float,
    'B': float,
    'K': float,
    'N': float,
    'P': float,
    'Q': float,
    'R': float,
    'b': float,
    'k': float,
    'n': float,
    'p': float,
    'q': float,
    'r': float
})


class SquareWeights:
    def __init__(self, weight_dict: Optional[PieceWeights] = None):
        if weight_dict is None:
            weight_dict = {'.': 0.1, 'B': 1.0, 'K': 2.0, 'N': 1.0, 'P': 0.2, 'Q': 1.5, 'R': 1.0, 'b': 1.0,
                           'k': 2.0, 'n': 1.0, 'p': 0.2, 'q': 1.5, 'r': 1.0}

        self.weight_dict = weight_dict
        self.weights = torch.Tensor([self.weight_dict[key] for key in sorted(self.weight_dict.keys())])


@HydraConfigurable
class ChessTransformer(ChessBackbone):
    def __init__(self, encoder: ChessDiscriminator,
                 decoder: nn.TransformerDecoder,
                 mask_handler: TransformerMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 masking_schedule: MaskingSchedule,
                 loss_weights: TransformerLossWeights = TransformerLossWeights(),
                 tokenizer: ChessTokenizer = SimpleChessTokenizer(),
                 square_weights: SquareWeights = SquareWeights()):
        super().__init__()
        self.dim = encoder.dim
        self.decoder_dim = decoder.layers[0].linear1.in_features

        self.encoder = encoder
        self.decoder = decoder
        self.mask_handler = mask_handler

        self.current_board_cls_token = nn.Parameter(torch.empty(1, self.dim))
        self.next_board_cls_token = nn.Parameter(torch.empty(1, self.dim))
        self.embedding_table = torch.nn.Parameter(torch.empty(tokenizer.vocab_size + 1, self.dim))
        self.whose_move_embedding = nn.Parameter(torch.empty(2, self.dim))
        self.pos_embedding = nn.Parameter(torch.empty(64, self.dim))

        self.connector = nn.Linear(self.dim, self.decoder_dim)
        self.mlm_head = nn.Linear(self.decoder_dim, tokenizer.vocab_size)

        self.masking_loss = nn.CrossEntropyLoss(weight=square_weights.weights)
        self.loss_weights = loss_weights
        self.square_weights = square_weights

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.masking_schedule = masking_schedule

        self.total_steps = None

        self.initialize_weights()
        self.save_hyperparameters()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.current_board_cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.next_board_cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.embedding_table, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        torch.nn.init.trunc_normal_(self.whose_move_embedding, std=0.02)

        with torch.no_grad():
            self.embedding_table.copy_(F.normalize(self.embedding_table, dim=-1))
            self.current_board_cls_token.copy_(F.normalize(self.current_board_cls_token, dim=-1))
            self.next_board_cls_token.copy_(F.normalize(self.next_board_cls_token, dim=-1))

        self.apply(self._init_weights)

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

    @staticmethod
    def norm(embedding):
        return F.normalize(embedding, p=2, dim=-1)

    def encode(self, board: torch.Tensor, whose_move: torch.Tensor, cls_token: torch.Tensor, ids_shuffle: torch.Tensor,
               ids_restore: torch.Tensor, len_keep: int):
        ids_keep = ids_shuffle[..., :len_keep]
        ids_mask = ids_shuffle[..., len_keep:]

        x_in = self.mask_handler.shuffle_and_mask(board, ids_shuffle, ids_restore, len_keep)

        x_in = self.norm(self.embedding_table)[x_in] + self.norm(self.pos_embedding).unsqueeze(0)

        x_in = x_in + self.norm(self.whose_move_embedding)[whose_move].unsqueeze(1)
        decoder_in = self.mask_handler.get_masked_embeddings(x_in, ids_mask)
        encoder_in = self.mask_handler.get_unmasked_embeddings(x_in, ids_keep)

        cls_token = (self.norm(cls_token).unsqueeze(0).expand(x_in.size(0), -1, -1) +
                     self.norm(self.whose_move_embedding)[whose_move].unsqueeze(1))
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
        current_board_encoded = self.encode(batch['board'], batch['whose_move'], self.current_board_cls_token, ids_shuffle, ids_restore, len_keep)
        next_board_encoded = self.encode(batch['next_board'], torch.logical_not(batch['whose_move']).long(), self.next_board_cls_token,
                                         ids_shuffle, ids_restore, len_keep)

        current_board_preds = self.decode(next_board_encoded['cls'], current_board_encoded['decoder_in'])
        next_board_preds = self.decode(current_board_encoded['cls'], next_board_encoded['decoder_in'])

        current_board_loss = self.masking_loss(current_board_preds, current_board_encoded['labels'])
        next_board_loss = self.masking_loss(next_board_preds, next_board_encoded['labels'])

        loss = (self.loss_weights.current * current_board_loss +
                self.loss_weights.next * next_board_loss)

        return {'current_board_loss': current_board_loss, 'next_board_loss': next_board_loss, 'loss': loss}

    def visualize_forward(self, batch: dict[str, torch.Tensor]):
        batch = self.squeeze_batch(batch)

        # mask is shared by current and next boards
        ids_shuffle, ids_restore, len_keep = self.mask_handler.get_mask(batch['board'])
        current_board_encoded = self.encode(batch['board'], batch['whose_move'], self.current_board_cls_token, ids_shuffle, ids_restore, len_keep)
        next_board_encoded = self.encode(batch['next_board'], torch.logical_not(batch['whose_move']).long(), self.next_board_cls_token,
                                         ids_shuffle, ids_restore, len_keep)

        current_board_preds = self.decode(next_board_encoded['cls'], current_board_encoded['decoder_in'])
        next_board_preds = self.decode(current_board_encoded['cls'], next_board_encoded['decoder_in'])

        return {'current_board_preds': current_board_preds, 'next_board_preds': next_board_preds, 'ids_shuffle': ids_shuffle, 'len_keep': len_keep}

    def training_step(self, batch, batch_idx):
        step = self.global_step
        new_ratio = self.masking_schedule(step)
        self.mask_handler.set_masking_ratio(new_ratio)

        loss = self(batch)

        self.log("masking_ratio", new_ratio, prog_bar=True, on_step=True)
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
