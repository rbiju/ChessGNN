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


class ChessTransformerEncoder(ChessEncoder):
    def __init__(self, transformer: "ChessTransformer"):
        super().__init__()
        self.encoder = transformer.encoder
        self.cls_token = transformer.current_board_cls_token
        self.whose_move_embedding = transformer.whose_move_embedding
        self.embedding_table = transformer.embedding_table
        self.pos_embedding = transformer.pos_embedding

    @property
    def dim(self):
        return self.encoder.layers[0].linear1.in_features

    @staticmethod
    def norm(embedding):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        # expects a batch of boards: x.shape() = b 64
        cls_token = self.cls_token.unsqueeze(0).expand(x.size(0), -1, -1)

        x_in = torch.cat([cls_token, self.embedding_table[x]], dim=1)

        x_in = (self.norm(x_in) +
                self.norm(self.pos_embedding.unsqueeze(0)) +
                self.norm(self.whose_move_embedding[whose_move].unsqueeze(1)))

        out = self.encoder(x_in)

        return {'cls': out[:, :1, :].squeeze(1),
                'tokens': out[:, 1:, :]}


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
    def __init__(self, weight_dict: Optional[PieceWeights] = None, no_weights: bool = False):
        if weight_dict is None:
            weight_dict = {'.': 0.5, 'B': 1.0, 'K': 2.0, 'N': 1.0, 'P': 0.75, 'Q': 1.5, 'R': 1.0, 'b': 1.0,
                           'k': 2.0, 'n': 1.0, 'p': 0.75, 'q': 1.5, 'r': 1.0}
        if no_weights:
            weight_dict = {'.': 1.0, 'B': 1.0, 'K': 1.0, 'N': 1.0, 'P': 1.0, 'Q': 1.0, 'R': 1.0, 'b': 1.0,
                           'k': 1.0, 'n': 1.0, 'p': 1.0, 'q': 1.0, 'r': 1.0}

        self.weight_dict = weight_dict
        self.weights = torch.Tensor([self.weight_dict[key] for key in sorted(self.weight_dict.keys())])


@HydraConfigurable
class ChessTransformer(ChessBackbone):
    def __init__(self, encoder: nn.TransformerEncoder,
                 decoder: nn.TransformerDecoder,
                 mask_handler: TransformerMaskHandler,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 masking_schedule: MaskingSchedule,
                 loss_weights: TransformerLossWeights = TransformerLossWeights(),
                 tokenizer: ChessTokenizer = SimpleChessTokenizer(),
                 square_weights: SquareWeights = SquareWeights(),
                 from_pretrained: bool = False,):
        super().__init__()
        self.dim = encoder.layers[0].linear1.in_features
        self.decoder_dim = decoder.layers[0].linear1.in_features

        self.encoder = encoder
        self.decoder = decoder
        self.mask_handler = mask_handler

        self.current_board_cls_token = nn.Parameter(torch.empty(1, self.dim))
        self.next_board_cls_token = nn.Parameter(torch.empty(1, self.dim))
        self.embedding_table = torch.nn.Parameter(torch.empty(tokenizer.vocab_size + 1, self.dim))
        self.whose_move_embedding = nn.Parameter(torch.empty(2, self.dim))
        self.pos_embedding = nn.Parameter(torch.empty(65, self.dim))

        self.connector = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.decoder_dim), nn.GELU())

        self.decoder_norm = nn.LayerNorm(self.decoder_dim)
        self.mlm_head = nn.Linear(self.decoder_dim, tokenizer.vocab_size)

        self.masking_loss = nn.CrossEntropyLoss(weight=square_weights.weights)
        self.loss_weights = loss_weights
        self.square_weights = square_weights

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.masking_schedule = masking_schedule

        if from_pretrained:
            self.pretrained = True
        else:
            self.pretrained = False
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
            self.pos_embedding.copy_(F.normalize(self.pos_embedding, dim=-1))
            self.whose_move_embedding.copy_(F.normalize(self.whose_move_embedding, dim=-1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_encoder(self):
        return ChessTransformerEncoder(self)

    @staticmethod
    def norm(embedding):
        return F.normalize(embedding, p=2, dim=-1)

    def encode(self, board: torch.Tensor, whose_move: torch.Tensor, cls_token: torch.Tensor, ids_shuffle: torch.Tensor,
               ids_restore: torch.Tensor, len_keep: int):
        ids_keep = ids_shuffle[..., :len_keep]
        ids_mask = ids_shuffle[..., len_keep:]

        x_in = self.mask_handler.shuffle_and_mask(board, ids_shuffle, ids_restore, len_keep)

        cls_token = cls_token.unsqueeze(0).expand(x_in.size(0), -1, -1)
        x_in = torch.cat([cls_token, self.embedding_table[x_in]], dim=1)

        x_in = (self.norm(x_in) +
                self.norm(self.pos_embedding.unsqueeze(0)) +
                self.norm(self.whose_move_embedding[whose_move].unsqueeze(1)))

        decoder_in = self.mask_handler.get_masked_embeddings(x_in[:, 1:, :], ids_mask)
        encoder_in = self.mask_handler.get_unmasked_embeddings(x_in[:, 1:, :], ids_keep)
        cls_token = x_in[:, :1, :]

        encoder_in = torch.cat([cls_token, encoder_in], dim=1)
        encoder_out = self.encoder(encoder_in)

        masked_labels = self.mask_handler.get_masked_tokens(board, ids_mask)

        return {'cls': encoder_out[:, :1, :],
                'tokens': encoder_out[:, 1:, :],
                'labels': masked_labels,
                'decoder_in': decoder_in}

    def decode(self, context: torch.Tensor, decoder_in: torch.Tensor):
        decoder_in = self.connector(decoder_in)
        context = self.connector(context)

        decoder_out = self.decoder(decoder_in, context)
        decoder_out = self.decoder_norm(decoder_out)
        decoder_out = self.mlm_head(decoder_out)
        return einops.rearrange(decoder_out, 'b l c -> b c l')

    def forward(self, batch: dict[str, torch.Tensor]):
        batch = self.squeeze_batch(batch)

        # mask is shared by current and next boards
        ids_shuffle, ids_restore, len_keep = self.mask_handler.get_mask(batch['board'])
        current_board_encoded = self.encode(batch['board'], batch['whose_move'], self.current_board_cls_token,
                                            ids_shuffle, ids_restore, len_keep)
        next_board_encoded = self.encode(batch['next_board'], torch.logical_not(batch['whose_move']).long(),
                                         self.next_board_cls_token,
                                         ids_shuffle, ids_restore, len_keep)

        current_board_preds = self.decode(next_board_encoded['cls'], current_board_encoded['decoder_in'])
        next_board_preds = self.decode(current_board_encoded['cls'], next_board_encoded['decoder_in'])

        current_board_loss = self.masking_loss(current_board_preds, current_board_encoded['labels'])
        next_board_loss = self.masking_loss(next_board_preds, next_board_encoded['labels'])

        loss = (self.loss_weights.current * current_board_loss +
                self.loss_weights.next * next_board_loss)

        return {'current_board_loss': current_board_loss, 'next_board_loss': next_board_loss, 'loss': loss}

    def training_step(self, batch, batch_idx):
        step = self.global_step
        new_ratio = self.masking_schedule(step)
        self.mask_handler.set_masking_ratio(new_ratio)

        loss = self(batch)

        self.log("masking_ratio", new_ratio, prog_bar=True, on_step=True)
        self.log("train_current_board_masking_loss", loss['current_board_loss'], on_step=True, sync_dist=True)
        self.log("train_next_board_loss", loss['next_board_loss'], on_step=True, sync_dist=True)
        self.log("train_all_loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("val_current_board_masking_loss", loss['current_board_loss'], sync_dist=True)
        self.log("val_next_board_loss", loss['next_board_loss'], sync_dist=True)
        self.log("val_all_loss", loss['loss'], prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)

        self.log("test_current_board_masking_loss", loss['current_board_loss'], sync_dist=True)
        self.log("test_next_board_loss", loss['next_board_loss'], sync_dist=True)
        self.log("test_all_loss", loss['loss'], prog_bar=True, sync_dist=True)

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
