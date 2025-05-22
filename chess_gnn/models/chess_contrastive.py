import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.nn import Parameter
import einops

from chess_gnn.bert import ElectraMaskHandler
from chess_gnn.configuration import HydraConfigurable
from chess_gnn.schedules.lr_schedules import LRSchedulerFactory
from chess_gnn.optimizers import OptimizerFactory

from .base import ChessBackbone, ChessEncoder
from .chess_electra import ChessDiscriminator
from .chess_bert import ChessBERT, ChessBERTEncoder
from .online_clustering import OnlineClustering


class ChessContrastiveEncoder(ChessEncoder):
    def __init__(self, backbone: "ChessContrastiveBackbone"):
        super().__init__()
        self.encoder = backbone.teacher.discriminator
        self.dim = backbone.dim

        self.cls_token = backbone.cls_token
        self.embeddings = backbone.embedding_table
        self.whose_move_embedding = backbone.whose_move_embedding
        self.pos_embedding = backbone.pos_embedding

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor, get_attn: bool = False) -> dict[str, torch.Tensor]:
        x_ = self.embeddings[x]
        cls_token = self.cls_token.unsqueeze(0).expand(x_.size(0), -1, -1)
        x_ = torch.cat([cls_token, x_], dim=1)
        x_ = x_ + self.whose_move[whose_move].unsqueeze(1)
        x_ = x_ + self.pos_emb.unsqueeze(0)

        out = self.encoder(x_, get_attn=get_attn)

        return out


class L2NormLinear(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
    ):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        nn.utils.weight_norm(self.last_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, eps=eps)
        return self.last_layer(x)


@dataclass
class ContrastiveLossWeights:
    mlm: float = 1.0
    discriminator: float = 2.0
    contrastive: float = 0.25
    clustering: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mlm < 0 or self.discriminator < 0 or self.contrastive < 0 or self.clustering < 0:
            raise ValueError(f"Loss proportions must be positive: {self.mlm, self.discriminator}")


class ChessGenerator(nn.Module):
    def __init__(self, encoder: ChessBERTEncoder, mlm_head: nn.Module, ):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = mlm_head

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor):
        out = self.encoder(x, whose_move)
        mlm_preds = self.mlm_head(out['tokens'])

        return {**out, 'mlm_preds': mlm_preds}


class ChessDINOComponent(nn.Module):
    def __init__(self, generator: ChessGenerator,
                 discriminator: ChessDiscriminator,
                 cls_token: Parameter,
                 embedding_table: Parameter,
                 whose_move_embedding: Parameter,
                 pos_embedding: Parameter, ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cls_token = cls_token
        self.embeddings = embedding_table
        self.whose_move_embedding = whose_move_embedding
        self.pos_embedding = pos_embedding

        self._mask_handler = None

    @property
    def mask_handler(self):
        return self._mask_handler

    def set_mask_handler(self, handler):
        self._mask_handler = handler

    @staticmethod
    def gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(logits).clamp(min=1e-6)
        gumbel_noise = -torch.log(-torch.log(u))
        return (logits + gumbel_noise).argmax(dim=-1)

    def mlm_forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        masked_input_ids, mask, mlm_labels = self.mask_handler(x)
        out = self.generator(masked_input_ids, whose_move)

        flat_logits = einops.rearrange(out['mlm_preds'], 'b l e -> (b l) e')
        sampled_tokens = self.gumbel_sample(flat_logits)
        sampled_tokens = einops.rearrange(sampled_tokens, '(b l) -> b l', b=x.shape[0]).detach()
        sampled_tokens[~mask] = x[~mask]

        return {'mlm_labels': mlm_labels,
                'sampled_tokens': sampled_tokens,
                'mlm_preds': einops.rearrange(out['mlm_preds'], 'b l c -> b c l')}

    def discriminator_forward(self, sampled_tokens: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        discriminator_in = self.embeddings[sampled_tokens]
        cls_token = einops.repeat(self.cls_token, 'l e -> b l e', b=sampled_tokens.shape[0])
        discriminator_in = (torch.concat([cls_token, discriminator_in], dim=1) +
                            self.whose_move_embedding[whose_move].unsqueeze(1) +
                            self.pos_embedding.unsqueeze(0))
        discriminator_out = self.discriminator(discriminator_in)

        return discriminator_out

    def forward(self):
        pass


@HydraConfigurable
class ChessContrastiveBackbone(ChessBackbone):
    def __init__(self, bert: ChessBERT,
                 discriminator: ChessDiscriminator,
                 student_mask_handler: ElectraMaskHandler,
                 teacher_mask_handler: ElectraMaskHandler,
                 clustering_head: OnlineClustering,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory,
                 loss_weights: ContrastiveLossWeights = ContrastiveLossWeights(),
                 momentum: float = 0.99):
        super().__init__()
        self.dim = discriminator.dim
        generator = ChessGenerator(bert.get_encoder(), bert.mlm_head)
        cls_token = nn.Parameter(torch.rand(1, self.dim))
        embedding_table = nn.Parameter(torch.rand(bert.vocab_size, self.dim))
        whose_move_embedding = nn.Parameter(torch.rand(2, self.dim))
        pos_emb = nn.Parameter(torch.rand(65, self.dim))
        self.student = ChessDINOComponent(generator=generator,
                                          discriminator=discriminator,
                                          cls_token=cls_token,
                                          embedding_table=embedding_table,
                                          whose_move_embedding=whose_move_embedding,
                                          pos_embedding=pos_emb, )
        self.student.set_mask_handler(student_mask_handler)

        self.teacher = copy.deepcopy(self.student)
        self.teacher.set_mask_handler(teacher_mask_handler)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.discriminator_head = nn.Linear(in_features=self.dim, out_features=1)
        self.clustering_head = clustering_head
        self.projection_head = L2NormLinear(in_dim=self.dim, out_dim=clustering_head.out_dim)

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.loss_weights = loss_weights

        self.masking_loss = nn.CrossEntropyLoss()
        self.discriminator_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()

        self.momentum = momentum

        self.save_hyperparameters()

    def get_encoder(self) -> ChessEncoder:
        return ChessContrastiveEncoder(self)

    @staticmethod
    def squeeze_batch(batch):
        return {key: batch[key].squeeze() for key in batch}

    def forward(self, x: torch.Tensor, whose_move: torch.Tensor) -> dict[str, torch.Tensor]:
        student_mlm = self.student.mlm_forward(x, whose_move)
        teacher_mlm = self.teacher.mlm_forward(x, whose_move)

        # masking loss
        mlm_loss = self.masking_loss(student_mlm['mlm_preds'], student_mlm['mlm_labels'])

        # discriminator loss
        student_discriminator = self.student.discriminator_forward(student_mlm['sampled_tokens'], whose_move)
        discriminator_preds = self.discriminator_head(student_discriminator['tokens']).squeeze()  # [B L]
        discriminator_labels = torch.eq(student_mlm['sampled_tokens'], x).float()
        discriminator_loss = self.discriminator_loss(discriminator_preds, discriminator_labels)

        # clustering loss
        teacher_discriminator = self.teacher.discriminator_forward(teacher_mlm['sampled_tokens'], whose_move)
        assignments, clustering_loss = self.clustering_head(teacher_discriminator['cls'])
        contrastive_loss = self.contrastive_loss(self.projection_head(student_discriminator['cls']), assignments)

        loss = (self.loss_weights.mlm * mlm_loss +
                self.loss_weights.discriminator * discriminator_loss +
                self.loss_weights.clustering * clustering_loss +
                self.loss_weights.contrastive * contrastive_loss)

        return {'mlm_loss': mlm_loss,
                'discriminator_loss': discriminator_loss,
                'clustering_loss': clustering_loss,
                'contrastive_loss': contrastive_loss,
                'loss': loss}

    def training_step(self, batch, batch_idx):
        batch = self.squeeze_batch(batch)
        loss = self(batch['board'], batch['whose_move'])

        with torch.no_grad():
            for student_ps, teacher_ps in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_ps.data.mul_(self.momentum)
                teacher_ps.data.add_(
                    (1 - self.momentum) * student_ps.detach().data
                )

        self.log("train_mlm_loss", loss['mlm_loss'], on_step=True, sync_dist=True)
        self.log("train_discriminator_loss", loss['discriminator_loss'], on_step=True, sync_dist=True)
        self.log("train_clustering_loss", loss['clustering_loss'], on_step=True, sync_dist=True)
        self.log("train_contrastive_loss", loss['contrastive_loss'], on_step=True, sync_dist=True)
        self.log("loss", loss['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.squeeze_batch(batch)
        loss = self(batch['board'], batch['whose_move'])

        self.log("val_mlm_loss", loss['mlm_loss'], sync_dist=True)
        self.log("val_discriminator_loss", loss['discriminator_loss'], sync_dist=True)
        self.log("val_clustering_loss", loss['clustering_loss'], sync_dist=True)
        self.log("val_contrastive_loss", loss['contrastive_loss'], sync_dist=True)
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
