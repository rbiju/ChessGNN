from typing import Iterable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from chess_gnn.models import ChessEngineEncoder
from chess_gnn.optimizers import OptimizerFactory
from chess_gnn.schedules import LRSchedulerFactory
from .loss import policy_loss, entropy_loss, value_loss


class PPOLightningAgent(pl.LightningModule):
    def __init__(
        self,
        actor: ChessEngineEncoder,
        critic: torch.nn.Module,
        optimizer_factory: OptimizerFactory,
        lr_scheduler_factory: LRSchedulerFactory,
        vf_coef: float = 1.0,
        ent_coef: float = 0.0,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        normalize_advantages: bool = False,
        **torchmetrics_kwargs,
    ):
        super().__init__()
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.critic = critic
        self.actor = actor
        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)

        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

    def get_action(self, board: Tensor, whose_move: Tensor, action: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.actor.get_action_logits(board, whose_move)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()

    def get_greedy_action(self, board: Tensor, whose_move: Tensor) -> Tensor:
        logits = self.actor.get_action_logits(board, whose_move)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def get_value(self, board: Tensor, whose_move: Tensor) -> Tensor:
        return self.critic(board, whose_move)

    def get_action_and_value(self, board: Tensor, whose_move: Tensor, action: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action, log_prob, entropy = self.get_action(board, action)
        value = self.get_value(board, whose_move)
        return action, log_prob, entropy, value

    def forward(self, x: Tensor, whose_move: Tensor, action: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.get_action_and_value(x, whose_move, action)

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[Tensor, Tensor]:
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = torch.logical_not(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = torch.logical_not(dones[t + 1])
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def training_step(self, batch: dict[str, Tensor]):
        # Get actions and values given the current observations
        _, newlogprob, entropy, newvalue = self(batch["obs"], batch["actions"].long())
        logratio = newlogprob - batch["logprobs"]
        ratio = logratio.exp()

        # Policy loss
        advantages = batch["advantages"]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = policy_loss(batch["advantages"], ratio, self.clip_coef)

        # Value loss
        v_loss = value_loss(
            newvalue,
            batch["values"],
            batch["returns"],
            self.clip_coef,
            self.clip_vloss,
            self.vf_coef,
        )

        # Entropy loss
        ent_loss = entropy_loss(entropy, self.ent_coef)

        # Update metrics
        self.avg_pg_loss(pg_loss)
        self.avg_value_loss(v_loss)
        self.avg_ent_loss(ent_loss)

        # Overall loss
        return pg_loss + ent_loss + v_loss

    def reset_metrics(self):
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    @staticmethod
    def configure_optimizer_from_params(params: Iterable[tuple[str, torch.nn.Parameter]],
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
