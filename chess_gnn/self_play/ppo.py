import torch
import torch.nn as nn
import pytorch_lightning as pl

from chess_gnn.models import ChessEngineEncoder, MLPHead


class ActorCriticModel(nn.Module):
    def __init__(self, engine: ChessEngineEncoder, value_head: MLPHead):
        super().__init__()
        self.engine = engine
        self.value_head = value_head


class PPOModel(pl.LightningModule):
    def __init__(self, actor_critic: ActorCriticModel):
        super().__init__()
        self.actor_critic = actor_critic

    @staticmethod
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns
