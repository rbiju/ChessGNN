from abc import ABC, abstractmethod

from .environment import ChessEnvironment
from .ppo_agent import PPOAgent
from .utils import PPOData


class Rollout(ABC):
    def __init__(self, size: int):
        self.size = size

    @abstractmethod
    def generate(self, env: ChessEnvironment, agent: PPOAgent):
        raise NotImplementedError


class PPORollout(Rollout):
    def __init__(self, size: int):
        super().__init__(size)

        self.data = None

    @staticmethod
    def take_action(env: ChessEnvironment, agent: PPOAgent):
        mask = env.get_legal_moves_mask()
        board, whose_move = env.get_obs()
        action, log_prob, entropy, value = agent.get_action_and_value(board, whose_move, mask)
        reward, done = env.step(action)

        return done, PPOData(board, whose_move, reward, action, log_prob, value, mask)

    def generate(self, env: ChessEnvironment, agent: PPOAgent):
        while True:
            done, data = self.take_action(env, agent)
            if self.data is None:
                self.data = data
            else:
                self.data = self.data.cat(data)
            if done:
                break
