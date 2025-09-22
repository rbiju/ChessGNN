"""
1. Game Generation
    a: Generate games to fill up a buffer of size n
    b: Use this buffer to generate batches
    b: Collect boards, moves, rewards
    c: split into black, white moves and negate reward for black

    def take_action(self, turn: int, episode: Episode):
        mask = self.env.get_all_actions(turn)[-1]
        state = self.env.get_state(turn)

        action, prob, value = self.learner.take_action(state, mask)
        rewards, done, infos = self.env.step(action)
        self.moves[turn, self.current_ep] += 1

        self.update_stats(infos)
        goal = InfoKeys.CHECK_MATE_WIN in infos[turn]
        episode.add(state, rewards[turn], action, goal, prob, value, mask)

        return done, [state, rewards, action, goal, prob, value, mask]

    def train_episode(self, render: bool):
        renders = []

        def render_fn():
            if self.env.render_mode != "human":
                renders.append(self.env.render())

        self.env.reset()
        episode_white = Episode()
        episode_black = Episode()
        white_data: list = None
        black_data: list = None
        render_fn()

        while True:
            done, white_data = self.take_action(Pieces.WHITE, episode_white)
            self.update_enemy(black_data, episode_black, white_data[1][Pieces.BLACK])
            render_fn()
            if done:
                break

            done, black_data = self.take_action(Pieces.BLACK, episode_black)
            self.update_enemy(white_data, episode_white, black_data[1][Pieces.WHITE])
            render_fn()
            if done:
                break

        self.add_episodes(episode_white, episode_black)
        self.rewards[Pieces.BLACK, self.current_ep] = episode_black.total_reward()
        self.rewards[Pieces.WHITE, self.current_ep] = episode_white.total_reward()

        if (render or self.env.done) and self.env.render_mode != "human":
            path = os.path.join(self.result_folder, "renders", f"episode_{self.current_ep}.mp4")
            save_to_video(path, np.array(renders))

"""
from dataclasses import dataclass

import torch

from .ppo_agent import PPOAgent
from .environment import ChessEnvironment


@dataclass
class PPOData:
    board: torch.Tensor
    whose_move: torch.Tensor
    reward: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor = None
    value: torch.Tensor = None
    mask: torch.Tensor = None


class Trainer:
    def __init__(self, agent: PPOAgent, env: ChessEnvironment):
        self.agent = agent
        self.env = env

    def take_action(self):
        mask = self.env.get_legal_moves_mask()
        board, whose_move = self.env.get_obs()
        action, log_prob, entropy, value = self.agent.get_action_and_value(board, whose_move, mask)
        reward, done = self.env.step(action)

        return done, PPOData(board, whose_move, reward, action, log_prob, value, mask)

    def generate_game(self):
        self.env.reset()
        while True:
            done, data = self.take_action()
            if done:
                break
