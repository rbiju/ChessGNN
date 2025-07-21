"""
1. Game Generation
    a: Inference to fill deque of size n
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
from .ppo_agent import PPOLightningAgent
from .environment import ChessEnvironment


class Trainer:
    def __init__(self, agent: PPOLightningAgent, env: ChessEnvironment):
        self.agent = agent
        self.env = env




