from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDataShuffler

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class ShuffleData(Task):
    def __init__(self, shuffler: BERTLichessDataShuffler):
        super().__init__()
        self.shuffler = shuffler

    def run(self):
        self.shuffler.shuffle()


if __name__ == '__main__':
    config_path = get_config_path(ShuffleData)
    task = ShuffleData.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
