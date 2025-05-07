from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDataAggregator

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class AggregateData(Task):
    def __init__(self, aggregator: BERTLichessDataAggregator):
        super().__init__()
        self.aggregator = aggregator

    def run(self, **kwargs):
        self.aggregator.aggregate()


if __name__ == '__main__':
    config_path = get_config_path(AggregateData)
    task = AggregateData.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
