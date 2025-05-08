import os
from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDataAggregator, BERTLichessDataShuffler

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class AggregateData(Task):
    def __init__(self, data_directory: str, batch_size: int, include_draw: bool = False):
        super().__init__()
        self.data_locations = [os.path.join(data_directory, directory) for directory in os.listdir(data_directory)
                               if os.path.isdir(os.path.join(data_directory, directory))]
        self.batch_size = batch_size
        self.include_draw = include_draw

    def run(self, **kwargs):
        for data_location in self.data_locations:
            print(f"Aggregating + Shuffling data for {data_location}")
            aggregator = BERTLichessDataAggregator(data_location=data_location, batch_size=self.batch_size, include_draw=self.include_draw)
            aggregated_path = aggregator.aggregate()
            shuffler = BERTLichessDataShuffler(aggregated_path)
            shuffler.shuffle()


if __name__ == '__main__':
    config_path = get_config_path('AggregateData')
    task = AggregateData.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
