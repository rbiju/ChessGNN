import time

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDatasetCreator, BERTLichessDataAggregator, BERTLichessDataShuffler

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class ParsePgn(Task):
    def __init__(self, parser: BERTLichessDatasetCreator, include_draw: bool = False):
        super().__init__()
        self.parser = parser
        self.include_draw = include_draw

    def run(self, **kwargs):
        start_time = time.time()
        data_folders = self.parser.create_dataset()
        end_time = time.time()
        print(f"PGN Parsing completed. Took {end_time - start_time} seconds. ")

        for data_folder in data_folders:
            print(f"Aggregating + shuffling {data_folder}")
            aggregator = BERTLichessDataAggregator(data_folder, self.include_draw)
            aggregated_path = aggregator.aggregate()
            shuffler = BERTLichessDataShuffler(aggregated_path)
            shuffler.shuffle()


if __name__ == '__main__':
    config_path = get_config_path(ParsePgn)
    task = ParsePgn.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
