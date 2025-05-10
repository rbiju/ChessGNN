import os
import time
import shutil

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDatasetCreator, BERTLichessDataAggregator, BERTLichessDataShuffler, HDF5DatasetBuilder

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class ParsePGN(Task):
    def __init__(self, parser: BERTLichessDatasetCreator, writer: HDF5DatasetBuilder, batch_size: int = 750, include_draw: bool = False):
        super().__init__()
        self.parser = parser
        self.batch_size = batch_size
        self.include_draw = include_draw
        self.writer = writer

    def run(self, **kwargs):
        start_time = time.time()
        data_folders = self.parser.create_dataset()
        end_time = time.time()
        print(f"PGN Parsing completed. Took {end_time - start_time} seconds.")

        for data_folder in data_folders:
            print(f"Aggregating + shuffling {data_folder}")
            aggregator = BERTLichessDataAggregator(data_folder, batch_size=self.batch_size, include_draw=self.include_draw)
            aggregated_path = aggregator.aggregate()
            shuffler = BERTLichessDataShuffler(aggregated_path)
            shuffled_file = shuffler.shuffle()
            self.writer.write_dataset(shuffled_file)

        print("Cleaning up temporary game files")
        for data_folder in data_folders:
            dirs = os.listdir(data_folder)
            for directory in dirs:
                if os.path.isdir(os.path.join(data_folder, directory)):
                    shutil.rmtree(os.path.join(data_folder, directory), ignore_errors=True)


if __name__ == '__main__':
    config_path = get_config_path('ParsePGN')
    task = ParsePGN.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
