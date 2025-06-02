import os
from pathlib import Path
import shutil
from typing import List, Union
from functools import partial

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import BERTLichessDatasetCreator, BERTLichessDataAggregator, BERTLichessDataShuffler, HDF5DatasetBuilder

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class ParsePGN(Task):
    def __init__(self, parser: Union[BERTLichessDatasetCreator, partial], writer: HDF5DatasetBuilder, batch_size: int = 750, include_draw: bool = False, pgn_files: List[str] = None):
        super().__init__()
        self.parser = parser
        if isinstance(parser, partial):
            if pgn_files is None:
                raise ValueError("pgn_files is required if parser is partial")

        self.pgn_files = pgn_files
        self.batch_size = batch_size
        self.include_draw = include_draw
        self.writer = writer

    def _aggregate_shuffle_h5(self, data_folders: list[Path]):
        for data_folder in data_folders:
            print(f"Aggregating + shuffling {data_folder}")
            aggregator = BERTLichessDataAggregator(str(data_folder), batch_size=self.batch_size, include_draw=self.include_draw)
            aggregated_path = aggregator.aggregate()
            shuffler = BERTLichessDataShuffler(aggregated_path)
            shuffled_file = shuffler.shuffle()
            self.writer.write_dataset(shuffled_file)

        print("Cleaning up temporary game files")
        for data_folder in data_folders:
            for name in os.listdir(data_folder):
                full_path = os.path.join(data_folder, name)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path, ignore_errors=True)
                elif os.path.isfile(full_path) and name.endswith('.txt'):
                    os.remove(full_path)
                else:
                    continue

    def run(self, **kwargs):
        if self.pgn_files:
            for pgn_file in self.pgn_files:
                parser = self.parser(pgn_file)
                data_folders = parser.create_dataset()
        else:
            data_folders = self.parser.create_dataset()

        self._aggregate_shuffle_h5(data_folders)


if __name__ == '__main__':
    config_path = get_config_path('ParsePGN')
    task = ParsePGN.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
