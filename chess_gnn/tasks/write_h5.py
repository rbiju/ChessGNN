import shutil
from pathlib import Path

from chess_gnn.configuration import HydraConfigurable, LocalHydraConfiguration
from chess_gnn.data_creation import HDF5DatasetBuilder

from chess_gnn.tasks.base import Task, get_config_path


@HydraConfigurable
class WriteH5(Task):
    def __init__(self, data_directory: str, writer: HDF5DatasetBuilder, prune_data_files: bool = False):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.writer = writer
        self.prune_data_files = prune_data_files

    def run(self, **kwargs):
        files = []
        split_folders = []
        for split_folder in self.data_directory.iterdir():
            if split_folder.is_dir():
                split_folders.append(split_folder)
                data_file = split_folder / 'shuffled.txt'
                if data_file.is_file():
                    files.append(data_file)

        for file in files:
            self.writer.write_dataset(file)

        data_dirs = []
        for split_folder in split_folders:
            for label_folder in split_folder.iterdir():
                if label_folder.is_dir():
                    data_dirs.append(label_folder)

        if self.prune_data_files:
            for data_dir in data_dirs:
                shutil.rmtree(str(data_dir))


if __name__ == '__main__':
    config_path = get_config_path(WriteH5)
    task = WriteH5.from_hydra_configuration(LocalHydraConfiguration(str(config_path)))
    task.run()
