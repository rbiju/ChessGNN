from abc import abstractmethod, ABC
from pathlib import Path
import re


class Task(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError


def get_config_path(task):
    cls = task.__name__
    file_name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls).lower()

    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    yaml_path = root_dir / "configs" / "tasks" / f"{file_name}.yaml"
    return yaml_path
