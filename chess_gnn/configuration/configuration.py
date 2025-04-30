from abc import ABC, abstractmethod
from typing import Any
import functools

import hydra
from hydra import initialize
from omegaconf import OmegaConf, DictConfig


class HydraConfiguration(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def cfg(self) -> DictConfig:
        raise NotImplementedError

    def resolve(self, key: str, error_msg: str = None):
        try:
            cfg = self.cfg[key]
        except KeyError:
            if error_msg is None:
                error_msg = f'Key {key} not found in Hydra config'
            raise KeyError(error_msg)
        obj = hydra.utils.instantiate(cfg, _convert_='all')
        return obj


class LocalHydraConfiguration(HydraConfiguration):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._cfg = OmegaConf.load(self.file_path)

    @property
    def cfg(self) -> DictConfig:
        return self._cfg


def HydraConfigurable(cls: Any):
    @functools.wraps(cls, updated=())
    class ConfigurableClass(cls):
        @staticmethod
        def from_hydra_configuration(config: HydraConfiguration):
            cfg = config.cfg
            cfg = cfg[cls.__name__]
            with initialize(version_base=None):
                obj = hydra.utils.instantiate(cfg, _convert_='all')
                return obj

    return ConfigurableClass
