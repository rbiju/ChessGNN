from abc import abstractmethod, ABC
from dataclasses import dataclass

from chess import Board
from networkx.classes import Graph
from torch import Tensor


@dataclass
class GraphCreationKwargs:
    board: Board
    board_features: Tensor


class GraphCreationStrategy(ABC):
    @abstractmethod
    def create_graph(self, args: GraphCreationKwargs) -> Graph:
        raise NotImplementedError
