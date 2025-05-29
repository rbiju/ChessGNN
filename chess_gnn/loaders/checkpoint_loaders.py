from abc import ABC, abstractmethod

from chess_gnn.models import ChessBackbone, ChessBERT, ChessELECTRA, ChessTransformer


class CheckpointLoader(ABC):
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    @abstractmethod
    def load(self) -> ChessBackbone:
        raise NotImplementedError


class BERTCheckpointLoader(CheckpointLoader):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

    def load(self) -> ChessBackbone:
        model = ChessBERT.load_from_checkpoint(self.ckpt_path)
        return model


class ELECTRACheckpointLoader(CheckpointLoader):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

    def load(self) -> ChessBackbone:
        model = ChessELECTRA.load_from_checkpoint(self.ckpt_path)
        return model


class TransformerCheckpointLoader(CheckpointLoader):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

    def load(self) -> ChessBackbone:
        model = ChessTransformer.load_from_checkpoint(self.ckpt_path)
        return model
    