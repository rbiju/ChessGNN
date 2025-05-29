from pytorch_lightning.callbacks import BaseFinetuning

from chess_gnn.models import ChessXAttnEngine


class TransformerFreezeCallback(BaseFinetuning):
    def __init__(self, unfrozen_layers: int = 1, encoder_str: str = 'encoder.encoder'):
        super().__init__()
        self.unfrozen_layers = unfrozen_layers
        self.encoder_str = encoder_str

    def freeze_before_training(self, pl_module: ChessXAttnEngine):
        self.freeze(pl_module.get_submodule(self.encoder_str)[:-self.unfrozen_layers])
