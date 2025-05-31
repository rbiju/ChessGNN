from pytorch_lightning.callbacks import BaseFinetuning

from chess_gnn.models import ChessXAttnEngine


class TransformerFreezeCallback(BaseFinetuning):
    def __init__(self, layers_to_freeze: int = 1, encoder_str: str = 'encoder.encoder.encoder'):
        super().__init__()
        self.layers_to_freeze = layers_to_freeze
        self.encoder_str = encoder_str

    def freeze_before_training(self, pl_module: ChessXAttnEngine):
        self.freeze(pl_module.get_submodule(self.encoder_str)[:self.layers_to_freeze])

    def finetune_function(self, pl_module, current_epoch, optimizer):
        pass
