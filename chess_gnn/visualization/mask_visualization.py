import torch
import torch.nn as nn
import einops

from chess_gnn.models import ChessTransformer
from chess_gnn.tokenizers import SimpleChessTokenizer


class MaskVisualizationHelper(nn.Module):
    def __init__(self, transformer: ChessTransformer):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = SimpleChessTokenizer()

    def get_preds(self, game_batch: dict[str, torch.Tensor]):
        batch = self.transformer.squeeze_batch(game_batch)

        # mask is shared by current and next boards
        ids_shuffle, ids_restore, len_keep = self.transformer.mask_handler.get_mask(batch['board'])
        current_board_encoded = self.transformer.encode(batch['board'], batch['whose_move'],
                                                        self.transformer.current_board_cls_token,
                                                        ids_shuffle, ids_restore, len_keep)
        next_board_encoded = self.transformer.encode(batch['next_board'], torch.logical_not(batch['whose_move']).long(),
                                                     self.transformer.next_board_cls_token,
                                                     ids_shuffle, ids_restore, len_keep)

        current_board_preds = self.transformer.decode(next_board_encoded['cls'], current_board_encoded['decoder_in'])
        next_board_preds = self.transformer.decode(current_board_encoded['cls'], next_board_encoded['decoder_in'])

        current_board_preds = nn.functional.softmax(einops.rearrange(current_board_preds, 'b c l -> b l c'), dim=-1).argmax(dim=-1)
        next_board_preds = nn.functional.softmax(einops.rearrange(next_board_preds, 'b c l -> b l c'), dim=-1).argmax(dim=-1)

        mask = torch.zeros_like(batch['board'])
        mask[..., len_keep:] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)

        full_current_pred = batch['board'].clone()
        full_current_pred = torch.gather(full_current_pred, dim=1, index=ids_shuffle)
        full_current_pred[..., len_keep:] = current_board_preds
        full_current_pred = torch.gather(full_current_pred, dim=1, index=ids_restore)

        full_next_pred = batch['next_board'].clone()
        full_next_pred = torch.gather(full_next_pred, dim=1, index=ids_shuffle)
        full_next_pred[..., len_keep:] = next_board_preds
        full_next_pred = torch.gather(full_next_pred, dim=1, index=ids_restore)

        return mask, full_current_pred, full_next_pred
