import chess
import torch

from chess_gnn.models import ChessEncoder
from chess_gnn.tokenizers import SimpleChessTokenizer
from chess_gnn.utils import process_board_string


class ChessBoardPredictor:
    def __init__(self, encoder: ChessEncoder):
        self.tokenizer = SimpleChessTokenizer()
        self.encoder = encoder
        self.encoder.eval()

    def single_board_forward(self, chess_board: chess.Board, get_attn: bool = False):
        board = process_board_string(str(chess_board))
        board_tokens = torch.Tensor(self.tokenizer.tokenize(board)).long().unsqueeze(0)
        whose_move = torch.Tensor([int(not chess_board.turn)]).long()

        return self.encoder(board_tokens, whose_move, get_attn=get_attn)

    def get_attn_at_layer(self, chess_board: chess.Board, layer: int, get_attn: bool = True):
        out = self.single_board_forward(chess_board, get_attn=get_attn)
        return out['attns'][layer].squeeze().detach().numpy()
