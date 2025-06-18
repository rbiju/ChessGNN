import chess
import torch

from chess_gnn.models import ChessEncoder
from chess_gnn.tokenizers import SimpleChessTokenizer


class ChessBoardPredictor:
    def __init__(self, encoder: ChessEncoder):
        self.tokenizer = SimpleChessTokenizer()
        self.encoder = encoder
        self.encoder.eval()

    def single_board_forward(self, chess_board: chess.Board, get_attn: bool = False):
        board_tokens, whose_move = self.tokenizer.tokenize_board(chess_board)
        return self.encoder(board_tokens, whose_move, get_attn=get_attn)

    def get_attn_at_head_and_layer(self, chess_board: chess.Board, layer: int, head: int, get_attn: bool = True):
        out = self.single_board_forward(chess_board, get_attn=get_attn)
        return out['attns'][layer].squeeze()[head].detach().numpy()

    def get_attn_at_layer(self, chess_board: chess.Board, layer: int, get_attn: bool = True):
        out = self.single_board_forward(chess_board, get_attn=get_attn)
        return out['attns'][layer].squeeze().detach().numpy()

    def get_attn(self, chess_board: chess.Board, get_attn: bool = True):
        out = self.single_board_forward(chess_board, get_attn=get_attn)
        return torch.stack(out['attns']).squeeze().detach().numpy()
