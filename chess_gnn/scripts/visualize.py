import chess

from chess_gnn.configuration import LocalHydraConfiguration
from chess_gnn.inference import ChessBoardPredictor
from chess_gnn.models import ChessBERT
from chess_gnn.visualization import visualize_chess_attention


def test_visualize():
    board = chess.Board()

    cfg = LocalHydraConfiguration('/Users/ray/Projects/ChessGNN/configs/bert/training/bert.yaml')
    model = ChessBERT.from_hydra_configuration(cfg)

    encoder = model.get_encoder()

    predictor = ChessBoardPredictor(encoder=encoder)

    attn = predictor.get_attn_at_layer(board, layer=11)

    visualize_chess_attention(board=board, attention_matrix=attn, query='h8')


if __name__ == '__main__':
    test_visualize()
