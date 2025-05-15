import einops
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox
import skunk
import chess
import chess.svg
import numpy as np

from chess_gnn.utils import ChessPoint


def visualize_chess_attention(board: chess.Board, attention_matrix: np.ndarray, query: str):
    """
    Plots 8x8 attention grid and inserts multiple SVGs at specified squares.

    Parameters:
        board: chess board to plot
        attention_matrix (np.ndarray): 65x65 attention matrix, includes the class token at position 1
        query: either 'cls' or a chess square
    """
    if query.lower() == 'cls':
        point_idx = 0
    else:
        point = ChessPoint.from_square(query)
        point_idx = point.to_1d()

    attention = einops.rearrange(attention_matrix[point_idx][1:], '(h w) -> h w', h=8)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attention, cmap='viridis')

    svg_map = {}

    for idx in range(64):
        point = ChessPoint.from_1d(idx)
        box_id = f"sk_{idx}"
        piece = board.piece_at(idx)
        if piece is not None:
            svg_map[box_id] = chess.svg.piece(board.piece_at(idx))
            box = skunk.Box(40, 40, box_id)
            ab = AnnotationBbox(box, (point.x, point.y), frameon=False)
            ax.add_artist(ab)

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(range(8))
    ax.set_yticklabels(range(8))
    ax.set_title("Chess Attention")
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Insert all SVGs
    svg_out = skunk.insert(svg_map)

    with open('board.svg', 'w') as f:
        f.write(svg_out)
