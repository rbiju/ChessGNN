from typing import List, TypedDict
from chess import Board

from chess_gnn.utils import ChessPoint, ChessEdge

LEGAL = 1
ADJACENT = 0


class EdgeSet(TypedDict):
    legal: List[tuple[int, int]]
    adjacent: List[tuple[int, int]]


def get_fc_edges() -> List[tuple[int, int]]:
    edges = []
    for idx in range(64):
        point = ChessPoint.from_1d(idx)
        edges.extend([ChessEdge(point, square).to_1d() for square in point.other_squares()])

    return edges


def get_adjacency_edges(idx) -> List[tuple[int, int]]:
    point = ChessPoint.from_1d(idx)
    adjacent_squares = point.adjacent()

    return [ChessEdge(point, square).to_1d() for square in adjacent_squares]


def get_legal_move_edges(board: Board) -> List[tuple[int, int]]:
    return [ChessEdge.from_move(move).to_1d() for move in board.legal_moves]


def merge_edges(adjacent: List[tuple[int, int]], legal: List[tuple[int, int]]) -> EdgeSet:
    legal_set = set(legal)
    adjacent_set = set(adjacent)
    adjacent_set = adjacent_set - legal_set

    return EdgeSet(legal=list(legal_set), adjacent=list(adjacent_set))
