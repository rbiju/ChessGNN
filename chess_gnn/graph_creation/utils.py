from typing import List, TypedDict
from chess import Board

from chess_gnn.utils import ChessPoint, ChessEdge

LEGAL = 1
ADJACENT = 0


class EdgeSet(TypedDict):
    legal: List[ChessEdge]
    adjacent: List[ChessEdge]


def get_adjacency_edges(idx) -> List[ChessEdge]:
    point = ChessPoint.from_1d(idx)
    adjacent_squares = point.adjacent()

    return [ChessEdge(point, square) for square in adjacent_squares]


def get_legal_move_edges(board: Board) -> List[ChessEdge]:
    return [ChessEdge.from_move(move) for move in board.legal_moves]


def merge_edges(adjacent: List[ChessEdge], legal: List[ChessEdge]) -> EdgeSet:
    legal_set = set(legal)
    adjacent_set = set(adjacent)
    adjacent_set = adjacent_set - legal_set

    return EdgeSet(legal=list(legal_set), adjacent=list(adjacent_set))
