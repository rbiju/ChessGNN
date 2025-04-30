from chess_gnn.tokenizers.base import ChessTokenizer
from networkx import DiGraph

from chess_gnn.utils.pgn_utils import process_board_string
from .base import GraphCreationStrategy, GraphCreationKwargs
from .utils import get_adjacency_edges, get_legal_move_edges, merge_edges, LEGAL, ADJACENT


class LegalMoveGraphCreationStrategy(GraphCreationStrategy):
    def __init__(self, tokenizer: ChessTokenizer):
        self.tokenizer = tokenizer

    def create_graph(self, args: GraphCreationKwargs) -> DiGraph:
        g = DiGraph()
        board_str = process_board_string(str(args.board))
        nodes = []
        for i, (token, feature) in enumerate(zip(board_str, args.board_features)):
            nodes.append((i, {"token": token, "feature": feature}))

        g.add_nodes_from(nodes)

        adjacent = []
        for i in range(len(board_str)):
            adjacent_edges = get_adjacency_edges(idx=i)
            adjacent.extend(adjacent_edges)

        legal_edges = get_legal_move_edges(args.board)
        edge_set = merge_edges(adjacent, legal_edges)

        g.add_edges_from(*edge_set['legal'], edge_type=LEGAL)
        g.add_edges_from(*edge_set['adjacent'], edge_type=ADJACENT)

        return g


