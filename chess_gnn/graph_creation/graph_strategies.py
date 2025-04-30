from chess_gnn.tokenizers.base import ChessTokenizer
from networkx import DiGraph

from chess_gnn.utils.pgn_utils import process_board_string
from .base import GraphCreationStrategy, GraphCreationKwargs
from .utils import get_edge_set_single, LEGAL, ADJACENT


class LegalMoveGraphCreationStrategy(GraphCreationStrategy):
    def __init__(self, tokenizer: ChessTokenizer):
        self.tokenizer = tokenizer

    def create_graph(self, args: GraphCreationKwargs) -> DiGraph:
        g = DiGraph()
        board_str = process_board_string(args.board)
        nodes = []
        for i, (token, feature) in enumerate(zip(board_str, args.board_features)):
            nodes.append((i, {"token": token, "feature": feature}))

        g.add_nodes_from(nodes)

        for i in range(len(board_str)):
            edges = get_edge_set_single(i, board_str)
            g.add_edges_from(*edges['legal'], edge_type=LEGAL)
            g.add_edges_from(*edges['adjacent'], edge_type=ADJACENT)

        return g
