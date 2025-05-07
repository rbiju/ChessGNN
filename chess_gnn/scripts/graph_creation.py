from chess import Board
import torch
import matplotlib.pyplot as plt
import networkx as nx

from chess_gnn.graph_creation import LegalMoveGraphCreationStrategy, GraphCreationKwargs, \
    FullyConnectedGraphCreationStrategy


def test_graph_creation():
    board = Board(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w K - 0 1')
    board_features = torch.rand(64, 13)

    graph_creation_kwargs = GraphCreationKwargs(board_features=board_features, board=board)

    strategy = LegalMoveGraphCreationStrategy()
    graph, pos = strategy.create_graph(args=graph_creation_kwargs)

    nx.draw(graph, pos)
    plt.show()

    return graph


def fc_graph_creation():
    board = Board()
    board_features = torch.rand(64, 13)

    graph_creation_kwargs = GraphCreationKwargs(board_features=board_features, board=board)
    strategy = FullyConnectedGraphCreationStrategy()
    graph, pos = strategy.create_graph(args=graph_creation_kwargs)

    nx.draw(graph, pos)
    plt.show()

    return graph


if __name__ == '__main__':
    fc_graph_creation()
