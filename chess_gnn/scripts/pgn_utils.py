from pathlib import Path
from chess_gnn.utils.pgn_utils import LichessChessBoardGetter
from chess_gnn.utils import ChessPoint
from chess_gnn.data_creation.bert import BERTLichessDatasetCreator


def get_chess_board():
    file = '/Users/ray/Datasets/lichess_db_standard_rated_2013-01.pgn'
    board_getter = LichessChessBoardGetter(Path(file))
    board_getter.get_game()
    game = board_getter.game()
    board = game.board()

    print(str(board).replace('\n', ' ').replace(" ", ""))

    return board


def create_adjacent_edges():
    point = ChessPoint.from_1d(9)
    bottom_left = point.down().left().clip()
    top_right = point.up().right().clip()
    adjacent_edges = bottom_left.range(top_right)

    return adjacent_edges


def write_dataset_to_txt():
    dataset_creator = BERTLichessDatasetCreator(pgn_file=Path('/Users/ray/Datasets/test.pgn'),
                                                data_directory=Path('/Users/ray/Datasets/txt/test'))

    dataset_creator.create_dataset()

if __name__ == "__main__":
    write_dataset_to_txt()
