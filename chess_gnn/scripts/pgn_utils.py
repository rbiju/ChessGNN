from pathlib import Path
import time

import torch

from converter.pgn_data import PGNData

from chess_gnn.tokenizers import SimpleChessTokenizer
from chess_gnn.utils import ChessPoint
from chess_gnn.data_creation.bert import BERTLichessDatasetCreator


def create_adjacent_edges():
    point = ChessPoint.from_1d(9)
    bottom_left = point.down().left().clip()
    top_right = point.up().right().clip()
    adjacent_edges = bottom_left.range(top_right)

    return adjacent_edges


def write_pgn_to_txt():
    dataset_creator = BERTLichessDatasetCreator(pgn_file=Path('/Users/ray/Datasets/chess/test.pgn'),
                                                data_directory=Path('/Users/ray/Datasets/txt/test'))

    dataset_creator.create_dataset()


def generate_df():
    data = PGNData('/Users/ray/Datasets/test.pgn',
                   file_name='/Users/ray/Datasets/test')
    result = data.export()

    games_df = result.get_games_df()

    return games_df


def test_tokenizer():
    tokenizer = SimpleChessTokenizer()

    tokens = tokenizer.tokenize('r..q.rk.pp..bppp.np.bp.........Q...P.....BP.BN..PP...PPPR...K..R')

    embeddings = torch.nn.Parameter(torch.rand(tokenizer.vocab_size, 32))

    seq = torch.index_select(input=embeddings, dim=0, index=tokens)

    return seq


if __name__ == "__main__":
    start = time.time()
    write_pgn_to_txt()
    end = time.time()

    print(f"Dataset creation time: {end - start}")
