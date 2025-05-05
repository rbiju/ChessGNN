from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

from chess_gnn.tokenizers import ChessTokenizer, SimpleChessTokenizer


class HDF5DatasetBuilder:
    def __init__(self,
                 chunk_size: int,
                 tokenizer: ChessTokenizer = SimpleChessTokenizer(),
                 max_len: int = 64):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def write_dataset(self, input_file: Path):
        input_file = Path(input_file)
        output_path = input_file.parent / "data.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # First, count the total number of lines (samples) in the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)

        with h5py.File(output_path, 'w') as h5f:
            # Create the 'board' and 'label' datasets with predetermined size
            board_ds = h5f.create_dataset(
                'board', shape=(total_samples, self.max_len), dtype='i8', chunks=(self.chunk_size, self.max_len)
            )
            label_ds = h5f.create_dataset(
                'label', shape=(total_samples,), dtype='f4', chunks=(self.chunk_size,)
            )

            total = 0
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_samples, desc=f"Writing {output_path.parent.name}"):
                    line = line.strip()
                    if not line:
                        continue

                    board_str = line[:-1]
                    label = int(line[-1])

                    # Tokenize the board and write to the dataset
                    board_tensor = np.array(self.tokenizer.tokenize(board_str))
                    label_tensor = float(label)

                    # Write directly to the dataset without resizing
                    board_ds[total] = board_tensor
                    label_ds[total] = label_tensor

                    total += 1

            # Store total length as a metadata attribute
            h5f.attrs['total_length'] = total
