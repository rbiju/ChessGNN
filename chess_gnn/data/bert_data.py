from base import BaseBERTDataset


class WinPredictionBERTDataset(BaseBERTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs['file'])

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
