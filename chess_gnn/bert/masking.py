from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from einops import repeat
import torch
import torch.nn as nn


@dataclass
class BERTProbabilities:
    mask: float
    random: float
    unchanged: float

    def validate(self):
        if self.mask + self.random + self.unchanged != 1.0:
            raise ValueError("BERT probabilities must add to 1")


class BaseMaskHandler(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mask_sequence(self, x: torch.Tensor, ) -> torch.Tensor:
        raise NotImplementedError


class BERTMaskHandler(nn.Module):
    """
    Args:
        mask_ratio: Proportion of sequence to mask
    """

    def __init__(
            self, mask_ratio: float,
            bert_probabilities: BERTProbabilities,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.bert_probabilities = bert_probabilities

    @staticmethod
    def get_noise(x: torch.Tensor):
        B, L, E = x.shape
        noise = torch.rand(B, L, device=x.device)

        return noise

    def get_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, E = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        noise = self.get_noise(x)

        ids_shuffle = torch.argsort(noise, dim=-1)
        ids_restore = torch.argsort(ids_shuffle, dim=-1)

        ids_keep = ids_shuffle[..., :len_keep]
        ids_keep = repeat(ids_keep, 'b l -> b l e', e=E)

        ids_mask = ids_shuffle[..., len_keep:]
        ids_mask = repeat(ids_mask, 'b l -> b l e', e=E)

        # generate the mask: 0 is keep, 1 is mask
        mask = torch.ones([B, L], device=x.device)
        mask[..., :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_keep, ids_restore, ids_mask

    def mask_sequence(self, x: torch.Tensor, vocab_size: int, mask_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, E = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        len_mask = int((L - len_keep) * self.bert_probabilities.mask)
        len_random = int((L - len_keep) * self.bert_probabilities.random)
        len_unchanged = int((L - len_keep) * self.bert_probabilities.unchanged)

        mask, ids_keep, ids_restore, ids_mask = self.get_mask(x)

        x_visible = torch.gather(x, dim=1, index=ids_keep)
        x_hidden = torch.gather(x, dim=1, index=ids_mask)

        mask_tensor = repeat(mask_token, 'e -> b l e', b=B, l=len_mask)
        random_tensor = torch.nn.functional.one_hot(torch.randint(low=0,
                                                                  high=vocab_size,
                                                                  size=(len_random,),
                                                                  device=x.device))
        unchanged_tensor = x_hidden[..., -len_unchanged:, :]

        x = torch.cat([x_visible, mask_tensor, random_tensor, unchanged_tensor], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore)

        return x, ids_mask

    @staticmethod
    def mask_loss(prediction: torch.Tensor, target: torch.Tensor, ids_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
             prediction: shape: [b L vocab_size]
             target:
             ids_mask:
        """

        prediction = torch.gather(prediction, dim=1, index=ids_mask)
        target = torch.gather(target, dim=1, index=ids_mask)

        return {'prediction': prediction,
                'target': target}
