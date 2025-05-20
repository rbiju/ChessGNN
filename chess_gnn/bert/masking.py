import einops
import torch
import torch.nn as nn


class BERTMaskHandler(nn.Module):
    def __init__(
            self,
            mask_token_id: int = 13,
            vocab_size: int = 13,
            mask_prob: float = 0.15,
            ignore_index: int = -100,
    ):
        """
        Args:
            mask_token_id: ID of the [MASK] token
            vocab_size: size of vocabulary (used for random token replacement)
            mask_prob: probability of masking a given token
            ignore_index: label to ignore in loss
        """
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T] input token IDs

        Returns:
            masked_input_ids: [B, T] with some tokens replaced
            labels: [B, T] with original token IDs at masked positions, else ignore_index
        """
        device = x.device
        labels = x.clone()

        rand = torch.rand(x.shape, device=device)
        masked_indices = rand < self.mask_prob

        labels[~masked_indices] = self.ignore_index  # only predict masked positions

        # Apply BERT masking strategy
        # 80% [MASK], 10% random token, 10% unchanged
        masked_input_ids = x.clone()

        mask_mask = masked_indices & (rand < self.mask_prob * 0.8)
        random_mask = masked_indices & (rand >= self.mask_prob * 0.8) & (rand < self.mask_prob * 0.9)
        # remaining 10% of masked_indices are left unchanged

        masked_input_ids[mask_mask] = self.mask_token_id

        random_tokens = torch.randint(low=0, high=self.vocab_size, size=x.shape, device=device)
        masked_input_ids[random_mask] = random_tokens[random_mask]

        return masked_input_ids, labels


class ElectraMaskHandler(nn.Module):
    def __init__(
            self,
            mask_token_id: int = 13,
            vocab_size: int = 13,
            mask_prob: float = 0.15,
            ignore_index: int = -100,
    ):
        """
        Args:
            mask_token_id: ID of the [MASK] token
            vocab_size: size of vocabulary (used for random token replacement)
            mask_prob: probability of masking a given token
            ignore_index: label to ignore in loss
        """
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T] input token IDs

        Returns:
            masked_input_ids: [B, T] with some tokens replaced
            labels: [B, T] with original token IDs at masked positions, else ignore_index
        """
        device = x.device
        labels = x.clone()

        rand = torch.rand(x.shape, device=device)
        masked_indices = rand < self.mask_prob

        masked_input_ids = x.clone()

        mask_mask = masked_indices & (rand < self.mask_prob)

        masked_input_ids[mask_mask] = self.mask_token_id

        labels[~masked_indices] = self.ignore_index

        return masked_input_ids, masked_indices, labels


class TransformerMaskHandler(nn.Module):
    def __init__(self, masking_ratio: float, mask_token_id: int = 13):
        """
        Args:
            masking_ratio (float): Ratio of tokens to mask per sequence (0 < r < 1).
            mask_token_id (int): Token ID to use for [MASK].
        """
        super().__init__()
        self.masking_ratio = masking_ratio
        self.mask_token_id = mask_token_id

    @staticmethod
    def get_noise(x: torch.Tensor):
        B, L = x.shape
        noise = torch.rand(B, L, device=x.device)

        return noise

    def get_mask(self, x: torch.Tensor):
        B, L = x.shape
        len_keep = int(L * (1 - self.masking_ratio))

        noise = self.get_noise(x)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=-1)
        ids_restore = torch.argsort(ids_shuffle, dim=-1)

        return ids_shuffle, ids_restore, len_keep

    def shuffle_and_mask(self, x: torch.Tensor, ids_shuffle: torch.Tensor, ids_restore: torch.Tensor, len_keep: int):
        x_shuffled = torch.gather(x, dim=1, index=ids_shuffle)
        x_shuffled[..., len_keep:] = self.mask_token_id
        shuffled_with_mask = torch.gather(x_shuffled, dim=1, index=ids_restore)

        return shuffled_with_mask

    @staticmethod
    def get_unmasked_tokens(x: torch.Tensor, ids_keep: torch.Tensor):
        unmasked = torch.gather(x, dim=1, index=ids_keep)
        return unmasked

    @staticmethod
    def get_masked_tokens(x: torch.Tensor, ids_mask: torch.Tensor):
        masked = torch.gather(x, dim=1, index=ids_mask)
        return masked

    @staticmethod
    def get_unmasked_embeddings(x: torch.Tensor, ids_keep: torch.Tensor):
        _, _, E = x.shape
        ids_keep = einops.repeat(ids_keep, 'b l -> b l e', e=E)

        return torch.gather(x, dim=1, index=ids_keep)

    @staticmethod
    def get_masked_embeddings(x: torch.Tensor, ids_mask: torch.Tensor):
        _, _, E = x.shape
        ids_mask = einops.repeat(ids_mask, 'b l -> b l e', e=E)

        return torch.gather(x, dim=1, index=ids_mask)
