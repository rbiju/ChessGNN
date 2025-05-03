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
            x: [B, T]

        Returns:
            masked_input_ids: [B, T] tensor with masking applied
            labels: [B, T] original token IDs at masked positions, rest = ignore_index
        """
        device = x.device
        labels = x.clone()

        probability_matrix = torch.full(x.shape, self.mask_prob, device=device)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = self.ignore_index

        masked_input_ids = x.clone()

        mask_mask = torch.bernoulli(torch.full(x.shape, 0.8, device=device)).bool() & masked_indices
        masked_input_ids[mask_mask] = self.mask_token_id

        random_mask = torch.bernoulli(torch.full(x.shape, 0.5, device=device)).bool() & masked_indices & ~mask_mask
        random_tokens = torch.randint(low=0, high=self.vocab_size, size=x.shape, device=device)
        masked_input_ids[random_mask] = random_tokens[random_mask]

        return masked_input_ids, labels
