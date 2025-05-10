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
