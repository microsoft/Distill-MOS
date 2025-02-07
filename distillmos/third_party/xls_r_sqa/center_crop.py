import torch
from torch.nn.functional import pad


class CenterCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor):
        # Center crop.
        unsqueezed = False
        if x.dim() == 2:
            unsqueezed = True
            x = x.unsqueeze(0)
        assert x.dim() == 3 # N, L, C

        if x.size(1) > self.seq_len:
            center_start_idx = int(x.size(1) / 2 - self.seq_len / 2)
            start_idx = center_start_idx
            end_idx = start_idx + self.seq_len
            x = x[:, start_idx:end_idx, :]
        if x.size(1) < self.seq_len:
            to_pad = self.seq_len - x.size(1)
            # Pad the end of sequence dimension.
            x = pad(x, (0,0,0,to_pad,0,0), mode="constant", value=0.0)

        if unsqueezed:
            x = x.squeeze(0)

        return x
