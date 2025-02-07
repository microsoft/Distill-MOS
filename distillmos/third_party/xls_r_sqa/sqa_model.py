import torch
from torch import nn, Tensor
from typing import Tuple

from .config import Config
from .pool_att_ff import PoolAttFF
from .transformer_wrapper import TransformerWrapper

class SingleLayerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.norm_input = nn.BatchNorm1d(config.dim_input)
        self.transformer = TransformerWrapper(config)
        self.norm_trans = nn.BatchNorm1d(config.dim_transformer)

        self.pool = PoolAttFF(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: Tensor):
        r"""Predict perceived speech quality in [0,1] for the given sequence of
        features.

        Args:
            features: sequence of XLS-R (or MFCC) features.

        Shape:
            - features: (N,L,C) or (L,C)
            - output: (N,1) or (1,)
        """

        unsqueezed = False
        if features.dim() == 2:
            features = features.unsqueeze(0) # Include batch dim
            unsqueezed = True

        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm_input(features.transpose(-1,-2)).transpose(-1,-2)
        x = self.transformer(x)
        x = self.norm_trans(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.pool(x).squeeze(-1)
        x = self.sigmoid(x)

        if unsqueezed:
            x = x.squeeze(0)

        return x


class FusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Batch norm for each XLS-R layer.
        _norm_inputs = [nn.BatchNorm1d(config.dim_input) for _ in range(2)]
        self.norm_inputs = nn.ModuleList(_norm_inputs)
        
        self.lin_infusion = nn.Linear(2, 1)
        self.transformer = TransformerWrapper(config)
        self.norm_trans = nn.BatchNorm1d(config.dim_transformer)

        self.pool = PoolAttFF(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: Tuple[Tensor, Tensor]):
        r"""Predict perceived speech quality in [0,1] for the given sequence of
        features.

        Args:
            features: tuple containing two items, each being a sequence of XLS-R
            features extracted from a certain layer.

        Shape:
            - features: tuple containing two Tensors of shape (N,L,C) or (L,C)
            - output: (N,1) or (1,)
        """

        unsqueezed = False
        if features[0].dim() == 2:
            features = tuple(x.unsqueeze(0) for x in features) # Include batch dim
            unsqueezed = True

        # Batch norm for each XLS-R layer.
        # Transform from (N, L, C) to (N, C, L) and back.
        features = tuple(
            self.norm_inputs[i].forward(features[i].transpose(-1,-2)).transpose(-1,-2)
            for i in range(2)
        )

        x = torch.stack(features, dim=-1)
        x = self.lin_infusion(x).squeeze(-1)

        x = self.transformer(x)
        x = self.norm_trans(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.pool(x).squeeze(-1)
        x = self.sigmoid(x)

        if unsqueezed:
            x = x.squeeze(0)

        return x
