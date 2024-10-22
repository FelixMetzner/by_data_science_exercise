#!/usr/bin/env python

import torch
import torch.nn as nn

__all__ = [
    "MeanModel",
]


class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(
            data=torch.randn(1),
            requires_grad=True,
        )

        print(f"Model initialized with random mean: {self.mean}")

    def forward(self, n: int) -> torch.Tensor:
        return self.mean * torch.ones(n)
