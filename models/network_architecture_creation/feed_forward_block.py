from torch import nn
import torch


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.transform(x)
