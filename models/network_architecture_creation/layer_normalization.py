from torch import nn
import torch


class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x.dim = (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)  # dim ->  (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # dim -> (batch, seq_len, 1)

        output = (x - mean) / (std * self.eps)
        output = self.alpha * output + self.bias
        return output # output.dim = (batch, seq_len, d_model)





