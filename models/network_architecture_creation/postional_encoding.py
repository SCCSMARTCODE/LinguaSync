import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_length, d_model)

        position = torch.arange(start=0, end=seq_length, step=1).unsqueeze(dim=1)   # dim -> (seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::1] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(name='pe', tensor=pe.unsqueeze(dim=0))

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)

