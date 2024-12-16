import math
from torch import nn


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()

        assert d_model % num_heads == 0, f"d_model [{d_model}] must be divisible by num_heads [{num_heads}]"

        self.d_model = d_model
        self.num_heads = num_heads
        self.h_d_model = d_model // num_heads

        self.query_t = nn.Linear(d_model, d_model, bias=False)
        self.ket_t = nn.Linear(d_model, d_model, bias=False)
        self.value_t = nn.Linear(d_model, d_model, bias=False)
        self.out_t = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.query_t(q)  # (batch, seq_length, d_model)
        key = self.ket_t(k)  # (batch, seq_length, d_model)
        value = self.value_t(v)  # (batch, seq_length, d_model)

        query = query.reshape(query.shape[0], query.shape[1], self.num_heads, self.h_d_model).transpose(1,
                                                                                                        2)  # (batch, num_heads, seq_length, h_d_model)
        key = key.reshape(key.shape[0], key.shape[1], self.num_heads, self.h_d_model).transpose(1,
                                                                                                2)  # (batch, num_heads, seq_length, h_d_model)
        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, self.h_d_model).transpose(1,
                                                                                                        2)  # (batch, num_heads, seq_length, h_d_model)

        x, _ = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.h_d_model)
        return self.out_t(x)

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(self.h_d_model)

        #  Note: pay attention to the mask creation
        if mask:
            attention_scores.masked_fill_(mask, float('-inf'))
        if dropout:
            attention_scores = dropout(attention_scores)
        attention_scores = attention_scores.softmax(dim=-1)
        return attention_scores @ value, attention_scores
