import torch.nn as nn
import math


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

    def forward(self, input_):
        return self.embedding(input_) * math.sqrt(self.d_model)
