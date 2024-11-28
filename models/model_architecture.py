import torch
import torch.nn as nn


class NMTModel(nn.Module):
    def __init__(self, en_vocab_size, fr_vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.1):
        """
        Initializes the NMT Model with a custom Transformer encoder and decoder.

        :param vocab_size: Size of the vocabulary.
        :param embed_size: Dimension of the embedding vectors.
        :param hidden_size: Hidden size for the encoder and decoder.
        :param num_layers: Number of Transformer layers in encoder and decoder.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout rate.
        """
        super(NMTModel, self).__init__()

        # Embedding layers for both source and target
        self.src_embedding = nn.Embedding(en_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(fr_vocab_size, embed_size)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(embed_size, max_len=512), requires_grad=False)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output layer
        self.fc_out = nn.Linear(hidden_size, fr_vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _get_positional_encoding(d_model, max_len):
        """
        Generates a positional encoding matrix.

        :param d_model: Dimension of the embeddings.
        :param max_len: Maximum sequence length.
        :return: Positional encoding matrix.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask=None):
        """
        Defines the forward pass for the model.

        :param src_input_ids: Source input IDs.
        :param src_attention_mask: Source attention mask.
        :param tgt_input_ids: Target input IDs.
        :param tgt_attention_mask: Target attention mask.
        :return: Output logits for predictions.
        """

        src_embeddings = self.src_embedding(src_input_ids) + self.positional_encoding[:, :src_input_ids.size(1), :]
        src_embeddings = self.dropout(src_embeddings)

        # Transformer Encoder Forward Pass
        memory = self.encoder(
            src=src_embeddings.permute(1, 0, 2),  # (src_len, batch_size, embed_size)
            src_key_padding_mask=~src_attention_mask.bool()  # Mask padded tokens
        )

        # Embed target inputs and add positional encoding
        tgt_embeddings = self.tgt_embedding(tgt_input_ids) + self.positional_encoding[:, :tgt_input_ids.size(1), :]
        tgt_embeddings = self.dropout(tgt_embeddings)

        # Prepare target mask for causal decoding
        tgt_seq_len = tgt_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input_ids.device)

        decoder_outputs = self.decoder(
            tgt=tgt_embeddings.permute(1, 0, 2),  # (tgt_len, batch_size, embed_size)
            memory=memory,                        # (src_len, batch_size, hidden_size)
            tgt_mask=tgt_mask,                    # Mask for causal decoding
            memory_key_padding_mask=~src_attention_mask.bool()  # Cross-attention mask
        )

        output = self.fc_out(decoder_outputs.permute(1, 0, 2))  # (batch_size, tgt_len, vocab_size)
        return output
