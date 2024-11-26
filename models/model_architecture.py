import torch
import torch.nn as nn
from transformers import BertModel


class NMTModel(nn.Module):
    def __init__(self, encoder_model_name, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(NMTModel, self).__init__()

        self.encoder = BertModel.from_pretrained(encoder_model_name)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask=None):
        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        memory = encoder_outputs.last_hidden_state  # Shape: (batch_size, src_len, hidden_size)

        # Decoder Embeddings
        tgt_embeddings = self.embedding(tgt_input_ids)  # Shape: (batch_size, tgt_len, embed_size)
        tgt_embeddings = self.dropout(tgt_embeddings)

        # Prepare masks for the decoder
        tgt_seq_len = tgt_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input_ids.device)

        # Decoder Forward Pass
        decoder_outputs = self.decoder(
            tgt=tgt_embeddings.permute(1, 0, 2),  # (tgt_len, batch_size, embed_size)
            memory=memory.permute(1, 0, 2),       # (src_len, batch_size, hidden_size)
            tgt_mask=tgt_mask,                   # Target mask for causal decoding
            memory_key_padding_mask=~src_attention_mask.bool()  # Cross-attention mask
        )

        # Final Linear Layer
        output = self.fc_out(decoder_outputs.permute(1, 0, 2))  # Shape: (batch_size, tgt_len, vocab_size)
        return output
