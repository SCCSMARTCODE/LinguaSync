from torch import nn
from embedding import Embedding as InputEmbeddings
from models.network_architecture_creation.attention import MultiHeadAttentionBlock
from models.network_architecture_creation.decoder import DecoderBlock, Decoder
from models.network_architecture_creation.encoder import EncoderBlock, Encoder
from models.network_architecture_creation.feed_forward_block import FeedForwardBlock
from models.network_architecture_creation.postional_encoding import PositionalEncoding
from models.network_architecture_creation.projection import ProjectionLayer
from models.network_architecture_creation.transformer import Transformer


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, n: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block_ = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block_, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block_ = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block_, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder_ = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder_ = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer_ = Transformer(encoder_, decoder_, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer_.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_
