"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding
import torch

class Encoder(nn.Module):

    def __init__(self, x_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, device,max_len = 128):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        d_size=x_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x) # Batch X Len
        x , attn, p_attn = layer(x,src_mask) 
        for layer in self.layers - 1:
            x , attn, p_attn = layer(x,src_mask,prior_attn = attn , pprior_attn = p_attn) # 인풋으로 attn, p_attn 이 들어가야 Loss 가 구해지나? 아니면 중첩 곱에 Loss 가 계산되나?

        # 여기서 로스 계산? NLL Loss 를 QXK에서 구해야함
        return x , attn , p_attn