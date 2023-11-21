"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.multi_head_attention import ProbMultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
import torch

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention= ProbMultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x,  src_mask, prior_attn = None , pprior_attn =  None):
        # 1. compute self attention
        _x = x
        if prior_attn == None:
            x, attn , p_attn = self.attention(aq = x , ak = x , av = x ,mask=src_mask)
        else:
             x, attn , p_attn = self.attention(aq = x , ak = x , av = x ,mask=src_mask,prior_attn = prior_attn , pprior_attn = pprior_attn)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, attn , p_attn

