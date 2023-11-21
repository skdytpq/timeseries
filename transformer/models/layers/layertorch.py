import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# MultiheadAttention의 경우 다른 모듈에서 가져오기.
from multi_head_attention import MultiHeadAttention
from scale_dot_product_attention import ScaleDotProductAttention
# 해당 부분이 디코더 연산에 들어가야함
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = self.positional_encoding(max_seq_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

    def positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.size(-1)
        scaled_attention_logits = matmul_qk / np.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
class AddPosition(nn.Module):
    def __init__(self, d_model, timesteps):
        super(AddPosition, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.posit_matrix = PositionalEncoding(d_model)

    def forward(self, x, t):
        return self.layer_norm(x + self.posit_matrix(torch.zeros(1, t + 1, self.posit_matrix.d_model)))

class GenLayer(nn.Module):
    def __init__(self, d_model, d_latent, timesteps, num_heads):
        super(GenLayer, self).__init__()
        self.d_model = d_model
        self.timesteps = timesteps

        self.w0 = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        self.dense1 = nn.Linear(d_latent, d_latent)
        self.dense2 = nn.Linear(d_latent, d_latent)
        self.dense3 = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.add_posit = AddPosition(d_model, timesteps)

    def forward(self, h_C, prior_W=None):
        batch_size = h_C.size(0)
        w = self.w0.repeat(batch_size, 1, 1)

        z_list = []
        w_hat_list = []
        mean_list = []
        var_list = []

        for i in range(self.timesteps):
            if prior_W is None:
                tmp_w_bar = self.layer_norm1(w[:, :, i:i+1] + self.mha1(w[:, :, i:i+1], w[:, :, :i+1], w[:, :, :i+1]))
            else:
                tmp_w_tilde = self.layer_norm3(w[:, :, i:i+1] + self.mha3(w[:, :, i:i+1], prior_W, prior_W))
                tmp_w_bar = self.layer_norm1(tmp_w_tilde + self.mha1(tmp_w_tilde, w[:, :, :i+1], w[:, :, :i+1]))

            tmp_w_hat = self.layer_norm2(tmp_w_bar + self.mha2(tmp_w_bar, h_C, h_C))
            tmp_mean = self.dense1(tmp_w_hat)
            tmp_eps = torch.normal(0, torch.exp(self.dense2(tmp_w_hat)))
            tmp_z = tmp_mean + tmp_eps
            tmp_w = self.add_posit(tmp_w_hat + self.dense3(tmp_z), i)

            w = torch.cat((w, tmp_w), dim=2)

            w_hat_list.append(tmp_w_hat)
            z_list.append(tmp_z)
            mean_list.append(tmp_mean)
            var_list.append(torch.exp(self.dense2(tmp_w_hat)))

        z = torch.cat(z_list, dim=2)
        w_hat = torch.cat(w_hat_list, dim=2)
        mean = torch.cat(mean_list, dim=2)
        var = torch.cat(var_list, dim=2)

        return w[:, :, 1:], z, w_hat, mean, var

class InfLayer(nn.Module):
    def __init__(self, d_model, d_latent, num_heads):
        super(InfLayer, self).__init()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.dense1 = nn.Linear(d_latent, d_latent)
        self.dense2 = nn.Linear(d_latent, d_latent)

    def forward(self, h_T, w_hat):
        k = self.mha1(h_T, h_T, h_T)
        hw = torch.cat((w_hat, k), dim=-1)
        mean = self.dense1(hw)
        eps = torch.normal(0, torch.exp(self.dense2(hw)))
        z = mean + eps
        return z, mean,