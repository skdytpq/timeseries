"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder
from layers.multi_head_attention import MultiHeadAttention


class Transformer(nn.Module):

    def __init__(self,  x_size, y_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.current_time = x_size
        self.tiemsteps = y_size
        self.lstm_layers = 2
        self.lstm_hidden_dim = 40
        self.LSTM_in = nn.LSTM(input_size = x_size, hidden_size =d_model)
        self.LSTM_out = nn.LSTM(input_size = y_size, hidden_size =d_model)
        self.finaldense_x = nn.Linear(x_size)
        self.finaldense_y = nn.Linear(y_size)
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               x_size=x_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               y_size=y_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(x_size * self.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(x_size * self.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()
        self.distribution_mu_out = nn.Linear(y_size * self.lstm_layers, 1)
        self.distribution_presigma_out = nn.Linear(y_size * self.lstm_layers, 1)
        self.distribution_sigma_out = nn.Softplus()
    def forward(self, src, trg,hidden,cell):
        src_mask = self.make_src_mask(src) # src
        trg_mask = self.get_attn_decoder_mask(trg) # decoder masking
        enc_src,attn , p_attn = self.encoder(src, src_mask)
        # enc_src is X_{1:C} 여기에 encoder 를 통과한 것과 decoder를 통과한 것에 대한 Loss 구하기 Decoder 내부에서 나온 결과 값을 넣어야함
        dec_src = self.decoder(trg, enc_src, trg_mask, src_mask)
        # 여기서의 output 은 encoder decoder의 결합 output이라고 할 수 있음.
        # 여기서 h 는 cur time 인 64 까지이기 때문에 w 는 이후 128까지 할 수 있게 만들어야 함
        output_in, (hidden, cell) = self.LSTM_in(enc_src, (hidden, cell))
        output_out, (hidden, cell) = self.LSTM_out(dec_src[-1:,:,:], (hidden, cell))
        # (batch_size, num_layers, timesteps, 1) # 여기서의 예측값은 미래시점에 대한 예측값
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        x_hat = self.finaldense_x(output_in)
        sigma = self.distribution_sigma(pre_sigma)
        pre_sigma_dec = self.distribution_presigma_out(hidden_permute)
        mu_dec = self.distribution_mu_out(hidden_permute)
        sigma_dec = self.distribution_sigma_out(pre_sigma_dec)
        y_hat = self.finaldense_y(output_out)
        return attn, p_attn ,x_hat,mu,sigma ,y_hat,mu_dec , sigma_dec#, gen_z, gen_mean, gen_var, inf_z, inf_mean, inf_var

    def make_src_mask(self, src):
        src_mask = (src != 'BOOM!!').unsqueeze(1).unsqueeze(2) # S
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    def get_attn_decoder_mask(self,seq):
        subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
        subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
        return subsequent_mask
    def init_hidden(self, input_size):
        return torch.zeros(self.lstm_layers, input_size, self.lstm_hidden_dim, device=self.device)

    def init_cell(self, input_size):
        return torch.zeros(self.lstm_layers, input_size, self.lstm_hidden_dim, device=self.device)