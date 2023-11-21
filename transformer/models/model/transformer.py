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
        self.finaldense_x = nn.Linear(x_size)
        self.finaldense_y = nn.Linear(y_size)
        self.gen_decoder = nn.ModuleList([GenLayer(d_model, d_model, max_len, n_head) for _ in range(n_layers)])
        self.inf_encoder = nn.ModuleList([InfLayer(d_model, d_model, n_head) for _ in range(n_layers)])
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

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) # src
        trg_mask = self.get_attn_decoder_mask(trg) # decoder masking
        enc_src,attn , p_attn = self.encoder(src, src_mask)
        # enc_src is X_{1:C} 여기에 encoder 를 통과한 것과 decoder를 통과한 것에 대한 Loss 구하기 Decoder 내부에서 나온 결과 값을 넣어야함
        dec_src = self.decoder(trg, enc_src, trg_mask, src_mask)
        # 여기서의 output 은 encoder decoder의 결합 output이라고 할 수 있음.
        h = enc_src # 여기서 h 는 cur time 인 64 까지이기 때문에 w 는 이후 128까지 할 수 있게 만들어야 함
        gen_w_list = []        
        gen_z_list = []
        gen_w_hat_list = []                    
        gen_mean_list = []
        gen_var_list = []
                    
        for i in range(self.num_layers):
            if i == 0 :
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :, :])
            else:
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :, :], tmp_gen_w)
            
            gen_w_list.append(tmp_gen_w.unsqueeze(1)) # (batch_size, 1, timesteps, d_model)
            gen_z_list.append(tmp_gen_z.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            gen_w_hat_list.append(tmp_gen_w_hat.unsqueeze(1)) # (batch_size, 1, timesteps, d_model)
            gen_mean_list.append(tmp_gen_mean.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            gen_var_list.append(tmp_gen_var.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            
        gen_w = torch.cat(gen_w_list, dim=1) # (batch_size, num_layers, timesteps, d_model)
        gen_z = torch.cat(gen_z_list, dim=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_w_hat = torch.cat(gen_w_hat_list, dim=1) # (batch_size, num_layers, timesteps, d_model)
        gen_mean = torch.cat(gen_mean_list, dim=1) # (batch_size, num_layers, timesteps, d_latent)
        gen_var = torch.cat(gen_var_list, dim=1) # (batch_size, num_layers, timesteps, d_latent)
        
        x_hat = self.final_dense(gen_w) # (batch_size, num_layers, timesteps, 1) # 여기서의 예측값은 미래시점에 대한 예측값
        
        inf_z_list = []
        inf_mean_list = []
        inf_var_list = []
        
        for i in range(self.num_layers):
            tmp_inf_z, tmp_inf_mean, tmp_inf_var = self.inf_encoder[i](dec_src, gen_w_hat[:, i, :]) # dec_src의 경우 미래 시점까지의 정보를 담고있는 벡터
            inf_z_list.append(tmp_inf_z.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            inf_mean_list.append(tmp_inf_mean.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            inf_var_list.append(tmp_inf_var.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
        
        inf_z = torch.cat(inf_z_list, dim=1)
        inf_mean = torch.cat(inf_mean_list, dim=1)
        inf_var = torch.cat(inf_var_list, dim=1)
        y_hat = inf_z
        return attn, p_attn ,x_hat, gen_z, gen_mean, gen_var, inf_z, inf_mean, inf_var

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

    def forward(self, h_C, prior_W=None):
        batch_size = h_C.size(0)  # 24 * 4 * 2
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
            tmp_w = tmp_w_hat + self.dense3(tmp_z)

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
    def __init__(self, d_model, d_latent, num_heads): # Dec src
        super(InfLayer, self).__init()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.dense1 = nn.Linear(d_latent, d_latent)
        self.dense2 = nn.Linear(d_latent, d_latent)

    def forward(self, h_T, w_hat):
        k = self.mha1(h_T, h_T, h_T)
        hw = torch.cat((w_hat, k), dim=-1)
        mean = self.dense1(hw)
        eps = torch.normal(0, nn.Softplus(self.dense2(hw)))
        z = mean + eps
        return z, mean,nn.Softplus(self.dense2(hw))
