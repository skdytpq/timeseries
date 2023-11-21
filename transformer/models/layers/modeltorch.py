import torch
import torch.nn as nn
import torch.nn.functional as F
from layertorch import *
from multi_head_attention import MultiHeadAttention
from scale_dot_product_attention import ScaleDotProductAttention
class ProTran(nn.Module):
    def __init__(self, d_output, d_model, d_latent, timesteps, current_time, num_heads, num_layers):
        super(ProTran, self).__init__()   
        self.d_model = d_model
        self.timesteps = timesteps
        self.current_time = current_time
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.dense = nn.Linear(d_model)
        self.add_posit = AddPosition2(d_model, timesteps) 
        
        self.gen_decoder = nn.ModuleList([GenLayer(d_model, d_latent, timesteps, num_heads) for _ in range(num_layers)])
        self.inf_encoder = nn.ModuleList([InfLayer(d_model, d_latent, num_heads) for _ in range(num_layers)])
        
        self.final_dense = nn.Linear(d_output)
    
    def forward(self, x):
        h = x
        gen_w_list = []        
        gen_z_list = []
        gen_w_hat_list = []                    
        gen_mean_list = []
        gen_var_list = []
                    
        for i in range(self.num_layers):
            if i == 0 :
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :])
            else:
                tmp_gen_w, tmp_gen_z, tmp_gen_w_hat, tmp_gen_mean, tmp_gen_var = self.gen_decoder[i](h[:, :self.current_time, :], tmp_gen_w)
            
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
        
        x_hat = self.final_dense(gen_w) # (batch_size, num_layers, timesteps, 1)
        
        inf_z_list = []
        inf_mean_list = []
        inf_var_list = []
        
        for i in range(self.num_layers):
            tmp_inf_z, tmp_inf_mean, tmp_inf_var = self.inf_encoder[i](h, gen_w_hat[:, i, :]) # h는 이전에 들어온 값.
            inf_z_list.append(tmp_inf_z.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            inf_mean_list.append(tmp_inf_mean.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
            inf_var_list.append(tmp_inf_var.unsqueeze(1)) # (batch_size, 1, timesteps, d_latent)
        
        inf_z = torch.cat(inf_z_list, dim=1)
        inf_mean = torch.cat(inf_mean_list, dim=1)
        inf_var = torch.cat(inf_var_list, dim=1)
        
        return x_hat, gen_z, gen_mean, gen_var, inf_z, inf_mean, inf_var