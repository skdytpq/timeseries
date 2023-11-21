"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention,ProbScaleDotProductAttention

import torch

import numpy

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head): 
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.qk1 = nn.Linear(d_model,d_model)
        self.qk2 = nn.Linear(d_model , d_model)
        self.qk3 = nn.Linear(d_model , d_model)
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # query key Matrix : MLP Layer 통과시키기

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
def reparameterize(mu, logvar):
    """
    Will a single z be enough ti compute the expectation
    for the loss??
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class ProbMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, n_layer):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_layer = n_layer
        self.attention = ScaleDotProductAttention()
        self.probattention = ProbScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.qk_mu = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(n_layer)])
        self.qk_var = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(n_layer)])
    def forward(self, aq, ak, av, mask=None,dot = None,prior_attn = None,pprior_attn = None):
        # 1. dot product with weight matrices
        aq, ak, av = self.w_q(aq) , self.w_k(ak) , self.w_v(av)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split tensor by number of heads
        aq, ak, av = self.split(aq), self.split(ak), self.split(av)
        # query key Matrix : MLP Layer 통과시키기

        if dot:
            d_tensor = ak[-1]
            ak_t = ak.transpose(2, 3)
            attn = (aq @ ak_t) / math.sqrt(d_tensor)
            attn = self.concat(attn)
            return attn
        d_tensor = ak[-1]
        ak_t = ak.transpose(2, 3) # B X C X H
        attn = (aq @ ak_t) / math.sqrt(d_tensor) # NLL Loss 이전 Matrix
        q_mu = self.qk_mu(aq)
        k_mu = self.qk_mu(ak)
        q_var = self.qk_var(aq)
        k_var = self.qk_var(ak)
        k_mu_t = k_mu.transpose(2,3)
        k_var_t = k_var.transpose(2,3)
        mu = q_mu@k_mu_t
        var = q_var@k_var_t
        p_alpha = reparameterize(mu,var)
        # 3. do scale dot product to compute similarity
        out, attention = self.probattention(p_alpha, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        attn = self.concat(attn)
        attn = self.w_concat(attn)
        attention = self.concat(attention)
        attention = self.w_concat(attention)
        if prior_attn and pprior_attn != None:
            attention = attention + pprior_attn
            attn = attn + prior_attn


        # 5. visualize attention map
        # TODO : we should implement visualization
        return out , attn , attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor