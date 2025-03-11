import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.nn.attention import flex_attention

@dataclass
class GPTConfig:
    edge_attn: bool
    embed_map: dict
    target_sizes: dict
    block_size: int
    num_layers: int
    num_heads: int
    embed_dim: int
    dropout: float
    bias: bool
    device: str
    
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, embed_dim, num_heads, dropout, bias, edge_attn):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.edge_attn = edge_attn
        self.n_head = num_heads
        self.n_embd = embed_dim
        self.dropout = dropout
        self.num_edge_types = cfg.embed_map["edge_type"]
        
        self.w_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.w_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_bias = nn.Embedding(num_embeddings=self.num_edge_types, embedding_dim=num_heads)
        
        if edge_attn:
            self.w_edge_k = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.w_edge_v = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.edge_embeddings = nn.Embedding(num_embeddings=self.num_edge_types, embedding_dim=embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        

    def forward(self, x, bias_matrix):
        B, T, C = x.size()
        q, k, v = self.w_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # edge info
        scale = math.sqrt(q.size(-1))
        if self.edge_attn:
            edge_embeddings = self.edge_embeddings(bias_matrix)
            e_k = self.w_edge_k(edge_embeddings) # for score calculation
            e_v = self.w_edge_v(edge_embeddings) # for aggregation
            e_k = e_k.view(B, T, T, self.n_head, C // self.n_head).permute(0,3,1,2,4)
            k_modified = k.unsqueeze(-2) * e_k
            attn_scores = torch.einsum('bhie,bhjie->bhij', q, k_modified) / scale # (B, num_heads, q, k)
        else:
            attn_scores = torch.einsum('bije,bike->bijk', q, k) / scale
        
        # get casual mask
        temp_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(diagonal=0)
        attn_scores.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_scores.to(q.dtype)
            
        # add attn_bias to scores
        attn_bias = self.attn_bias(bias_matrix)
            
        if attn_bias.dim() == 3: # happens if batch size is 1 
            attn_bias = attn_bias.unsqueeze(0)
            
        attn_bias = attn_bias.permute(0, 3, 1, 2)
        attn_bias.masked_fill_(temp_mask.logical_not(), 0)
        attn_bias.to(q.dtype)
        attn_scores += attn_bias
        attn_scores = torch.softmax(attn_scores, dim=-1)
        
        # aggregate
        if self.edge_attn:
            e_v = e_v.view(B, T, T, self.n_head, C // self.n_head).permute(0,3,1,2,4)
            v_modified = v.unsqueeze(-2) * e_v
            y = torch.einsum('bhij,bhjie->bhie', attn_scores, v_modified)
        else:
            y = torch.einsum('bhje,bhek->bhjk', attn_scores, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.w_proj(y))
        return y
   
class Block(nn.Module):
    def __init__(self, cfg, embed_dim, num_heads, dropout, bias, edge_attn):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = CausalSelfAttention(cfg, embed_dim, num_heads, dropout, bias, edge_attn)
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(self, x, bias_matrix):
        x = x + self.attn(self.ln_1(x), bias_matrix)
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == "__main__":
    n_h = 2
    seq_length = 3
    batch_size = 4
    hidden_dim = 16
    cfg = {"block_size": seq_length, "num_heads": n_h}
    x = torch.randn(size=(batch_size ,seq_length, hidden_dim), device="cpu")
    bias = torch.randint(low=0, high=4, size=( batch_size,seq_length,seq_length)).to(x.device)
    bias_attn = CausalSelfAttention(embed_dim=hidden_dim, num_heads=n_h, dropout=0.0, bias=False, cfg=cfg).to(x.device)
    res = bias_attn(x, bias)
    
    #distances = torch.randint(low=0, high=4, size=(3,10,10)).to(x.device)
    #res = bias_attn(x, distances)
    #print(res.shape)
    
