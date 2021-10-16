import torch
import torch.nn as nn


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU() if act =='gelu' else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.net(x)) + x


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim,
                 heads=8, 
                 dim_head = 64,
                 dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5     # 1 / sqrt(d_k)

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False) # W_q, W_k, W_v
        self.to_k = nn.Linear(dim, inner_dim, bias = False) # W_q, W_k, W_v
        self.to_v = nn.Linear(dim, inner_dim, bias = False) # W_q, W_k, W_v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        B, NQ = query.shape[:2]
        B, NK = key.shape[:2]
        B, NV = value.shape[:2]
        # Input：x -> [B, N, C_in]
        # [B, N, h*d] -> [B, N, h, d] -> [B, h, N, d]
        q = self.to_q(query).view(B, NQ, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.to_k(key).view(B, NK, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = self.to_v(value).view(B, NV, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Q*K^T / sqrt(d_k) : [B, h, N, d] X [B, h, d, N] = [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax
        attn = self.attend(dots)
        # softmax(Q*K^T / sqrt(d_k)) * V ：[B, h, N, N] X [B, h, N, d] = [B, h, N, d]
        out = torch.matmul(attn, v)
        # [B, h, N, d] -> [B, N, h*d]=[B, N, C_out], C_out = h*d
        out = out.permute(0, 2, 1, 3).contiguous().view(B, NQ, -1)
        
        return self.norm(self.to_out(out)) + query


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 dim,            # hidden_dim
                 heads, 
                 dim_head,
                 mlp_dim=2048,
                 dropout = 0.,
                 act='relu'):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout, act)

    def forward(self, x, pos=None):
        # x -> [B, N, d_in]
        q = k = x if pos is None else x + pos
        v = x
        x = self.attn(q, k, v)
        x = self.ffn(x)

        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 dim,            # hidden_dim
                 depth,          # num_encoder
                 heads,
                 dim_head,
                 mlp_dim=2048,
                 dropout = 0.,
                 act='relu'):
        super().__init__()
        # build encoder
        self.encoders = nn.ModuleList([
                                TransformerEncoderLayer(
                                    dim, 
                                    heads, 
                                    dim_head, 
                                    mlp_dim, 
                                    dropout, 
                                    act) for _ in range(depth)])

    def forward(self, x, pos=None):
        for m in self.encoders:
            x = m(x, pos)
        
        return x
