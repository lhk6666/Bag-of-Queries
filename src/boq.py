# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn

@torch.no_grad()
def init_queries_gaussian(param: nn.Parameter, num_queries: int, emb_dim: int,
                          sigma: float|None=None, noise: float=1e-3, rotate: bool=False, orthogonalize: bool=False):

    device, dtype = param.device, param.dtype
    Q, C = num_queries, emb_dim

    centers = torch.linspace(0, C-1, steps=Q, device=device, dtype=dtype)     # [Q]
    grid    = torch.arange(C, device=device, dtype=dtype)[None, :]            # [1,C]
    spacing = (C-1) / max(Q-1, 1)
    if sigma is None:
        sigma = 20 * spacing                                               

    q = torch.exp(-0.5 * ((grid - centers[:, None]) / sigma)**2)              # [Q,C]
    if noise > 0:
        q = q + noise * torch.randn_like(q)
    # q = F.normalize(q, p=2, dim=1)

    if rotate:
        A = torch.randn(C, C, device=device, dtype=dtype)
        R, _ = torch.linalg.qr(A) 
        q = q @ R
        q = F.normalize(q, p=2, dim=1)

    if orthogonalize:
        U, _ = torch.linalg.qr(q.T)  # [C,Q]
        q = U.T                      # [Q,C]

    param.copy_(q.unsqueeze(0))    

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, num_anchored=0):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.num_anchored = num_anchored

        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.1)
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        # init_queries_gaussian(self.queries, num_queries, in_dim,
        #                       sigma=None, noise=1e-3, rotate=False, orthogonalize=False)
        # self.pool = nn.AdaptiveAvgPool1d(num_queries)  # Pooling to get a single vector per query

        # self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.x_q_cnn = torch.nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0)
        self.q_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        # x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)

        # q = q + self.self_attn(q, q, q)[0]
        q = q + self.cross_attn(q, x, x)[0]  # Cross-attention with the input features
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)


        out = self.norm_out(out)
        return x, out, attn.detach().cpu(), q[0].detach().cpu()  # return the features, output, attention, and queries

class RowNormalize(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (batch, n)
        return x * (x.shape[1] / x.sum(dim=self.dim, keepdim=True))
    
class BoQBlockWithMask(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super().__init__()
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, 1 + num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.cross_attn_q = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        # self.mask_fc = nn.Sequential(
        #     torch.nn.Linear(in_dim, num_queries),
        #     nn.Sigmoid()
        # )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        self.ffn_q = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim), 
            nn.GELU(),
            nn.Linear(4 * in_dim, in_dim)
        )

        self.router = nn.Linear(in_dim, in_dim)  # A
        self.temp = nn.Parameter(torch.tensor(1.0)) 

        self.num_queries = num_queries

    def compute_gate(self,q, q_cls, topk=None):
        cls_proj = self.router(q_cls).unsqueeze(1)               # [B,1,C]
        score = (q * cls_proj).sum(dim=-1) / (q.size(-1) ** 0.5) # [B,N]
        score = score / (self.temp + 1e-6)

        gate = torch.sigmoid(score)  # [B,N]
        if topk is not None:
            idx = gate.topk(topk, dim=-1).indices
            mask = torch.zeros_like(gate).scatter(1, idx, 1.0)
            gate = gate * mask
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-6)
        return gate 

    def forward(self, x, cls=None):
        B = x.size(0)
        # x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)
        L = self.num_queries + 1
        self_attn_mask = torch.zeros(L, L, dtype=torch.bool, device=x.device)
        self_attn_mask[1:, 0] = True  
        q = q + self.self_attn(q, q, q, attn_mask=self_attn_mask)[0]
        q = q + self.cross_attn_q(q, x, x)[0]  # Cross-attention with the input features
        q = q + self.ffn_q(q)  # Feed-forward network
        q = self.norm_q(q)
        q_cls = q[:, 0]  # [B, C]
        q = q[:, 1:]  # [B, num_queries, C]

        # out_gate = self.mask_fc(q_cls).unsqueeze(-1)  # [B, num_queries, 1]
        out_gate = self.compute_gate(q, q_cls).unsqueeze(-1)  # [B, num_queries, 1]

        out, attn = self.cross_attn(q, x, x)
        
        out = self.norm_out(out)
        out = out * out_gate

        return x, out, attn.detach().cpu(), q[0].detach().cpu(), out_gate.squeeze(-1).detach().cpu()  # return the features, output, attention, queries, and out_gate


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=1, padding=0)
        in_dim = proj_channels
        self.norm_input = torch.nn.LayerNorm(in_dim)

        self.slot_mask = slot_mask
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQBlockWithMask(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x, cls=None):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        outs = []
        attns = []
        qs = []
        out_gate_samples = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, q, out_gate = self.boqs[i](x, cls)
                for b in range(out_gate.shape[0]):
                    if b in [0, 32, 64]:
                        out_gate_samples.append(out_gate[b])
            else:
                x, out, attn, q = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
            qs.append(q)
        
        out_gate_samples = torch.stack(out_gate_samples, dim=0) if out_gate_samples else None

        out = torch.cat(outs, dim=1) # [B, num_layers * num_queries, in_dim]

        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, qs, out_gate_samples