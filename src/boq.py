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

        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        init_queries_gaussian(self.queries, num_queries, in_dim,
                              sigma=None, noise=1e-3, rotate=False, orthogonalize=False)

        self.query_pos = torch.zeros(1, num_queries, in_dim)
        if num_anchored > 0:
            grid_size = int(num_anchored**0.5)
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, grid_size),
                torch.linspace(0, 1, grid_size)
            ), -1).reshape(-1, 2)[:num_anchored]
            pos_embed = self.build_sincos_position_embedding(coords, in_dim)
            actual_anchored = min(coords.shape[0], num_anchored)
            self.query_pos[0, :actual_anchored, :] = pos_embed[:actual_anchored]

        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    @staticmethod
    def build_sincos_position_embedding(coords, dim):
        import math
        num = coords.shape[0]
        pe = torch.zeros(num, dim)
        div_term = torch.exp(torch.arange(0, dim // 2, 2).float() * -(math.log(10000.0) / (dim // 2)))
        for i, (x, y) in enumerate(coords):
            pe[i, 0::4] = torch.sin(x * div_term)
            pe[i, 1::4] = torch.cos(x * div_term)
            pe[i, 2::4] = torch.sin(y * div_term)
            pe[i, 3::4] = torch.cos(y * div_term)
        return pe

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)

        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)


        out = self.norm_out(out)
        return x, out, attn.detach(), self.queries.squeeze(0)  # return the features, output, attention, and queries

class RowNormalize(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (batch, n)
        return x * (x.shape[1] / x.sum(dim=self.dim, keepdim=True))
    
class BoQBlockWithMask(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, K=32):
        super().__init__()
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.global_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        self.num_queries = num_queries

        self.K = K  # Number of top-k queries to select

        # if not hidden_layer:
        #     self.slot_mask = torch.nn.Sequential(
        #         torch.nn.Linear(in_dim * 3, 1),
        #         torch.nn.ReLU(),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Dropout(0.3),
        #     )
        # else:
        #     hidden_dim = in_dim * 2
        #     self.slot_mask = torch.nn.Sequential(
        #         torch.nn.Linear(in_dim * 2, hidden_dim),
        #         torch.nn.ReLU(),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Dropout(0.2),
        #         torch.nn.Linear(hidden_dim, num_queries),
        #         torch.nn.ReLU(),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Dropout(0.2),
        #     )

        self.query_scorer = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim // 2, num_queries)
        )
        self.gumbel_tau = 1.0

    def st_gumbel_topk(self, logits, k, tau=1.0):
        y_soft = F.gumbel_softmax(logits, tau=tau, hard=False) 
        topk = torch.topk(y_soft, k=k, dim=-1).indices
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, topk, 1.0)
        y = y_hard - y_soft.detach() + y_soft  # ST trick
        return y

    def forward(self, x, cls=None):
        B = x.size(0)
        # x = self.encoder(x)

        global_x = x.mean(dim=1, keepdim=True)  # [B, 1, in_dim]
        global_q_cls = cls[:, None, :] if cls is not None else global_x
        global_feat = self.global_cross_attn(global_q_cls, x, x)[0]  # [B, 1, in_dim]
        query_logits = self.query_scorer(global_feat.squeeze(1)) # Shape: [B, num_queries]

        query_gate = self.st_gumbel_topk(query_logits, k=self.K, tau=self.gumbel_tau)  # [B, num_queries]

        q = self.queries.repeat(B, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        q = q * query_gate.unsqueeze(-1)  # Apply the query gate

        out, attn = self.cross_attn(q, x, x) 
        out = out + q
        out = self.norm_out(out)

        # mask = self.slot_mask(global_feat)  # [B, num_queries]
        # mask = mask.unsqueeze(-1)  # [B, num_queries, 1]
        # out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), query_gate, self.queries.squeeze(0) 


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
        masks_std = []
        self.masks = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, query_gate, q = self.boqs[i](x, cls)
                query_gate = query_gate.squeeze(-1)
                self.masks.append(query_gate)
                # mask [B, num_queries, 1]
                masks_std.append(query_gate.sum(dim=0).std(dim=0))  
            else:
                x, out, attn, q = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
            qs.append(q)
        
        masks_std = torch.stack(masks_std, dim=0).mean() if self.slot_mask else 0

        out = torch.cat(outs, dim=1) # [B, num_layers * num_queries, in_dim]

        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, qs, masks_std