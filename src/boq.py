# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as f

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        # x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()
    
class BoQBlockWithMask(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8,):
        super().__init__()
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        self.cls_cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.temp = torch.nn.Parameter(torch.ones(1), requires_grad=True)  # temperature for the gate
        self.router = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim * 4, in_dim),
        )
        self.norm_cls = torch.nn.LayerNorm(in_dim)

    def compute_gate(self,q, cls, topk=None):
        cls = self.router(cls).unsqueeze(1) if cls is not None else None
        score = (q * cls).sum(dim=-1)/(q.size(-1) ** 0.5) # [B, N]
        score = score / (self.temp + 1e-6)
        gate = torch.sigmoid(score)  # [B,N]
        # gate = torch.softmax(score / 10.0, dim=-1)  # [B,N]
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
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        mask = self.compute_gate(q, cls) if cls is not None else torch.ones(B, out.size(1), device=out.device)[:, :, None]  # [B, num_queries, 1]
        mask = mask.unsqueeze(-1)  # [B, num_queries, 1]

        out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), mask, q


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True, mlp=False, hidden_layer=False):
        super().__init__()
        # self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        self.slot_mask = slot_mask
        in_dim = in_channels
        # self.norm_input = torch.nn.LayerNorm(in_dim)
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQBlockWithMask(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x, cls=None):
        # reduce input dimension using 3x3 conv when using ResNet
        # x = self.proj_c(x)
        # x = x.flatten(2).permute(0, 2, 1)
        # x = self.norm_input(x)
        
        outs = []
        attns = []
        masks = []
        queries = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, mask, q = self.boqs[i](x, cls)
                masks.append(mask)
                queries.append(q)
            else:
                x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
        

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, masks, queries