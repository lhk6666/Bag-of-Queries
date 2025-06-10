# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
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
    def __init__(self, in_dim, num_queries, nheads=8):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        # self.slot_mask = torch.nn.Parameter(torch.ones(num_queries)) 

        self.slot_mask_mlp = torch.nn.Linear(in_dim, num_queries)

        # hidden_dim = in_dim // 2
        # self.slot_mask_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(in_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, num_queries)
        # )

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        # mask = torch.sigmoid(self.slot_mask)[None, :, None]  # [1, num_queries, 1]

        global_feat = x.mean(dim=1)
        mask = torch.sigmoid(self.slot_mask_mlp(global_feat))  # [B, num_queries, 1]
        mask = mask[:, :, None]  # [B, num_queries, 1]

        out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), mask


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        self.slot_mask = slot_mask
        in_dim = proj_channels
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQBlockWithMask(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, _ = self.boqs[i](x)
            else:
                x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns