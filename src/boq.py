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
    def __init__(self, in_dim, num_queries, nheads=8, mlp=True, hidden_layer=False):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        self.mlp = mlp
        if not mlp:
            self.slot_mask = torch.nn.Parameter(torch.ones(num_queries)) 
        else:
            if not hidden_layer:
                self.slot_mask = torch.nn.Linear(in_dim, num_queries)
            else:
                hidden_dim = in_dim // 4
                self.slot_mask = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, num_queries),
                )

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        if not self.mlp:
            mask = torch.sigmoid(self.slot_mask)[None, :, None]  # [1, num_queries, 1]
        else:
            global_feat = x.mean(dim=1)
            mask = torch.sigmoid(self.slot_mask(global_feat))  # [B, num_queries, 1]
            mask = mask[:, :, None]  # [B, num_queries, 1]

        out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), mask


class BoQ(torch.nn.Module):
    def __init__(self, 
                 in_channels,        # list, e.g. [1024, 1024]
                 proj_channels=512,
                 num_queries=32,
                 num_layers=2,  
                 row_dim=32,
                 slot_mask=True,
                 mlp=False,
                 hidden_layer=False):
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list, e.g. [1024, 1024]"
        self.proj_cs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_ch, proj_channels, kernel_size=3, padding=1) for in_ch in in_channels
        ])
        self.norm_inputs = torch.nn.ModuleList([
            torch.nn.LayerNorm(proj_channels) for _ in in_channels
        ])
        self.slot_mask = slot_mask

        self.boq_blocks = torch.nn.ModuleList()
        for _ in in_channels:
            layer_blocks = torch.nn.ModuleList()
            for _ in range(num_layers):
                if slot_mask:
                    layer_blocks.append(
                        BoQBlockWithMask(proj_channels, num_queries, nheads=proj_channels//64, mlp=mlp, hidden_layer=hidden_layer)
                    )
                else:
                    layer_blocks.append(
                        BoQBlock(proj_channels, num_queries, nheads=proj_channels//64)
                    )
            self.boq_blocks.append(layer_blocks)
 
        self.fc = torch.nn.Linear(len(in_channels)*num_queries, row_dim)
        
    def forward(self, features_list):
        outs, attns = [], []
        for i, x in enumerate(features_list):
            # x: [B, C, H, W]
            x = self.proj_cs[i](x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.norm_inputs[i](x)
     
            for boq in self.boq_blocks[i]:
                if self.slot_mask:
                    x, out, attn, _ = boq(x)
                else:
                    x, out, attn = boq(x)
            outs.append(out)
            attns.append(attn)
        out = torch.cat(outs, dim=1)                      # [B, num_input_layers*num_queries, in_dim]
        out = self.fc(out.permute(0, 2, 1))               # [B, in_dim, row_dim]
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns