# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
from torch_geometric.nn import GCNConv

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
        return x, out, attn.detach(), q
    
def grouped_mean_pooling(x, group_num):
    # x: [B, N, C]
    B, N, C = x.shape
    group_size = N // group_num
    pooled = []
    for g in range(group_num):
        start = g * group_size
        end = (g + 1) * group_size if g < group_num - 1 else N
        pooled.append(x[:, start:end, :].mean(dim=1))  # [B, C]
    pooled_feat = torch.cat(pooled, dim=-1)  # [B, group_num * C]
    return pooled_feat     

class RowNormalize(torch.nn.Module):
    def forward(self, x):
        # x: (batch, n)
        return x * (x.shape[1] / x.sum(dim=1, keepdim=True)) 
    
class SlotMaskAttention(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=4, attn_dim=None):
        super().__init__()
        self.num_queries = num_queries
        attn_dim = attn_dim or in_dim
        self.slot_embed = torch.nn.Parameter(torch.randn(1, num_queries, attn_dim))
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.2)
        self.attn = torch.nn.MultiheadAttention(attn_dim, nheads, batch_first=True)
        self.score_proj = torch.nn.Linear(attn_dim, 1)
        self.norm = torch.nn.LayerNorm(attn_dim)

    def forward(self, global_feat):
        # global_feat: [B, in_dim]
        x = self.encoder(global_feat)
        B = global_feat.shape[0]
        slot_embed = self.slot_embed.repeat(B, 1, 1)   # [B, num_queries, attn_dim]
        
        # k = self.k_proj(global_feat).unsqueeze(1)      # [B, 1, attn_dim]
        # v = self.v_proj(global_feat).unsqueeze(1)      # [B, 1, attn_dim]

        attn_output, _ = self.attn(slot_embed, x, x)   
        attn_output = self.norm(attn_output) 
        # print(attn_output[0])
        mask_scores = self.score_proj(attn_output).squeeze(-1)  # [B, num_queries]
        # print(mask_scores)
        mask = torch.sigmoid(mask_scores)  # or softmax
        return mask
    
class GNNPoolingHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        self.gnns = torch.nn.ModuleList([GCNConv(in_dim, in_dim) for _ in range(num_layers)])
        self.fc = torch.nn.Linear(in_dim, out_dim)
    def forward(self, slots):  # slots: [B, N, D]
        batch_out = []
        for s in slots:
            N = s.size(0)
            edge_index = torch.combinations(torch.arange(N, device=s.device), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            x = s
            for gcn in self.gnns:
                x = torch.relu(gcn(x, edge_index))
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
            pooled = x.mean(dim=0)   # [D]
            batch_out.append(self.fc(pooled))
        return torch.stack(batch_out, dim=0)
    
class BoQBlockWithMask(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, mlp=True, hidden_layer=False):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.dropout = torch.nn.Dropout(0.5)

        # self.slot_mask_attention = torch.nn.Sequential(
        #     SlotMaskAttention(in_dim, num_queries, nheads=nheads),
        #     RowNormalize()
        # )

        self.mlp = mlp
        if not mlp:
            self.slot_mask = torch.nn.Parameter(torch.ones(num_queries)) 
        else:
            if not hidden_layer:
                self.slot_mask = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, num_queries),
                    torch.nn.ReLU(),
                    torch.nn.Sigmoid(),
                    RowNormalize(),
                    torch.nn.Dropout(0.35),
                )
            else:
                hidden_dim = in_dim * 4
                self.slot_mask = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Sigmoid(),
                    RowNormalize(),
                    torch.nn.Dropout(0.25),
                    torch.nn.Linear(hidden_dim, num_queries),
                    torch.nn.ReLU(),
                    torch.nn.Sigmoid(),
                    RowNormalize(),
                    torch.nn.Dropout(0.25),
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
            # global_feat = x.mean(dim=1)
            # print(x.shape)
            # global_feat, _ = self.pool(x)  # [B, in_dim]
            # global_feat = grouped_mean_pooling(x, self.queries.shape[1]) 
            global_feat = x.mean(dim=1)  # [B, in_dim]
            # print(global_feat.shape)
            mask = self.slot_mask(global_feat)  # [B, num_queries]  
            # mask = self.slot_mask_attention(x)  # [B, num_queries]
            mask = mask[:, :, None]  # [B, num_queries, 1]
        # mask = mask * (mask.shape[1] / mask.sum(dim=1, keepdim=True)) 
        # mask = self.dropout(mask)
        # print(mask)

        out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), mask, q


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True, mlp=False, hidden_layer=False, gnn_pooling=False):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        self.slot_mask = slot_mask
        self.gnn_pooling = gnn_pooling

        in_dim = proj_channels
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQBlockWithMask(in_dim, num_queries, nheads=in_dim//64, mlp=mlp, hidden_layer=hidden_layer) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        if gnn_pooling:
            self.gnn_pooling_head = GNNPoolingHead(in_dim, row_dim, num_layers=num_layers//2)
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x):
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
                x, out, attn, mask, q = self.boqs[i](x)
                self.masks.append(mask)
                masks_std.append(mask.std(dim=1).mean(dim=0))
            else:
                x, out, attn, q = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
            qs.append(q)
        
        masks_std = torch.stack(masks_std, dim=0).mean(dim=0) if self.slot_mask else 0
        out = torch.cat(outs, dim=1)
        if not self.gnn_pooling:
            out = self.fc(out.permute(0, 2, 1))
            out = out.flatten(1)
        else:
            out = self.gnn_pooling_head(out)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, qs, masks_std