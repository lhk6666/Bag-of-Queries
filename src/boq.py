# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, num_anchored=0):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.num_anchored = num_anchored

        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

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
        # x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)

        pos = self.query_pos.repeat(B, 1, 1).to(q.device)
        q = q + pos 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)


        out = self.norm_out(out)
        return x, out, attn.detach(), q

class RowNormalize(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (batch, n)
        return x * (x.shape[1] / x.sum(dim=self.dim, keepdim=True))
    
class BoQBlockWithMask(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, mlp=True, hidden_layer=False, in_channels=768):
        super().__init__()
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_channels))
        self.self_attn = torch.nn.MultiheadAttention(in_channels, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_channels)
        self.cross_attn = torch.nn.MultiheadAttention(in_channels, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_channels)

        # self.slot_mask_attention = torch.nn.Sequential(
        #     SlotMaskAttention(in_dim, num_queries, nheads=nheads),
        #     RowNormalize()
        # )

        self.mlp = mlp
        self.num_queries = num_queries
        if not mlp:
            self.slot_mask = torch.nn.Parameter(torch.ones(num_queries)) 
        else:
            if not hidden_layer:
                self.slot_mask = torch.nn.Sequential(
                    torch.nn.Linear(in_channels * 3, 1),
                    # torch.nn.ReLU(),
                    torch.nn.Sigmoid(),
                    # torch.nn.Softmax(dim=1),
                    # RowNormalize(),
                    torch.nn.Dropout(0.1),
                )
            else:
                # hidden_dim = in_channels * 4
                # self.slot_mask = torch.nn.Sequential(
                #     torch.nn.Linear(in_channels, hidden_dim),
                #     torch.nn.ReLU(),
                #     torch.nn.Sigmoid(),
                #     RowNormalize(),
                #     torch.nn.Dropout(0.2),
                #     torch.nn.Linear(hidden_dim, num_queries),
                #     torch.nn.ReLU(),
                #     torch.nn.Sigmoid(),
                #     RowNormalize(),
                #     torch.nn.Dropout(0.2),
                # )
                self.slot_mask = torch.nn.Sequential(
                    torch.nn.Linear(in_channels * 3, in_channels * 4),
                    torch.nn.ReLU(),
                    torch.nn.Softmax(dim=1),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(in_channels * 4, 1),
                    torch.nn.ReLU(),
                    # RowNormalize(),
                    torch.nn.Softmax(dim=1),
                )

    def forward(self, x, cls=None):
        B = x.size(0)
        # x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        # cls = cls[:, None, :]
        # q_cls = torch.cat([cls, q], dim=1)  # [B, num_queries + 1, in_dim]
        # global_q_cls = q_cls.mean(dim=1)  # [B, in_dim]

        global_x = x.mean(dim=1)  # [B, in_dim]
        global_x = global_x[:, None, :]  # [B, 1, in_dim]
        global_x = global_x.repeat(1, self.num_queries, 1)  # [B, num_queries, in_dim]
        cls = cls[:, None, :]
        cls = cls.repeat(1, self.num_queries, 1)  # [B, num_queries, in_dim]
        q_cls = torch.cat([cls, q, global_x], dim=-1)  # [B, num_queries, in_dim * 3]

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        if not self.mlp:
            mask = torch.sigmoid(self.slot_mask)[None, :, None]  # [1, num_queries, 1]
        else:
            # mask = self.slot_mask(global_q_cls)  # [B, num_queries]  

            mask = self.slot_mask(q_cls)  # [B, num_queries, 1]

        # mask = mask * (mask.shape[1] / mask.sum(dim=1, keepdim=True)) 
        # mask = self.dropout(mask)
        # print(mask)

        out = out * mask           # [B, num_queries, in_dim]

        return x, out, attn.detach(), mask, q


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True, mlp=False, hidden_layer=False, gnn_pooling=False, global_slot_mask=False):
        super().__init__()
        # self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(in_channels)
        
        self.slot_mask = slot_mask

        in_dim = in_channels
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQBlockWithMask(in_dim, num_queries, nheads=in_dim//64, mlp=mlp, hidden_layer=hidden_layer, in_channels=in_channels) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x, cls=None):
        # reduce input dimension using 3x3 conv when using ResNet
        # x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        attns = []
        qs = []
        masks_std = []
        self.masks = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, mask, q = self.boqs[i](x, cls)
                self.masks.append(mask)
                # mask [B, num_queries, 1]
                masks_std.append(mask.std(dim=0))
                # masks_std.append(mask.std(dim=0))
                # print(masks_std[0].shape)
            else:
                x, out, attn, q = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
            qs.append(q)
        
        masks_std = torch.stack(masks_std, dim=0).mean(dim=0) if self.slot_mask else 0

        out = torch.cat(outs, dim=1) # [B, num_layers * num_queries, in_dim]

        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, qs, masks_std