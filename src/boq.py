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
    
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Step 1: CLS -> Prototypes (in K-space) ----------
class CLSPrototypes(nn.Module):
    """
    用 CLS 生成 num_clusters 个 K 空间原型（每图一套），全可微。
    C = normalize( Linear(CLS_in_K) ), 然后对 Kx 做 soft assignment 得到 P。
    """
    def __init__(self, dim: int, num_clusters: int, scale_init: float = 20.0, repel_reg: float = 0.0):
        super().__init__()
        self.num_clusters = num_clusters
        # 将 K 空间中的 CLS 映射到 N 个原型（每个 D 维）
        self.cls_to_proto = nn.Linear(dim, num_clusters * dim, bias=True)
        nn.init.xavier_uniform_(self.cls_to_proto.weight, gain=0.01)
        nn.init.zeros_(self.cls_to_proto.bias)
        # 可学习的尺度，相当于温度的倒数
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale_init)))
        self.repel_reg = float(repel_reg)

    def forward(self, Kx: torch.Tensor, Kcls: torch.Tensor):
        """
        Kx:   [B, Nx, D]  (L2-normalized)
        Kcls: [B, 1,  D]  (L2-normalized)
        返回:
          P:    [B, Nx, N]   patch→cluster 软分配
          C:    [B, N,  D]   原型（K 空间）
          loss: 原型分离正则（可加到总loss）
        """
        B, Nx, D = Kx.shape
        # 由 CLS 生成原型（每图一套）
        C = self.cls_to_proto(Kcls.squeeze(1)).view(B, self.num_clusters, D)  # [B,N,D]
        C = F.normalize(C, dim=-1)

        scale = torch.clamp(self.log_scale.exp(), min=1.0, max=100.0)
        # logits: [B, Nx, N] = <Kx, C>*scale
        logits = scale * torch.einsum('bid,bnd->bin', Kx, C)
        P = torch.softmax(logits, dim=-1)

        # 原型分离正则（可选）：惩罚不同原型夹角过小
        loss = torch.tensor(0.0, device=Kx.device)
        if self.repel_reg > 0:
            # 余弦相似的平方和（去对角）
            cos = torch.einsum('bnd,bmd->bnm', C, C)  # [B,N,N]
            eye = torch.eye(self.num_clusters, device=Kx.device).unsqueeze(0)
            off = (cos - eye) ** 2
            loss = self.repel_reg * off.sum(dim=(1,2)).mean()
        return P, C, loss
    
class CLSBinner(nn.Module):
    """
    把 CLS↔x 的余弦相似度 s ∈ [-1,1] 通过 N 个可学习中心做软分箱：
      logits_{j,k} = - (s_j - mu_k)^2 / (2*sigma^2)
      P_{j,k} = softmax_k(logits_{j,k})
    可选 Sinkhorn 保列和均衡（每簇拿到差不多数量的 patch）。
    """
    def __init__(self, n_bins: int, sigma: float = 0.15, use_sinkhorn: bool = False, sinkhorn_iters: int = 5):
        super().__init__()
        self.n_bins = n_bins
        # 初始化中心在 [-1,1] 上平均分布
        centers = torch.linspace(-1.0, 1.0, steps=n_bins)
        self.mu = nn.Parameter(centers)         # [N]
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters

    @staticmethod
    def _sinkhorn(P, iters: int):
        """
        P: [B, Nx, N]  行和=1（按 softmax 已满足）；再规范列和≈Nx/N，保持可微。
        """
        B, Nx, N = P.shape
        # 目标列和（每簇均衡）：每列 ≈ Nx/N
        col_target = float(Nx) / float(N)
        for _ in range(iters):
            # 归一行（数值稳健）
            P = P / (P.sum(dim=2, keepdim=True) + 1e-6)
            # 归一列到目标和
            col_sum = P.sum(dim=1, keepdim=True) + 1e-6          # [B,1,N]
            P = P * (col_target / col_sum)                       # scale 每列
        return P

    def forward(self, Kx: torch.Tensor, Kcls: torch.Tensor):
        """
        Kx:   [B, Nx, D]  L2-normalized 的 patch（在 K 空间）
        Kcls: [B, 1,  D]  L2-normalized 的 CLS（在同一个 K 空间）
        返回:
          P: [B, Nx, N]   每个 patch 的软分配到 N 个簇
          s: [B, Nx]      CLS↔x 的余弦相似度
        """
        # 1) 计算 CLS↔x 的余弦相似度（完全可微）
        s = torch.einsum('bid,bjd->bij', Kcls, Kx).squeeze(1)    # [B, Nx]  ∈ [-1,1]
        print(s)

        # 2) RBF 软分箱
        sigma = torch.clamp(self.log_sigma.exp(), min=1e-3)
        # broadcasting: s->[B,Nx,1], mu->[1,1,N]
        dist2 = (s.unsqueeze(-1) - self.mu.view(1, 1, -1)) ** 2  # [B,Nx,N]
        logits = - dist2 / (2 * sigma**2 + 1e-6)
        P = torch.softmax(logits, dim=-1)                        # [B,Nx,N]

        # 3) 可选：Sinkhorn 让每个簇拿到接近 Nx/N 的质量（仍可微）
        if self.use_sinkhorn:
            P = self._sinkhorn(P, self.sinkhorn_iters)

        return P, s


# ---------- Step 2: Query -> Cluster 路由，得到每个 query 的目标注意力 T ----------
class QueryClusterRouter(nn.Module):
    """
    可学习的 query→cluster 路由矩阵 R ∈ R^{Q×N}，经 softmax 得到 Ŕ。
    用 T_i = Σ_k Ŕ_{ik} * P_k 生成每个 query 的目标注意力分布（对 Nx 归一化）。
    """
    def __init__(self, num_queries: int, num_clusters: int, temp_init: float = 1.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_clusters = num_clusters
        self.logits = nn.Parameter(torch.zeros(num_queries, num_clusters))
        nn.init.xavier_uniform_(self.logits, gain=0.01)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temp_init)))

    def forward(self, P: torch.Tensor):
        """
        P: [B, Nx, N]   patch→cluster 分布
        返回:
          T:     [B, Q, Nx]  每个 query 的目标注意力
          R_hat: [Q, N]      路由矩阵（行 softmax）
        """
        B, Nx, N = P.shape
        Q = self.num_queries
        Ttemp = torch.clamp(self.log_temp.exp(), min=0.2, max=10.0)
        R_hat = F.softmax(self.logits / Ttemp, dim=-1)           # [Q,N]
        # P^T: [B,N,Nx]; T = R_hat @ P^T → [B,Q,Nx]
        Pt = P.transpose(1, 2).contiguous()                      # [B,N,Nx]
        T = torch.einsum('qn,bnk->bqk', R_hat, Pt)               # [B,Q,Nx]
        T = T / (T.sum(dim=-1, keepdim=True) + 1e-6)
        return T, R_hat


# ---------- Step 3: 整体模块，构造 mask 加到 cross-attn ----------
class BoQWithProtoMask(nn.Module):
    """
    三步合一：
      1) CLS->Prototypes 得到 P (patch→cluster)
      2) Router: P -> 每个 query 的目标注意力 T
      3) 在 cross-attn 的 logits 上加 additive bias = alpha * log(T+eps)

    只需指定：num_clusters (N) 与 num_queries (Q)。
    """
    def __init__(self, dim: int, num_queries: int, num_clusters: int, nheads: int = 8,
                 alpha_init: float = 0.4, eps: float = 1e-3,
                 proto_scale_init: float = 20.0, proto_repel: float = 0.0,
                 router_temp_init: float = 1.0,
                 kl_weight: float = 0.0, div_weight: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_clusters = num_clusters
        self.nheads = nheads
        self.eps = eps

        # Queries & attentions
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.self_attn = nn.MultiheadAttention(dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

        # Step 1: CLS->Prototypes (in K-space)
        # self.protos = CLSPrototypes(dim, num_clusters, scale_init=proto_scale_init, repel_reg=proto_repel)
        self.binners = CLSBinner(num_clusters, sigma=0.15, use_sinkhorn=True, sinkhorn_iters=5)

        # Step 2: Query->Cluster Router
        self.router = QueryClusterRouter(num_queries, num_clusters, temp_init=router_temp_init)

        # Step 3: mask 混合强度 α（可退火）
        self.register_buffer("_alpha", torch.tensor(alpha_init, dtype=torch.float32))

        # 可选训练项
        self.kl_weight = float(kl_weight)   # KL(A || T)
        self.div_weight = float(div_weight) # diversity on R

    def set_alpha(self, alpha: float):
        alpha = max(0.0, min(1.0, float(alpha)))
        self._alpha.data = torch.tensor(alpha, dtype=torch.float32, device=self._alpha.device)

    def _project_K(self, x: torch.Tensor):
        """用 cross_attn 的 W_K 把 x 投到 K 空间并 L2-normalize。"""
        W = self.cross_attn.in_proj_weight
        b = self.cross_attn.in_proj_bias
        D = self.cross_attn.embed_dim
        Wk = W[D:2*D, :]
        bk = b[D:2*D] if b is not None else None
        Kx = F.linear(x, Wk, bk)
        Kx = F.normalize(Kx, dim=-1)
        return Kx

    def forward(self, x: torch.Tensor, cls: torch.Tensor):
        """
        x:   [B, Nx, D]   patch tokens
        cls: [B, D] or [B,1,D]  global token
        返回：
          out:   [B, Q, D]
          attn:  [B, H, Q, Nx]  (带 bias 后的注意力)
          aux:   dict(P, C, T, R_hat, losses, alpha_used)
        """
        B, Nx, D = x.shape
        H, Q, N = self.nheads, self.num_queries, self.num_clusters
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)  # [B,1,D]

        # ---- Queries (self-attn) ----
        q0 = self.queries.repeat(B, 1, 1)                  # [B,Q,D]
        q_sa, _ = self.self_attn(q0, q0, q0, need_weights=False)
        q = self.norm_q(q0 + q_sa)                         # [B,Q,D]

        # ---- K-space ----
        # Kx   = self._project_K(x)                          # [B,Nx,D]
        # Kcls = self._project_K(cls)                        # [B,1, D]

        # ---- Step 1: CLS->Prototypes → P ----
        # P, C, proto_loss = self.protos(x, cls)           # P: [B,Nx,N], C: [B,N,D]
        P, s = self.binners(x, cls)                        # P: [B,Nx,N], s: [B,Nx]

        # ---- Step 2: Router(P) → T ----
        T, R_hat = self.router(P)                          # T: [B,Q,Nx], R_hat: [Q,N]

        # ---- Step 3: add mask to cross-attn logits ----
        alpha = float(self._alpha.item())
        if alpha > 0:
            log_bias = alpha * torch.log(T + self.eps)     # [B,Q,Nx]
            attn_mask = log_bias.unsqueeze(1).repeat(1, H, 1, 1).reshape(B * H, Q, Nx)
        else:
            attn_mask = None

        out, attn = self.cross_attn(
            q, x, x,
            need_weights=True,
            average_attn_weights=True,
            attn_mask=attn_mask
        )                                                  # attn: [B,H,Q,Nx]
        out = self.norm_out(out)

        # ---- Optional losses ----
        losses = {}
        total_aux_loss = torch.tensor(0.0, device=x.device)

        if self.kl_weight > 0:
            # 让真实注意力靠近 T（多头平均后对齐）
            A = attn.mean(dim=1)                           # [B,Q,Nx]
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
            kl = (A * (A.add(1e-6).log() - T.add(1e-6).log())).sum(dim=-1).mean()
            losses["kl_loss"] = self.kl_weight * kl
            total_aux_loss = total_aux_loss + losses["kl_loss"]

        if self.div_weight > 0:
            # 多样性：惩罚 R_hat R_hat^T 的非对角
            G = R_hat @ R_hat.t()                          # [Q,Q]
            I = torch.eye(Q, device=G.device)
            div = ((G - I) ** 2).sum() / (Q ** 2)
            losses["div_loss"] = self.div_weight * div
            total_aux_loss = total_aux_loss + losses["div_loss"]

        # if self.protos.repel_reg > 0:
        #     losses["proto_repel"] = proto_loss
        #     total_aux_loss = total_aux_loss + proto_loss

        # aux = dict(
        #     P=P.detach(), C=C.detach(), T=T.detach(),
        #     R_hat=R_hat.detach(),
        #     alpha_used=alpha,
        #     losses=losses,
        #     attn=attn.detach()
        # )

        return x, out, attn.detach(), attn_mask, q



class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32, slot_mask=True, mlp=False, hidden_layer=False):
        super().__init__()
        # self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        self.slot_mask = slot_mask
        in_dim = in_channels
        # self.norm_input = torch.nn.LayerNorm(in_dim)
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQWithProtoMask(in_dim, num_queries, num_clusters=8, nheads=in_dim//64) for _ in range(num_layers)])
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