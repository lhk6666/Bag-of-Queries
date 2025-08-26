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
    不加额外loss的防塌版本：
      - 可选 QR 正交化 / soft-orth
      - 全局码书门控融合
      - P 的列均衡(Sinkhorn)或均匀混合
      - 温度自适应
    """
    def __init__(self, dim:int, num_clusters:int,
                 use_qr_ortho: bool = True,      # True: QR 正交；False: soft-orth
                 global_codebook: bool = True,   # 是否引入全局码书
                 mix_uniform_eps: float = 0.03,  # P ← (1-ε)P+ε/N
                 sinkhorn_iters: int = 0,        # >0 则做列均衡
                 target_logit_std: float = 0.5,  # 目标 logits 方差
                 per_head: int = 0               # >0 则多头，每头簇数=per_head，头数=N//per_head
                 ):
        super().__init__()
        self.N, self.D = num_clusters, dim
        self.use_qr_ortho = use_qr_ortho
        self.global_codebook = global_codebook
        self.mix_uniform_eps = mix_uniform_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.target_logit_std = target_logit_std

        # (可选) 多头
        if per_head and num_clusters % per_head == 0:
            self.num_heads = num_clusters // per_head
            self.per_head = per_head
        else:
            self.num_heads = 1
            self.per_head = num_clusters

        self.cls_to_proto = nn.Linear(dim, num_clusters * dim, bias=True)
        nn.init.xavier_uniform_(self.cls_to_proto.weight, gain=0.01)
        nn.init.zeros_(self.cls_to_proto.bias)

        # 门控：控制 全局码书 vs 局部原型 的融合
        if global_codebook:
            self.G = nn.Parameter(F.normalize(torch.randn(num_clusters, dim), dim=-1))
            self.gate = nn.Sequential(
                nn.Linear(dim, num_clusters), nn.Sigmoid()
            )

    @staticmethod
    def _batch_qr_orthonormalize(C):  # C:[B,N,D], N<=D
        # 对每个样本做 QR，使行（N个向量）正交
        # 实现：对转置做QR，再转回
        # Q_col:[B,D,N] → 转回 [B,N,D]
        Q_col, _ = torch.linalg.qr(C.transpose(1,2), mode='reduced')  # [B,D,N]
        C_ortho = Q_col.transpose(1,2)  # [B,N,D]
        return F.normalize(C_ortho, dim=-1)

    @staticmethod
    def _soft_orth_step(C, alpha=0.5):  # C:[B,N,D], 任意 N,D
        # 一步的去相关修正：C <- C - α ( (C C^T - I) C )
        B,N,D = C.shape
        I = torch.eye(N, device=C.device).unsqueeze(0)
        G = torch.matmul(C, C.transpose(1,2)) - I               # [B,N,N]
        C = C - alpha * torch.matmul(G, C)                       # [B,N,D]
        return F.normalize(C, dim=-1)

    @staticmethod
    def _sinkhorn(P, iters:int):
        # P:[B,Nx,N]，行和=1; 让列和≈均衡
        B,Nx,N = P.shape
        col_target = float(Nx)/float(N)
        for _ in range(iters):
            P = P / (P.sum(dim=2, keepdim=True) + 1e-6)         # 行归一
            col_sum = P.sum(dim=1, keepdim=True) + 1e-6         # 列和
            P = P * (col_target / col_sum)
        return P

    def forward(self, Kx: torch.Tensor, Kcls: torch.Tensor):
        """
        Kx:   [B, Nx, D] (L2-normalized)
        Kcls: [B, 1,  D] (L2-normalized)
        返回: P:[B,Nx,N], C:[B,N,D]
        """
        B,Nx,D = Kx.shape
        cls = Kcls.squeeze(1)                                    # [B,D]

        # 1) 由 CLS 生成局部原型
        C_loc = self.cls_to_proto(cls).view(B, self.N, D)        # [B,N,D]
        C_loc = F.normalize(C_loc, dim=-1)

        # 2) 正交化（零loss防塌的关键）
        if self.use_qr_ortho and self.N <= D:
            C_loc = self._batch_qr_orthonormalize(C_loc)         # 真·正交
        else:
            C_loc = self._soft_orth_step(C_loc, alpha=0.5)       # 近似正交一步

        # 3) 与全局码书门控融合（提升稳态 & 跨图一致性）
        if self.global_codebook:
            G = self.G.unsqueeze(0).expand(B, -1, -1)            # [B,N,D]
            gate = self.gate(cls).unsqueeze(-1)                  # [B,N,1] in [0,1]
            C = F.normalize(gate * C_loc + (1 - gate) * G, dim=-1)
        else:
            C = C_loc

        # 4) 计算 logits，并做温度自适应（目标方差匹配）
        logits_raw = torch.einsum('bid,bnd->bin', Kx, C)         # <Kx,C>
        # 调整缩放使得 logits 的 std 接近 target
        std = logits_raw.detach().std(dim=(1,2), keepdim=True).clamp(min=1e-3)
        scale = (self.target_logit_std / std).clamp(0.5, 5.0)    # 防极端
        logits = scale * logits_raw

        # 5) 软分配 + 均衡/熵下界
        P = torch.softmax(logits, dim=-1)
        if self.sinkhorn_iters > 0:
            P = self._sinkhorn(P, self.sinkhorn_iters)
        if self.mix_uniform_eps > 0:
            eps = self.mix_uniform_eps
            P = (1 - eps) * P + eps * (1.0 / self.N)

        return P, C

    
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
        self.alpha = nn.Parameter(torch.tensor(0.4))  # learnable min
        self.beta  = nn.Parameter(torch.tensor(0.8))   # learnable max
        base = torch.linspace(0, 1, steps=n_bins, device=self.alpha.device)
        centers = self.alpha + (self.beta - self.alpha) * base  # [n_bins]
        self.mu = nn.Parameter(centers)  # [n_bins]
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
        s = F.cosine_similarity(Kx, Kcls, dim=-1)    # [B, Nx]  ∈ [-1,1]

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
    def __init__(self, num_queries: int, num_clusters: int, temp_init: float = 1.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_clusters = num_clusters
        self.logits = nn.Parameter(torch.zeros(num_queries, num_clusters))
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temp_init)))
        # 用带偏置的初始化替换全零
        self.reset_router_bias(strength=2.0, neg=-2.0, jitter=1, shuffle=False, seed=None)

    @torch.no_grad()
    def reset_router_bias(self, strength: float = 6.0, neg: float = -2.0,
                          jitter: float = 0.1, shuffle: bool = False, seed: int | None = None):
        """
        用分工偏置初始化 logits：
          - 每个 query 先被“指派”到一个 cluster（轮转分配）
          - 被指派的 cluster 赋值为 `strength`，其它为 `neg`
          - 加少量随机抖动，避免同质
        注意：softmax 的温度会进一步影响实际尖锐度 → 有效差值是 (strength - neg)/temp
        """
        Q, N = self.num_queries, self.num_clusters
        if seed is not None:
            torch.manual_seed(seed)

        # 轮转分配：q -> (q mod N)
        # assign = torch.arange(Q) % N
        assign = torch.arange(Q) // (Q // N)  # 每个 query 分配到一个 cluster
        if shuffle:
            perm = torch.randperm(Q)
            assign = assign[perm]  # 打乱，让每次运行分配顺序不同
            # 如果你希望可重复，传入 seed

        logits = torch.full((Q, N), neg)
        logits[torch.arange(Q), assign] = strength

        if jitter > 0:
            logits = logits + jitter * torch.randn_like(logits)

        self.logits.copy_(logits)

    def forward(self, P: torch.Tensor):
        Ttemp = torch.clamp(self.log_temp.exp(), min=0.2, max=10.0)
        R_hat = F.softmax(self.logits / Ttemp, dim=-1)  # [Q,N]
        # R_hat = F.sigmoid(self.logits / Ttemp)  # [Q,N]
        Pt = P.transpose(1, 2).contiguous()             # [B,N,Nx]
        T = torch.einsum('qn,bnk->bqk', R_hat, Pt)      # [B,Q,Nx]
        T = T / (T.sum(dim=-1, keepdim=True) + 1e-6)
        # print(T[0, 0])  # 打印第一个 batch 的第一个 query 的前 10 个注意力值
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
                 alpha_init: float = 0.4, eps: float = 5e+2,
                 router_temp_init: float = 1.0,
                 mask_gain: float = 5.0):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_clusters = num_clusters
        self.nheads = nheads
        self.mask_gain = mask_gain

        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32))  # 防止 log(0)

        # Queries & attentions
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.self_attn = nn.MultiheadAttention(dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

        # Step 1: CLS->Prototypes (in K-space)
        # self.protos = CLSPrototypes(dim, num_clusters)
        self.binners = CLSBinner(num_clusters, sigma=0.05, use_sinkhorn=True, sinkhorn_iters=5)

        # Step 2: Query->Cluster Router
        self.router = QueryClusterRouter(num_queries, num_clusters, temp_init=router_temp_init)

        # Step 3: mask 混合强度 α（可退火）
        self.register_buffer("_alpha", torch.tensor(alpha_init, dtype=torch.float32))


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
    
    def masked_cross_attn(self, q, x, mha, attn_bias):  # attn_bias: [B,H,Q,Nx]
        W = mha.in_proj_weight; b = mha.in_proj_bias; D = mha.embed_dim
        Wq, Wk, Wv = W[:D,:], W[D:2*D,:], W[2*D:,:]
        bq = b[:D] if b is not None else None
        bk = b[D:2*D] if b is not None else None
        bv = b[2*D:] if b is not None else None
        Wo, bo = mha.out_proj.weight, mha.out_proj.bias
        H = mha.num_heads; d = D // H

        q_proj = F.linear(q, Wq, bq)               # [B,Q,D]
        k_proj = F.linear(x, Wk, bk)               # [B,Nx,D]
        v_proj = F.linear(x, Wv, bv)               # [B,Nx,D]

        def split(z):  # [B,L,D] -> [B,H,L,d]
            B,L,_ = z.shape
            return z.view(B, L, H, d).transpose(1, 2).contiguous()

        qh, kh, vh = split(q_proj), split(k_proj), split(v_proj)
        scores = torch.einsum('bhqd,bhkd->bhqk', qh, kh) / (d ** 0.5)
        scores = scores + attn_bias                # ★ 可微的偏置
        w = torch.softmax(scores, dim=-1)          # [B,H,Q,Nx]
        oh = torch.einsum('bhqk,bhkd->bhqd', w, vh)
        o  = oh.transpose(1, 2).reshape(q.size(0), q.size(1), D)
        o  = F.linear(o, Wo, bo)
        return o, w

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
        # P, C = self.protos(x, cls)           # P: [B,Nx,N], C: [B,N,D]
        P, s = self.binners(x, cls)                        # P: [B,Nx,N], s: [B,Nx]

        # ---- Step 2: Router(P) → T ----
        T, R_hat = self.router(P)                          # T: [B,Q,Nx], R_hat: [Q,N]

        # ---- Step 3: add mask to cross-attn logits ----
        alpha = float(self._alpha.item())
        if alpha > 0:
            log_bias = alpha * torch.log(T * self.eps)     # [B,Q,Nx]
            attn_mask = self.mask_gain * log_bias.unsqueeze(1).repeat(1, H, 1, 1)
            # attn_mask = torch.clamp(attn_mask, min=-5.0, max=5.0)  # 限制数值范围
        else:
            attn_mask = None

        # out, attn = self.cross_attn(
        #     q, x, x,
        #     need_weights=True,
        #     average_attn_weights=True,
        #     attn_mask=attn_mask
        # )                                                  # attn: [B,H,Q,Nx]
        out, attn = self.masked_cross_attn(q, x, self.cross_attn, attn_mask)  # [B,Q,D], [B,H,Q,Nx]
        out = self.norm_out(out)

        # # ---- Optional losses ----
        # losses = {}
        # total_aux_loss = torch.tensor(0.0, device=x.device)

        # if self.kl_weight > 0:
        #     # 让真实注意力靠近 T（多头平均后对齐）
        #     A = attn.mean(dim=1)                           # [B,Q,Nx]
        #     A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        #     kl = (A * (A.add(1e-6).log() - T.add(1e-6).log())).sum(dim=-1).mean()
        #     losses["kl_loss"] = self.kl_weight * kl
        #     total_aux_loss = total_aux_loss + losses["kl_loss"]

        # if self.div_weight > 0:
        #     # 多样性：惩罚 R_hat R_hat^T 的非对角
        #     G = R_hat @ R_hat.t()                          # [Q,Q]
        #     I = torch.eye(Q, device=G.device)
        #     div = ((G - I) ** 2).sum() / (Q ** 2)
        #     losses["div_loss"] = self.div_weight * div
        #     total_aux_loss = total_aux_loss + losses["div_loss"]

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

        return x, out, attn.detach(), attn_mask.detach(), q.detach(), R_hat.detach(), s.detach()



class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_clusters=8, num_layers=2, row_dim=32, slot_mask=True, mlp=False, hidden_layer=False):
        super().__init__()
        # self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        self.slot_mask = slot_mask
        in_dim = in_channels
        # self.norm_input = torch.nn.LayerNorm(in_dim)
        if slot_mask:
            self.boqs = torch.nn.ModuleList([
                BoQWithProtoMask(in_dim, num_queries, num_clusters=num_clusters, nheads=in_dim//64) for _ in range(num_layers)])
        else:
            self.boqs = torch.nn.ModuleList([
                BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])

        self.fc = torch.nn.Linear(num_layers*num_queries, num_layers * num_clusters)

    def forward(self, x, cls=None):
        # reduce input dimension using 3x3 conv when using ResNet
        # x = self.proj_c(x)
        # x = x.flatten(2).permute(0, 2, 1)
        # x = self.norm_input(x)
        
        outs = []
        attns = []
        masks = []
        queries = []
        R = []
        s = []
        for i in range(len(self.boqs)):
            if self.slot_mask:
                x, out, attn, mask, q, r, si = self.boqs[i](x, cls)
                masks.append(mask)
                queries.append(q)
                R.append(r)
                s.append(si)
            else:
                x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
        

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns, masks, queries, R, s