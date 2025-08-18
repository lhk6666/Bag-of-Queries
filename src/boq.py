# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math

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
    def __init__(
        self,
        in_dim: int,
        num_queries: int,
        nheads: int = 8,
        temperature_init: float = 0.1,
        alpha_init: float = 0.0,
        eps: float = 0.02,
        use_cls_router: bool = False,
        detach_gate_inputs: bool = True,
        detach_gate_weights: bool = True,
        gate_activation: str = "sigmoid",   # or "softmax"
        entropy_reg: float = 0.0,
        topk: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.nheads = nheads

        # Queries & attentions
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim, elementwise_affine=True)

        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim, elementwise_affine=True)

        # Optional CLS router (tiny MLP) to adapt CLS before gating
        self.use_cls_router = use_cls_router
        if use_cls_router:
            self.cls_router = nn.Sequential(
                nn.Linear(in_dim, in_dim * 4),
                nn.GELU(),
                nn.Linear(in_dim * 4, in_dim),
            )
        else:
            self.cls_router = nn.Identity()

        # Gate temperature (learnable, clamped to >=1 in compute)
        self.temp = nn.Parameter(torch.ones(1) * temperature_init, requires_grad=True)

        # Mix for CLS→patch bias injected into cross-attn logits
        self.register_buffer("_alpha", torch.tensor(alpha_init, dtype=torch.float32))  # you can anneal via set_alpha()
        self.eps = eps

        # Detach strategy for stability
        self.detach_gate_inputs = detach_gate_inputs
        self.detach_gate_weights = detach_gate_weights

        # Gate behavior
        assert gate_activation in ("sigmoid", "softmax")
        self.gate_activation = gate_activation
        self.topk = topk
        self.entropy_reg = float(entropy_reg)  # coefficient; returned but not applied here

    # --------- helpers ---------
    @staticmethod
    def _split_qkv_from_mha(mha: nn.MultiheadAttention):
        """Return (Wq, bq, Wk, bk, Wv, bv) from an nn.MultiheadAttention as DETACHED views."""
        W = mha.in_proj_weight.detach()          # [3D, D]
        b = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None
        D = mha.embed_dim
        Wq = W[:D, :]
        Wk = W[D:2*D, :]
        Wv = W[2*D:, :]
        bq = b[:D]       if b is not None else None
        bk = b[D:2*D]    if b is not None else None
        bv = b[2*D:]     if b is not None else None
        return Wq, bq, Wk, bk, Wv, bv

    def set_alpha(self, alpha: float):
        """Set mixing weight for CLS→patch bias (0~1). Use for curriculum/annealing."""
        alpha = max(0.0, min(1.0, float(alpha)))
        self._alpha.data = torch.tensor(alpha, dtype=torch.float32, device=self._alpha.device)

    # --------- CLS→patch importance in K-space ---------
    def _cls_patch_importance(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """
        Compute g in K-space: g[j] ∝ exp( cos(Kcls, Kx[j]) / T )  (length Nx).
        Detaches inputs/weights for stability as configured.
        """
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)  # [B,1,D]
        cls = self.cls_router(cls)  # optional small adaptation

        # Project to K space (reuse cross_attn's Wk)
        Wq, bq, Wk, bk, _, _ = self._split_qkv_from_mha(self.cross_attn)

        x_in   = x.detach()   if self.detach_gate_inputs   else x
        cls_in = cls.detach() if self.detach_gate_inputs   else cls
        Wk_in  = Wk.detach()  if self.detach_gate_weights  else self.cross_attn.in_proj_weight[self.in_dim:2*self.in_dim, :]
        bk_in  = bk.detach()  if (self.detach_gate_weights and bk is not None) else (self.cross_attn.in_proj_bias[self.in_dim:2*self.in_dim] if self.cross_attn.in_proj_bias is not None else None)

        Kx   = F.linear(x_in,   Wk_in, bk_in)   # [B,Nx,D]
        Kcls = F.linear(cls_in, Wk_in, bk_in)   # [B,1, D]
        Kx   = F.normalize(Kx,   dim=-1)
        Kcls = F.normalize(Kcls, dim=-1)

        sim = torch.einsum('bid,bjd->bij', Kcls, Kx).squeeze(1)  # [B,Nx]
        T = torch.clamp(self.temp, min=1.0)
        g = torch.softmax(sim / (T + 1e-6), dim=-1)              # [B,Nx]
        return g  # sum=1 per batch

    # --------- q↔cls gate in Q/K space ---------
    def _query_gate_qcls(self, q_raw: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """
        Gate per query from cosine(Qq, Kcls) in cross-attn's Q/K space.
        Uses q BEFORE LayerNorm (avoid direction distortion).
        Returns [B, Nq].
        """
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)
        cls = self.cls_router(cls)

        Wq, bq, Wk, bk, _, _ = self._split_qkv_from_mha(self.cross_attn)

        q_in   = q_raw.detach() if self.detach_gate_inputs  else q_raw
        cls_in = cls.detach()   if self.detach_gate_inputs  else cls
        Wq_in  = Wq.detach()    if self.detach_gate_weights else self.cross_attn.in_proj_weight[:self.in_dim, :]
        bq_in  = bq.detach()    if (self.detach_gate_weights and bq is not None) else (self.cross_attn.in_proj_bias[:self.in_dim] if self.cross_attn.in_proj_bias is not None else None)
        Wk_in  = Wk.detach()    if self.detach_gate_weights else self.cross_attn.in_proj_weight[self.in_dim:2*self.in_dim, :]
        bk_in  = bk.detach()    if (self.detach_gate_weights and bk is not None) else (self.cross_attn.in_proj_bias[self.in_dim:2*self.in_dim] if self.cross_attn.in_proj_bias is not None else None)

        Qq   = F.linear(q_in,   Wq_in, bq_in)   # [B,Nq,D]
        Kcls = F.linear(cls_in, Wk_in, bk_in)   # [B,1, D]

        Qq   = F.normalize(Qq,   dim=-1)
        Kcls = F.normalize(Kcls, dim=-1)

        sim = torch.einsum('bid,bjd->bij', Qq, Kcls).squeeze(-1)  # [B,Nq]
        T = torch.clamp(self.temp, min=0.01)
        score = sim / (T + 1e-6)

        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(score)  # independent switches per query
        else:
            gate = torch.softmax(score, dim=-1)  # relative allocation across queries

        # optional sparsify
        if (self.topk is not None) and (self.gate_activation == "softmax"):
            idx  = gate.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(gate).scatter(1, idx, 1.0)
            gate = gate * mask
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-6)

        return gate  # [B,Nq]

    # --------- forward ---------
    def forward(self, x: torch.Tensor, cls: torch.Tensor | None = None):
        """
        x:   [B, Nx, D]    patch tokens from backbone
        cls: [B, D] or [B,1,D] global token (optional but recommended)

        Returns:
          x (passthrough),
          out:   [B, Nq, D]        final query outputs (after gating)
          attn:  [B, H, Nq, Nx]    per-head attention (detached, after bias)
          mask:  [B, Nq, 1]        query gates as mask
          q:     [B, Nq, D]        normalized queries used by cross-attn
          aux:   dict with diagnostics (g_patch, gate_q, gate_entropy, alpha_used, ...)
        """
        B, Nx, D = x.shape
        device = x.device
        H = self.nheads

        cls=cls.unsqueeze_(1)
        cosine = torch.nn.CosineSimilarity(dim=-1)
        cls_patch_sim = cosine(cls, x)
        print(cls_patch_sim[0])

        # 0) Optional CLS→patch bias g computed BEFORE cross-attn (so we can bias logits once)
        g = None
        alpha = float(self._alpha.item())
        if (cls is not None) and (alpha > 0.0):
            g = self._cls_patch_importance(x, cls)                     # [B,Nx]
            # bias to logits: alpha * log(g + eps), broadcast to [B*H, Nq, Nx]
            log_bias = torch.log(g + self.eps) * alpha                 # [B,Nx]
            # expand to [B,H,Nq,Nx] later; build after q sizes known

        # 1) Build queries (self-attn among queries)
        q0 = self.queries.repeat(B, 1, 1)                              # [B,Nq,D]
        q_sa, _ = self.self_attn(q0, q0, q0, need_weights=False)
        q_raw = q0 + q_sa                                              # >>> for gate (pre-LN)
        q = self.norm_q(q_raw)                                         # >>> for cross-attn

        # 2) Cross-attn with optional per-batch bias from g
        attn_mask = None
        if (cls is not None) and (alpha > 0.0):
            # Build [B*H, Nq, Nx] additive bias mask
            log_bias = torch.log(g + self.eps)                         # [B,Nx]
            log_bias = (alpha * log_bias).unsqueeze(1)                 # [B,1,Nx]
            log_bias = log_bias.expand(B, q.shape[1], Nx)              # [B,Nq,Nx]
            # repeat per head as MHA expects [B*H, Nq, Nx]
            attn_mask = log_bias.repeat_interleave(H, dim=0)           # [B*Nq?] careful: expand heads differently
            # Correct shape: we need [B*H, Nq, Nx]
            # Expand along head by reshaping:
            attn_mask = log_bias.unsqueeze(1).repeat(1, H, 1, 1)       # [B,H,Nq,Nx]
            attn_mask = attn_mask.reshape(B * H, q.shape[1], Nx)       # [B*H,Nq,Nx]

        # Cross-attn forward (returns per-head weights)
        out, attn = self.cross_attn(
            q, x, x,
            need_weights=True,
            attn_mask=attn_mask               # additive bias to logits
        )
        # out: [B,Nq,D], attn: [B,H,Nq,Nx]
        out = self.norm_out(out)

        # 3) Query-wise gate from q↔cls in Q/K space (independent of step 2)
        if cls is not None:
            gate_q = self._query_gate_qcls(q_raw, cls)                 # [B,Nq]
        else:
            gate_q = torch.ones(B, out.size(1), device=device)

        mask = gate_q.unsqueeze(-1)                                    # [B,Nq,1]
        out = out * mask                                               # scale each query output

        # 4) Diagnostics & optional entropy loss (returned, not summed here)
        aux = {}
        aux["gate_q"] = gate_q
        aux["alpha_used"] = alpha
        aux["g_patch"] = g                                             # [B,Nx] or None
        aux["attn"] = attn.detach()                                    # [B,H,Nq,Nx]

        if self.entropy_reg > 0.0:
            # entropy over queries per batch item
            p = torch.clamp(gate_q, min=1e-12, max=1-1e-12)
            if self.gate_activation == "softmax":
                ent = -(p * p.log()).sum(dim=-1).mean()
            else:
                # sigmoid case: compute Bernoulli entropy and average
                ent = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
            aux["gate_entropy"] = ent
            aux["gate_entropy_loss"] = self.entropy_reg * (-ent)  # encourage higher entropy (less collapse)
        else:
            aux["gate_entropy"] = None
            aux["gate_entropy_loss"] = torch.tensor(0.0, device=device)

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