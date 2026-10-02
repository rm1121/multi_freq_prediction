"""
ms-Mamba: Multi-scale Mamba for Time-Series Forecasting
(Karadag et al., Neurocomputing 2026) 的核心层适配实现。

官方仓库: https://github.com/meric-karadag/ms-mamba
原实现依赖 mamba-ssm + causal-conv1d（仅支持 NVIDIA CUDA），
此处使用纯 PyTorch 的选择性扫描（selective scan, S6）实现：
  - 数学上与官方 selective scan 等价（顺序扫描，序列长度 60 性能足够）
  - 保证在 CPU / Apple MPS / CUDA 上行为一致
  - 多尺度机制（不同采样率 Δ_i 的并行 Mamba 分支 + 平均融合）完整保留

多尺度原理：SSM 的采样率 Δ 是块的内部参数而非输入下采样，
因此各分支输入输出形状一致，输出可直接平均融合，无需时间对齐。
分支 i 的 Δ_i = base_dt * 2^i（对应论文 fixed/learnable 时间尺度设定，
Δ₁ 可学习，其余为倍数关系）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """softplus 的逆，用于把期望的初始 Δ 换算到 dt_proj.bias。"""
    return y + math.log(-math.expm1(-y))


class MambaBlock(nn.Module):
    """单个 Mamba (S6) 块，纯 PyTorch 顺序选择性扫描实现。

    dt_init: 初始采样率 Δ（softplus 前的 bias 初始化），
    多尺度分支通过不同 dt_init 区分时间尺度。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_init=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Mamba 官方的 dt_proj 权重初始化
        dt_init_std = self.dt_rank ** -0.5 * dt_init
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # 多尺度：不同分支用不同 dt_init（≈不同采样率 Δ_i）
        with torch.no_grad():
            self.dt_proj.bias.fill_(_inv_softplus(dt_init))

        # A: (d_inner, d_state)，负值保证稳定，A_log 参数化
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)                 # 各 (B, L, d_inner)

        # 因果深度卷积 + SiLU
        x = x.transpose(1, 2)                       # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = F.silu(x).transpose(1, 2)               # (B, L, d_inner)

        # 输入依赖的 Δ / B / C（选择性）
        x_dbl = self.x_proj(x)                      # (B, L, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))           # (B, L, d_inner)

        A = -torch.exp(self.A_log)                  # (d_inner, d_state)

        # 顺序选择性扫描
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                         # (B, d_inner)
            dA = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            h = dA * h + (dt_t * x[:, t]).unsqueeze(-1) * Bp[:, t].unsqueeze(1)
            ys.append((h * Cp[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)                  # (B, L, d_inner)

        # 跳跃连接 + 门控
        y = y + x * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class HierarchicalMambaBlock(nn.Module):
    """ms-Mamba 多尺度块：并行多个不同 Δ 的 Mamba 分支，输出平均融合。

    各分支接收相同输入（Δ 是 SSM 内部参数），输出形状一致，
    按论文方式直接取平均；残差连接 + LayerNorm（pre-norm）。
    """

    def __init__(self, d_model, num_scales=4, d_state=16, d_conv=4, expand=2,
                 dropout_rate=0.1, base_dt=0.01):
        super().__init__()
        self.branches = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                       dt_init=base_dt * (2 ** i))
            for i in range(num_scales)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        h = self.norm(x)
        out = 0.0
        for branch in self.branches:
            out = out + branch(h)
        out = out / len(self.branches)              # 论文的平均融合
        return x + self.dropout(out)
