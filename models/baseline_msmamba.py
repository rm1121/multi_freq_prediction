"""
Baseline: ms-Mamba (Multi-scale Mamba for Time-Series Forecasting,
Karadag et al., Neurocomputing 2026)。model_type: 'baseline_msmamba'

官方仓库: https://github.com/meric-karadag/ms-mamba
适配方式（与其他对比 baseline 一致）：
  三尺度输入 -> align_scales 拼接 (B, 60, 8+8+intra_hidden)
  -> Linear 嵌入 -> depth 个 HierarchicalMambaBlock（多尺度 Δ 并行分支 + 平均融合）
  -> Mean + Last 双向池化 -> 统一分类头 -> (logits, proj)

多尺度 Mamba 块为纯 PyTorch 实现（见 tslib_models/mamba_layers.py），
无 mamba-ssm 依赖，CPU/MPS/CUDA 行为一致。
"""

import torch
import torch.nn as nn

from .tslib_models.wrappers import (IntradayCompressor, align_scales,
                                     ClassificationHead, ProjectionHead)
from .tslib_models.mamba_layers import HierarchicalMambaBlock


class MSMambaClassifier(nn.Module):
    """ms-Mamba baseline。model_type: 'baseline_msmamba'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.5, intra_hidden=32, d_model=64, depth=2,
                 num_scales=4, d_state=16, base_dt=0.01, **kwargs):
        super().__init__()
        # 美股数据管线的日/周/分时输入均为 8 维；align_scales 同样按 8 维对齐。
        enc_in = 8 + 8 + intra_hidden
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        self.embed = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList([
            HierarchicalMambaBlock(d_model, num_scales=num_scales,
                                   d_state=d_state, dropout_rate=dropout_rate,
                                   base_dt=base_dt)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(2 * d_model, hidden_dim,
                                       num_classes, dropout_rate)
        self.projection = ProjectionHead(2 * d_model, dropout_rate)

    def _encode(self, x):
        """x: (B, 60, enc_in) -> (B, 2*d_model)"""
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)                             # (B, 60, d_model)
        return torch.cat([h.mean(dim=1), h[:, -1, :]], dim=-1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj
