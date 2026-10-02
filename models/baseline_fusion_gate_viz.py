"""
可视化专用模型副本（不改动原始模型文件 baseline_fusion_gate.py）。

在 `ScaleCentricFusionGateMHA`（实验 L2 Gate 融合 + MHA 市场编码）基础上，
新增一个子类 `ScaleCentricFusionGateMHAViz`，仅重写 forward，
在返回 logits 的同时额外返回中间量：

    return logits, {
        "ratio":        (B, 60, 12)  相对特征（4指数 × return/amp/vol，log1p 压缩）
        "market_state": (B, 60, D)   市场状态编码器输出 S（创新点2）
        "gate_week":    (B, 60, D)   周频尺度逐元素门控权重（创新点3）
        "gate_day":     (B, 60, D)   日频尺度逐元素门控权重
        "gate_intra":   (B, 60, D)   分时尺度逐元素门控权重
    }

画图时对最后一维（通道 D）取平均即可得到：
    s_t       = mean_j S[t, j]          → 60 天市场状态曲线
    g_t^scale = mean_j G[t, j]          → 60 天三尺度门控曲线

使用方法：
    - 在 models/__init__.py 中用模型类型 'scale_centric_fusion_gate_mha_viz' 构建；
    - 权重与原始 `ScaleCentricFusionGateMHA` 完全兼容（同一套参数），
      因此可以直接 load_state_dict 原始 checkpoint，无需重新训练。
"""

import torch
import torch.nn as nn

from .baseline_fusion_gate import (
    ScaleCentricFusionGateMHA,
    _ScaleCentricFusionGateBase,
)


class ScaleCentricFusionGateMHAViz(ScaleCentricFusionGateMHA):
    """
    可视化子类：Gate 融合 + MHA 市场编码。

    权重结构与 ScaleCentricFusionGateMHA 完全一致，仅 forward 额外吐出中间量。
    继承关系: ScaleCentricFusionGateMHAViz → ScaleCentricFusionGateMHA
              → _ScaleCentricFusionGateBase → ScaleCentricRatio4Idx
    """

    def _compute_repr_viz(self, daily, weekly, intraday, macro):
        """在 _compute_repr 基础上额外记录 ratio / market_state / 三尺度 gate。

        返回 (repr_vec, viz_dict)。与原始 `_compute_repr` 计算结果完全一致，
        因为 gate 融合公式就是原始 `_fuse_scales` 中 gate 分支的复刻。
        """
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        stock_feat = self.stock_proj(daily_flat)
        ratio_feat = self.ratio_proj(ratio_feats)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        # 市场状态编码（创新点2 的 S）
        market_state = self.market_encoder(macro)  # (B, 60, market_dim)

        # Gate 融合（创新点3 的三尺度门控）。对应原始 _fuse_scales 的 'gate' 分支：
        #   gate = fusion_net(market_state) → (B, 60, 3*D)
        #   g_w, g_d, g_i = gate.chunk(3, dim=-1)
        #   fused = (week_repr*g_w + day_repr*g_d + intra_repr*g_i) / 3
        gate = self.fusion_net(market_state)
        g_w, g_d, g_i = gate.chunk(3, dim=-1)

        # 保持数值与原始 forward 完全一致
        fused = (week_repr * g_w + day_repr * g_d + intra_repr * g_i) / 3
        fused_out, _ = self.temporal_encoder(fused)

        viz = {
            'ratio': ratio_feats.detach(),
            'market_state': market_state.detach(),
            'gate_week': g_w.detach(),
            'gate_day': g_d.detach(),
            'gate_intra': g_i.detach(),
        }
        return fused_out[:, -1, :], viz

    def forward(self, daily, weekly, intraday, macro):
        repr_vec, viz = self._compute_repr_viz(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        return logits, viz
