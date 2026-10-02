"""
实验 L：市场状态控制三频融合（Weighted / Gate / FiLM）。

在 ScaleCentricRatio4Idx（实验 A）基础上，用市场状态动态控制
三频（week/day/intra）的融合权重，替代原有的静态 scale_fusion(MLP)。

市场状态编码器复用 baseline_ratio4idx 中的 4 种方案：
  - _MarketStateFactorEncoder  (factor)
  - _MarketStyleEncoderMHA     (mha)
  - _MarketStyleEncoderSoftmax (softmax)
  - _RelationMatrixEncoder     (relation_matrix)

融合方式（3 种）：
  - weighted: market → 3 scalars → Softmax → 加权求和
  - gate:     market → 3*D → Sigmoid → 逐元素门控
  - film:     market → 6*D → scale + shift → 调制后平均

共 3 × 4 = 12 个实验变体。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_ratio4idx import (
    ScaleCentricRatio4Idx,
    IntradayCompressor,
    _make_encoder,
    _MarketStateFactorEncoder,
    _MarketStyleEncoderMHA,
    _MarketStyleEncoderSoftmax,
    _RelationMatrixEncoder,
)


class _ScaleCentricFusionGateBase(ScaleCentricRatio4Idx):
    """
    基类：用市场状态动态控制三频融合（week/day/intra 的融合权重）。

    替代原有的静态 scale_fusion(MLP)，在每个时间步用 market_state 生成
    融合权重，动态调节三个尺度的贡献。

    子类只需设置 _market_encoder_key 和 _fusion_mode：
      - _fusion_mode='weighted'  (Weighted):
          market → Linear(market_dim→3) → Softmax → 加权求和
          每个时间步学习 3 个 scalar 权重
      - _fusion_mode='gate'      (Gate):
          market → Linear(market_dim→3*D) → Sigmoid → 逐元素门控
          每个时间步对每个维度独立调节
      - _fusion_mode='film'      (FiLM):
          market → Linear(market_dim→6*D) → scale + shift
          每个时间步可放大/抑制 + 偏置

    _ENCODER_MAP = { 'factor', 'mha', 'softmax', 'relation_matrix' }

    model_type 格式: 'scale_centric_fusion_{mode}_{encoder}'
    """
    _market_encoder_key = None
    _fusion_mode = 'weighted'  # 'weighted' | 'gate' | 'film'

    _ENCODER_MAP = {
        'factor': _MarketStateFactorEncoder,
        'mha': _MarketStyleEncoderMHA,
        'softmax': _MarketStyleEncoderSoftmax,
        'relation_matrix': _RelationMatrixEncoder,
    }

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        encoder_cls = self._ENCODER_MAP[self._market_encoder_key]
        if encoder_cls is _MarketStateFactorEncoder:
            self.market_encoder = encoder_cls(
                macro_dim=macro_dim, hidden_dim=16,
                dropout_rate=dropout_rate)
            market_dim = 16
        else:
            self.market_encoder = encoder_cls(
                macro_dim=macro_dim, hidden_dim=per_scale_dim,
                dropout_rate=dropout_rate)
            market_dim = per_scale_dim

        if self._fusion_mode == 'weighted':
            self.fusion_net = nn.Sequential(
                nn.Linear(market_dim, 3),
                nn.Softmax(dim=-1),
            )
        elif self._fusion_mode == 'gate':
            self.fusion_net = nn.Sequential(
                nn.Linear(market_dim, 3 * per_scale_dim),
                nn.Sigmoid(),
            )
        elif self._fusion_mode == 'film':
            self.fusion_net = nn.Linear(market_dim, 6 * per_scale_dim)
        else:
            raise ValueError(f"Unknown fusion mode: {self._fusion_mode}")

    def _fuse_scales(self, week_repr, day_repr, intra_repr, market_state):
        """
        week_repr, day_repr, intra_repr: (B, 60, D)
        market_state: (B, 60, market_dim)
        → fused: (B, 60, D)
        """
        B, T, D = week_repr.shape

        if self._fusion_mode == 'weighted':
            weights = self.fusion_net(market_state)  # (B, 60, 3)
            stacked = torch.stack(
                [week_repr, day_repr, intra_repr], dim=-2)  # (B, 60, 3, D)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # (B, 60, D)

        elif self._fusion_mode == 'gate':
            gate = self.fusion_net(market_state)  # (B, 60, 3*D)
            g_w, g_d, g_i = gate.chunk(3, dim=-1)  # 各 (B, 60, D)
            fused = (week_repr * g_w + day_repr * g_d + intra_repr * g_i) / 3

        elif self._fusion_mode == 'film':
            film = self.fusion_net(market_state)  # (B, 60, 6*D)
            s_w, s_d, s_i, sh_w, sh_d, sh_i = film.chunk(6, dim=-1)
            week_mod  = week_repr  * (1 + s_w) + sh_w
            day_mod   = day_repr   * (1 + s_d) + sh_d
            intra_mod = intra_repr * (1 + s_i) + sh_i
            fused = (week_mod + day_mod + intra_mod) / 3

        return fused

    def _compute_repr(self, daily, weekly, intraday, macro):
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

        # 市场状态编码
        market_state = self.market_encoder(macro)  # (B, 60, market_dim)

        # 市场状态控制三频融合（替代静态 scale_fusion）
        fused = self._fuse_scales(week_repr, day_repr, intra_repr, market_state)

        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 实验 L1: Weighted 融合 × 4 种市场编码器
# ============================================================

class ScaleCentricFusionWeightedFactor(_ScaleCentricFusionGateBase):
    """
    Weighted 融合 + 市场因子编码。
    model_type: 'scale_centric_fusion_weighted_factor'
    """
    _market_encoder_key = 'factor'
    _fusion_mode = 'weighted'


class ScaleCentricFusionWeightedMHA(_ScaleCentricFusionGateBase):
    """
    Weighted 融合 + MHA 市场编码。
    model_type: 'scale_centric_fusion_weighted_mha'
    """
    _market_encoder_key = 'mha'
    _fusion_mode = 'weighted'


class ScaleCentricFusionWeightedSoftmax(_ScaleCentricFusionGateBase):
    """
    Weighted 融合 + Softmax 市场编码。
    model_type: 'scale_centric_fusion_weighted_softmax'
    """
    _market_encoder_key = 'softmax'
    _fusion_mode = 'weighted'


class ScaleCentricFusionWeightedRelationMatrix(_ScaleCentricFusionGateBase):
    """
    Weighted 融合 + 关系矩阵编码。
    model_type: 'scale_centric_fusion_weighted_relation_matrix'
    """
    _market_encoder_key = 'relation_matrix'
    _fusion_mode = 'weighted'


# ============================================================
# 实验 L2: Gate 融合 × 4 种市场编码器
# ============================================================

class ScaleCentricFusionGateFactor(_ScaleCentricFusionGateBase):
    """Gate 融合 + 市场因子编码。model_type: 'scale_centric_fusion_gate_factor'"""
    _market_encoder_key = 'factor'
    _fusion_mode = 'gate'


class ScaleCentricFusionGateMHA(_ScaleCentricFusionGateBase):
    """Gate 融合 + MHA 市场编码。model_type: 'scale_centric_fusion_gate_mha'"""
    _market_encoder_key = 'mha'
    _fusion_mode = 'gate'


class ScaleCentricFusionGateSoftmax(_ScaleCentricFusionGateBase):
    """Gate 融合 + Softmax 市场编码。model_type: 'scale_centric_fusion_gate_softmax'"""
    _market_encoder_key = 'softmax'
    _fusion_mode = 'gate'


class ScaleCentricFusionGateRelationMatrix(_ScaleCentricFusionGateBase):
    """Gate 融合 + 关系矩阵编码。model_type: 'scale_centric_fusion_gate_relation_matrix'"""
    _market_encoder_key = 'relation_matrix'
    _fusion_mode = 'gate'


# ============================================================
# 实验 L3: FiLM 融合 × 4 种市场编码器
# ============================================================

class ScaleCentricFusionFiLMFactor(_ScaleCentricFusionGateBase):
    """FiLM 融合 + 市场因子编码。model_type: 'scale_centric_fusion_film_factor'"""
    _market_encoder_key = 'factor'
    _fusion_mode = 'film'


class ScaleCentricFusionFiLMMHA(_ScaleCentricFusionGateBase):
    """FiLM 融合 + MHA 市场编码。model_type: 'scale_centric_fusion_film_mha'"""
    _market_encoder_key = 'mha'
    _fusion_mode = 'film'


class ScaleCentricFusionFiLMSoftmax(_ScaleCentricFusionGateBase):
    """FiLM 融合 + Softmax 市场编码。model_type: 'scale_centric_fusion_film_softmax'"""
    _market_encoder_key = 'softmax'
    _fusion_mode = 'film'


class ScaleCentricFusionFiLMRelationMatrix(_ScaleCentricFusionGateBase):
    """FiLM 融合 + 关系矩阵编码。model_type: 'scale_centric_fusion_film_relation_matrix'"""
    _market_encoder_key = 'relation_matrix'
    _fusion_mode = 'film'
