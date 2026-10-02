"""
无注意力机制的 Baseline 模型：

baseline2  FeatureCentricSimple — 先跨尺度(日/周/分时)融合，再跨特征融合
baseline3  ScaleCentricSimple   — 先跨特征融合，再跨尺度(日/周/分时)融合
baseline4  BothCentricSimple    — 同时保留两路信息，拼接后分类

所有模型均使用可配置的序列编码器（GRU/LSTM/TCN），MLP 做融合，不使用注意力机制。
输入接口与 seqnets.py 一致：model(daily, weekly, intraday)。

改进：引入 IntradayCompressor，将每天 13 根分时 bar 显式压缩为一个日级向量，
     再与日/周数据在 60 步天轴上对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seqnets import TCNEncoder
import math


# ============================================================
# 日内压缩器：13 根 bar → 1 个日级向量
# ============================================================
class IntradayCompressor(nn.Module):
    """将每个交易日的 13 根分时 bar 压缩为一个日级摘要向量。

    对于 per-feature 场景: 输入 (B, 60, 8) → 输出 (B, 60, hidden_dim)
    对于 per-scale 场景:  输入 (B, 60, 13, F) → 输出 (B, 60, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim,
                          num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: (B, 60, 8) 或 (B, 60, 13, F)
        → (B, 60, hidden_dim)
        """
        B = x.size(0)
        if x.dim() == 3:
            # Per-feature: (B, 60, 8) → (B*60, 8, 1)
            x = x.reshape(B * 60, 13, 1)
        else:
            # Per-scale: (B, 60, 13, F) → (B*60, 8, F)
            x = x.reshape(B * 60, 13, x.size(-1))

        out, _ = self.gru(x)          # (B*60, 8, hidden_dim)
        out = out[:, -1, :]            # (B*60, hidden_dim)
        out = self.dropout(out)
        return out.reshape(B, 60, -1)  # (B, 60, hidden_dim)


class TCNSeqEncoder(nn.Module):
    """TCN 包装器：对外接口与 GRU/LSTM 保持一致。
    forward(x) → (output, None)
    x: (B, T, F) → output: (B, T, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, num_levels=4,
                 kernel_size=3, dropout_rate=0.3):
        super().__init__()
        self.tcn = TCNEncoder(input_dim, hidden_dim, num_levels,
                              kernel_size, dropout_rate)

    def forward(self, x):
        # x: (B, T, F) → tcn: (B, F, T) internally → (B, hidden_dim, T)
        out = self.tcn(x)            # (B, hidden_dim, T)
        out = out.transpose(1, 2)    # (B, T, hidden_dim)
        return out, None             # 兼容 out, _ = encoder(x) 的调用


def _make_encoder(encoder_type, input_size, hidden_size, num_layers,
                  dropout, kernel_size=3, num_levels=4):
    """创建序列编码器，支持 GRU / LSTM / TCN。

    num_layers:  GRU/LSTM 的堆叠层数
    num_levels:  TCN 的扩张级数（独立于 num_layers，防止 GRU 调优时误伤 TCN）
    """
    if encoder_type == 'gru':
        return nn.GRU(input_size, hidden_size, num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0)
    elif encoder_type == 'lstm':
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
    elif encoder_type == 'tcn':
        return TCNSeqEncoder(input_size, hidden_size, num_levels=num_levels,
                             kernel_size=kernel_size, dropout_rate=dropout)
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")


# ============================================================
# Baseline 2: 先跨尺度 → 再跨特征（Feature-Centric，无注意力）
# ============================================================
class FeatureCentricSimple(nn.Module):
    """
    对每个特征 f ∈ {0..8}：
      1. 提取日频/周频序列，分时保留原始 13 bar 值
      2. 分别用三个序列编码器提取尺度表示
      3. MLP 融合三个尺度表示 → 该特征的融合嵌入
    8 个特征嵌入取平均池化后送入分类头。

    注：per-feature 场景不适用 IntradayCompressor（input_dim=1 的 GRU 无意义），
        分时数据保持原始 (B, 60, 8) 送入编码器。
    """

    def __init__(self, per_feat_dim=64, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=None):
        super().__init__()

        # 三条共享编码器：在 8 个特征上复用
        mk = lambda sz: _make_encoder(encoder_type, sz, per_feat_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(1)   # (B, 60, 1)  单特征 × 60 天
        self.week_encoder  = mk(1)   # (B, 60, 1)  单特征 × 60 天（周 broadcast）
        self.intra_encoder = mk(8)   # (B, 60, 8)  单特征 × 13 bar（原始值）

        # 跨尺度融合 MLP：3 个尺度表示拼接 → 单个特征嵌入
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_feat_dim, per_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_feat_dim),
            nn.Linear(per_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_scale(self, x, encoder):
        """编码一个尺度的一条序列，取最后时刻隐状态"""
        out, _ = encoder(x)           # (B, T, per_feat_dim)
        return out[:, -1, :]          # (B, per_feat_dim)

    def _compute_repr(self, daily, weekly, intraday):
        """
        计算跨特征融合后的全局表示。
        daily:    (B, 12, 5, 8)
        weekly:   (B, 1, 12, 8)
        intraday: (B, 60, 13, 8)
        → 返回 (B, per_feat_dim)
        """
        B = daily.size(0)
        feature_reprs = []

        for f in range(8):
            # 日频：真实 60 天
            day_f = daily[:, :, :, f].reshape(B, 60, 1)

            # 周频：12 周 broadcast 到 60 天，形成「周常量 + 日变化」
            week_f = (weekly[:, 0, :, f]
                      .repeat_interleave(5, dim=1)
                      .unsqueeze(-1))                  # (B, 60, 1)

            # 分时：原始 13 bar 值，直接送入编码器
            intra_f = intraday[:, :, :, f]              # (B, 60, 8)

            # 编码三尺度
            d_enc = self._encode_scale(day_f,   self.day_encoder)
            w_enc = self._encode_scale(week_f,  self.week_encoder)
            i_enc = self._encode_scale(intra_f, self.intra_encoder)

            # 融合三尺度
            cat_enc = torch.cat([w_enc, d_enc, i_enc], dim=-1)   # (B, 3*per_feat_dim)
            feat_repr = self.scale_fusion(cat_enc)                # (B, per_feat_dim)
            feature_reprs.append(feat_repr)

        # 跨特征融合：平均池化
        all_feats = torch.stack(feature_reprs, dim=1)    # (B, 8, per_feat_dim)
        return all_feats.mean(dim=1)                     # (B, per_feat_dim)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Baseline 3: 先跨特征 → 再跨尺度（Scale-Centric，无注意力）
# ============================================================
class ScaleCentricSimple(nn.Module):
    """
    真正的 Scale-Centric 两阶段设计：

    Stage 1 — 先跨特征（尺度内）：每个尺度用独立编码器将 8 个特征融合，
             保留 60 步完整时序输出，而非静态 last-hidden。
    Stage 2 — 再跨尺度（尺度间）：对三尺度表示做逐时间步的跨尺度融合，
             再过时序编码器捕捉尺度间的动态交互。

    与 MultiFreqGRU 的本质区别：
    - MultiFreqGRU: 8维原始特征 cat → 单 GRU（特征融合 & 尺度交互混在一起）
    - 本模型:      尺度内先学会融合特征 → 再让尺度间在完整时序上对话
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：13 bar × 9 特征 → 日级摘要
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（输出完整时序）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)              # (B,60,9)  → (B,60,d)
        self.week_encoder  = mk(8)              # (B,60,9)  → (B,60,d)
        self.intra_encoder = mk(intra_hidden)   # (B,60,intra_hidden) → (B,60,d)

        # Stage 2: 逐时间步跨尺度融合
        # 在每个交易日 t，三尺度表示拼接后过 MLP → 一个 t 时刻的融合向量
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 对比投影头（如需 CE + SupCon 混合训练，设 loss.cl_weight > 0 即可）
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _encode_full(self, x, encoder):
        """编码完整时序，返回 (B, T, hidden)，不截断 last。"""
        out, _ = encoder(x)
        return out                                  # (B, T, per_scale_dim)

    def _compute_repr(self, daily, weekly, intraday):
        """
        daily:    (B, 12, 5, 8)
        weekly:   (B, 1, 12, 8)
        intraday: (B, 60, 13, 8)
        → 返回 (B, per_scale_dim)
        """
        B = daily.size(0)

        # === Stage 1: 尺度内特征融合（保留完整 60 步时序） ===
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)  # (B, 60, 8)
        intra_daily = self.intra_compressor(intraday)              # (B, 60, intra_hidden)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)   # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)  # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)  # (B, 60, d)

        # === Stage 2: 逐时间步跨尺度融合 + 时序编码 ===
        # 每个时间步 t: [week_repr(t) | day_repr(t) | intra_repr(t)] → 融合向量
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)  # (B, 60, 3*d)
        fused = self.scale_fusion(stacked)                               # (B, 60, d)

        fused_out, _ = self.temporal_encoder(fused)   # (B, 60, d)
        return fused_out[:, -1, :]                        # (B, d)

    def forward(self, daily, weekly, intraday):
        """返回 (logits, proj) — proj 用于 SupCon 对比损失。"""
        repr_vec = self._compute_repr(daily, weekly, intraday)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj



# ============================================================
# Baseline 3-Gated: 尺度门控 + 残差融合（Scale-Centric改进版）
# ============================================================
class ScaleCentricGated(nn.Module):
    """
    ScaleCentricSimple 的 Stage 2 改进版：

    沿用 Stage 1（尺度内特征融合）完全相同。
    Stage 2 改为「门控加权 + 残差拼接」双路融合：
      - Gated 路径:  每个时间步学习三个尺度的 softmax 权重，加权求和
      - Residual 路径: 沿用原始 MLP(384→128) 拼接融合
      - 最终: fused = gated + residual

    门控路径的作用：
      - 显式建模「不同时刻信哪个频率」——震荡市偏向分时，趋势市偏向日周
      - 同时残差保留原始 concat→MLP 的通路，防止门控学偏时直接崩盘

    使用方式：config 中 model.type = 'scale_centric_gated'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：与 ScaleCentricSimple 完全一致
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（不变）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # Stage 2 门控路径: 每个时间步学三个尺度的 softmax 权重
        self.scale_gate = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, 3),          # 输出 3 个 logit
        )

        # Stage 2 残差路径: 保留原始的 concat → MLP 通路
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 融合后的层归一化（两路相加后可能有 scale 差异）
        self.fusion_norm = nn.LayerNorm(per_scale_dim)

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头（不变）
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        # === Stage 1: 尺度内特征融合（不变） ===
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)   # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B, 60, d)

        # === Stage 2: 尺度门控 + 残差 ===
        # 拼接三尺度 → 用于门控计算和残差通路
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1) # (B, 60, 3d)

        # 门控路径: 每个时间步独立权重
        gate_logits = self.scale_gate(stacked)          # (B, 60, 3)
        gate = torch.softmax(gate_logits, dim=-1)        # (B, 60, 3)
        # 加权求和: gate[:,:,0]*week + gate[:,:,1]*day + gate[:,:,2]*intra
        fused_gated = (gate[:, :, 0:1] * week_repr +
                       gate[:, :, 1:2] * day_repr +
                       gate[:, :, 2:3] * intra_repr)    # (B, 60, d)

        # 残差路径: 原始 concat → MLP
        fused_residual = self.scale_fusion(stacked)      # (B, 60, d)

        # 两路相加 + 归一化
        fused = self.fusion_norm(fused_gated + fused_residual)  # (B, 60, d)

        # === 时序编码 + 分类 ===
        fused_out, _ = self.temporal_encoder(fused)      # (B, 60, d)
        return fused_out[:, -1, :]                        # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Baseline 3-GatingOnly: 纯门控加权（消融实验：去掉残差MLP）
# ============================================================
class ScaleCentricGatingOnly(nn.Module):
    """
    ScaleCentricSimple 的 Stage 2 消融变体——仅门控，无残差MLP：

    沿用 Stage 1（尺度内特征融合）完全相同。
    Stage 2 改为纯 softmax 加权求和：
      - 拼接三尺度 → 小 MLP 输出 3 个 logit → softmax → 加权求和
      - fused = gate_w * week + gate_d * day + gate_i * intra

    与 ScaleCentricGated 的区别：没有残差 MLP 通路。
    消融目的：测试「每个时间步只信一个尺度」是否够用。

    使用方式：config 中 model.type = 'scale_centric_gating_only'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # 门控网络: 拼接三尺度 → 输出 3 个 logit
        self.scale_gate = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, 3),
        )

        # 时序编码
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        # Stage 1: 尺度内特征融合（不变）
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)

        # Stage 2: 纯门控加权
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        gate = torch.softmax(self.scale_gate(stacked), dim=-1)  # (B, 60, 3)
        fused = (gate[:, :, 0:1] * week_repr +
                 gate[:, :, 1:2] * day_repr +
                 gate[:, :, 2:3] * intra_repr)                  # (B, 60, d)

        # 时序编码 + 分类
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Baseline 3-Hierarchical: 层级化融合（ScaleCentric 的改进版）
# ============================================================
class ScaleCentricHierarchical(nn.Module):
    """
    ScaleCentricSimple 的 Stage 2 层级化改进版：

    沿用 Stage 1（尺度内特征融合）完全相同。
    Stage 2 改为「层级化融合」——显式建模周→日→分时的自上而下调节关系：
      - week_ctx = week_repr                               # 周频提供「大势」上下文
      - day_ctx  = day_repr + MLP(day | week_ctx)           # 日频受周频调节
      - intra_ctx = intra_repr + MLP(intra | day_ctx)       # 分时受日频(带周上下文)调节
      - fused = MLP(cat(week_ctx, day_ctx, intra_ctx))     # 最终融合

    层级关系与金融数据因果逻辑一致：
      - 大周期定调，小周期执行
      - 周不看涨 → 日看涨的信号自动打折
      - 日不活跃 → 分时的波动被降权

    使用方式：config 中 model.type = 'scale_centric_hierarchical'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：与 ScaleCentricSimple 完全一致
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（不变）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # Stage 2 层级调节: 周 → 日
        # 输入 [day_repr | week_repr] = 2d → d 的调制残差
        self.week_to_day = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 层级调节: 日 → 分时
        # 输入 [intra_repr | day_ctx] = 2d → d 的调制残差
        self.day_to_intra = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 最终三尺度融合 MLP
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.fusion_norm = nn.LayerNorm(per_scale_dim)

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头（不变）
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        # === Stage 1: 尺度内特征融合（不变） ===
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        week_repr  = self._encode_full(week_seq,  self.week_encoder)   # (B, 60, d)
        day_repr   = self._encode_full(day_seq,   self.day_encoder)    # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B, 60, d)

        # === Stage 2: 层级化融合 ===
        # Step 1: 周频调节日频——日频看自己，但被周频的「大势」修正
        day_modulated = day_repr + self.week_to_day(
            torch.cat([day_repr, week_repr], dim=-1))              # (B, 60, d)

        # Step 2: 日频调节分时——分时看自己，但被「带周上下文的日频」修正
        intra_modulated = intra_repr + self.day_to_intra(
            torch.cat([intra_repr, day_modulated], dim=-1))        # (B, 60, d)

        # Step 3: 三尺度最终融合
        stacked = torch.cat([week_repr, day_modulated, intra_modulated], dim=-1)  # (B, 60, 3d)
        fused = self.fusion_norm(self.scale_fusion(stacked))       # (B, 60, d)

        # === 时序编码 + 分类 ===
        fused_out, _ = self.temporal_encoder(fused)                # (B, 60, d)
        return fused_out[:, -1, :]                                 # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Baseline 3-Progressive: 渐进层级融合（ScaleCentric 的改进版）
# ============================================================
class ScaleCentricProgressive(nn.Module):
    """
    ScaleCentricSimple 的 Stage 2 渐进式改进版：

    沿用 Stage 1（尺度内特征融合）完全相同。
    Stage 2 改为「渐进式层级融合」——链式推进，最终输出只保留最细粒度的融合结果：
      Week
        ↓  MLP([Week | Day]) + Day(residual) → WeekDay
        ↓  MLP([WeekDay | Intra]) + Intra(residual) → 最终 fused

    与 ScaleCentricHierarchical 的区别：
      - Hierarchical: 三条线各自保留，最后 cat(Week, Day_mod, Intra_mod) → MLP
      - Progressive:  链式推进，周→周日→周日分时，上一级信息被逐级吸收，
                      最终只输出最细粒度的融合结果

    使用方式：config 中 model.type = 'scale_centric_progressive'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # 第一级融合: Week + Day → WeekDay
        self.week_day_fusion = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.fusion_norm1 = nn.LayerNorm(per_scale_dim)

        # 第二级融合: WeekDay + Intra → 最终 fused
        self.day_intra_fusion = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(per_scale_dim)

        # 时序编码
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        # Stage 1: 尺度内特征融合（不变）
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)

        # === Stage 2: 渐进式层级融合 ===

        # 第一级: Week + Day → WeekDay（残差连接 Day）
        week_day = self.week_day_fusion(
            torch.cat([week_repr, day_repr], dim=-1))          # (B, 60, d)
        week_day = self.fusion_norm1(week_day + day_repr)       # (B, 60, d)

        # 第二级: WeekDay + Intra → 最终 fused（残差连接 Intra）
        hier_repr = self.day_intra_fusion(
            torch.cat([week_day, intra_repr], dim=-1))         # (B, 60, d)
        hier_repr = self.fusion_norm2(hier_repr + intra_repr)  # (B, 60, d)

        # 时序编码 + 分类
        fused_out, _ = self.temporal_encoder(hier_repr)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Week 处理实验: 三种上采样策略对比
# 核心变化：week encoder 不再接收 repeat_interleave 后的伪 60 步台阶信号，
#          而是面对真实的 12 步周频序列，编码后再扩展到 60 步。
# ============================================================

# ------------------------------------------------------------
# 方案 1: Encoder 之后 repeat（分段常数，语义正确）
# ------------------------------------------------------------
class ScaleCentricPostRepeatWeek(nn.Module):
    """
    ScaleCentricSimple 的 Week 处理改进——方案 1：
    周频 encoder 先编码真实的 12 步序列，再将语义级表示 repeat 到 60 步。

    与 ScaleCentricSimple 的对比：
      - Simple: week (B,12,9) → repeat_interleave(5) → (B,60,9) → encoder → (B,60,d)
                ↑ 60 步台阶信号喂给 encoder，对 TCN 致命
      - 本模型:  week (B,12,9) → encoder → (B,12,d) → repeat_interleave(5) → (B,60,d)
                ↑ encoder 学习真实的周间变化，repeat 发生在语义层

    语义上合理：周频信号在一个自然周内本就应是分段常数。
    repeat_interleave 在 d 维语义空间的「台阶」是有意义的结构，而非噪声。

    使用方式：config 中 model.type = 'scale_centric_post_repeat_week'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: week_encoder 输入 8 维特征 × 12 步真实周频序列
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)              # input (B, 12, 8) 而非 (B, 60, 9)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1)                              # (B, 12, 8)  不 repeat
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)   # (B, 12, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B, 60, d)

        # 编码后 repeat：语义层分段常数
        week_repr = week_repr.repeat_interleave(5, dim=1)          # (B, 60, d)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ------------------------------------------------------------
# 方案 2: F.interpolate 线性插值（平滑过渡）
# ------------------------------------------------------------
class ScaleCentricInterpolateWeek(nn.Module):
    """
    ScaleCentricSimple 的 Week 处理改进——方案 2：
    周频 encoder 输出 (B,12,d)，用 F.interpolate 线性插值到 (B,60,d)。

    与方案 1 的区别：
      - PostRepeat: 分段常数，周内 5 天共享同一表示
      - Interpolate: 线性过渡，周一→周五平滑渐变

    注意：线性插值在两个不同周之间产生「虚构中间态」，在金融场景下
    可能引入不存在的噪声。保留此方案纯粹为对照实验。

    使用方式：config 中 model.type = 'scale_centric_interpolate_week'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)              # input (B, 12, 8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1)                              # (B, 12, 8)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)   # (B, 12, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B, 60, d)

        # 线性插值: (B,12,d) → (B,d,12) → interpolate → (B,d,60) → (B,60,d)
        week_repr = week_repr.transpose(1, 2)                      # (B, d, 12)
        week_repr = F.interpolate(week_repr, size=60,
                                  mode='linear', align_corners=False)
        week_repr = week_repr.transpose(1, 2)                      # (B, 60, d)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ------------------------------------------------------------
# 方案 3: 保留原生长度 + 段级对齐融合（week 不展开到 60）
# ------------------------------------------------------------
class ScaleCentricAlignWeek(nn.Module):
    """
    ScaleCentricSimple 的 Week 处理改进——方案 3：
    三尺度 encoder 全部保留原生长度，融合时做段级对齐。

    week:  (B, 12, d)   — 每段对应 5 个交易日
    day:   (B, 60, d)
    intra: (B, 60, d)

    融合方式：
      1. 对每个 day 时间步 t，按 k = t // 5 找到对应 week 段
      2. week 段先学出一个 Tanh 调制向量，对该段内 5 天做逐元素乘法
      3. 三尺度 cat 后过 MLP → temporal_encoder

    与方案 1 的区别：
      - 方案 1: week 物理展开为 60 步 → 简单的 per-timestep cat + MLP
      - 方案 3: week 保持 12 步，通过段级 Tanh 调制显式控制 day/intra 的激活强度
        「本周看涨」→ day 的信号放大，「本周看跌」→ day 的信号压抑

    使用方式：config 中 model.type = 'scale_centric_align_week'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)
        self.week_encoder  = mk(8)              # input (B, 12, 8)
        self.intra_encoder = mk(intra_hidden)

        # 段级调制器：week_repr 的每个段 → 调制 day/intra 的向量
        # Tanh 输出 [-1,1]，day_modulated = day * (1 + tanh_out)
        # 当 tanh_out > 0 时放大，< 0 时压抑
        self.week_modulator = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.Tanh(),
        )

        # 最终融合: [week_expanded | day_modulated | intra_modulated]
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday):
        B = daily.size(0)

        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1)                              # (B, 12, 8)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)   # (B, 12, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B, 60, d)

        # 段级调制: 每个周步学出调制向量
        week_mod = self.week_modulator(week_repr)                  # (B, 12, d)

        # 构建 60 步索引映射: t → t // 5
        idx = torch.arange(60, device=week_repr.device) // 5       # (60,) [0,0,0,0,0, 1,...]

        # 展开到 60 步（索引查找，不复制内存）
        week_expanded = week_repr[:, idx, :]                       # (B, 60, d)
        mod_expanded  = week_mod[:, idx, :]                        # (B, 60, d)

        # 周频调制日频和分时
        day_modulated   = day_repr   * (1.0 + mod_expanded)        # (B, 60, d)
        intra_modulated = intra_repr * (1.0 + mod_expanded)        # (B, 60, d)

        # 融合: week 原始信号 + 被调制的 day/intra
        stacked = torch.cat([week_expanded, day_modulated, intra_modulated], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# Baseline 4: 两路并行 → 拼接（Feature ⇄ Scale，无注意力）
# ============================================================
class BothCentricSimple(nn.Module):
    """
    同时计算两路表示：
      - 先跨尺度再跨特征（FeatureCentricSimple 的 repr）
      - 先跨特征再跨尺度（ScaleCentricSimple 的 repr）
    拼接后送入分类头。
    """

    def __init__(self, per_feat_dim=64, per_scale_dim=128,
                 hidden_dim=128, num_classes=5, dropout_rate=0.3,
                 num_layers=1, encoder_type='gru', kernel_size=3,
                 num_levels=4, intra_hidden_scale=32):
        super().__init__()
        self.fc_branch = FeatureCentricSimple(
            per_feat_dim=per_feat_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels)
        self.sc_branch = ScaleCentricSimple(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden_scale)

        # 拆分头部：head 仅保留到 hidden_dim 的前半，后半复用
        self.fc_branch.head = nn.Sequential(
            nn.BatchNorm1d(per_feat_dim),
            nn.Linear(per_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sc_branch.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
        )

        combined_dim = hidden_dim + hidden_dim   # fc repr + sc repr

        self.combined_head = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, daily, weekly, intraday):
        # 两路各自提取 repr（编码 + 自身内部融合）
        fc_repr = self.fc_branch.head(
            self.fc_branch._compute_repr(daily, weekly, intraday))
        sc_repr = self.sc_branch.head(
            self.sc_branch._compute_repr(daily, weekly, intraday))

        combined = torch.cat([fc_repr, sc_repr], dim=-1)
        return self.combined_head(combined)


# ============================================================
# 纯分时网络：仅利用 Intraday (B,60,13,8) → (B,480,9)
# ============================================================

class IntradayOnlyFlatten(nn.Module):
    """纯分时 Baseline — 直接平铺为 780 步 × 13 特征的时序序列。

    数据流: (B, 60, 13, 8) → reshape (B, 780, 8) → 时序编码器 → last → 分类头。

    780 = 60 天 × 13 根分时 bar，8 个金融特征不变。
    适配不同编码器类型（GRU/LSTM/TCN），对外接口兼容三频输入。
    """

    def __init__(self, hidden_dim=128, num_classes=5, dropout_rate=0.3,
                 num_layers=1, encoder_type='gru', kernel_size=3, num_levels=4):
        super().__init__()
        self.encoder = _make_encoder(
            encoder_type, 8, hidden_dim, num_layers,
            dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, daily, weekly, intraday):
        """intraday: (B, 60, 13, 8) → (B, 780, 8)"""
        B = intraday.size(0)
        x = intraday.reshape(B, 780, 8)       # (B, 780, 8)
        out, _ = self.encoder(x)               # (B, 780, hidden_dim)
        return self.head(out[:, -1, :])         # (B, num_classes)


# ============================================================
# Baseline 3-微调: 先跨特征 → 再跨尺度（Scale-Centric，无注意力）
# ============================================================
class ScaleCentricSimpleTuned(nn.Module):

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：13 bar × 9 特征 → 日级摘要
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（输出完整时序）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)              # (B,60,9)  → (B,60,d)
        self.week_encoder  = mk(8)              # (B,60,9)  → (B,60,d)
        self.intra_encoder = mk(intra_hidden)   # (B,60,intra_hidden) → (B,60,d)

        # Stage 2: 逐时间步跨尺度融合
        # 在每个交易日 t，三尺度表示拼接后过 MLP → 一个 t 时刻的融合向量
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)
    # yy： 分类头调整
        # 分类头-只用一层隐藏层
        self.head = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim // 2, num_classes),
        )
        # 分类头-1层线性 + 标签平滑
        # self.head = nn.Linear(per_scale_dim, num_classes)

    def _encode_full(self, x, encoder):
        """编码完整时序，返回 (B, T, hidden)，不截断 last。"""
        out, _ = encoder(x)
        return out                                  # (B, T, per_scale_dim)

    def _compute_repr(self, daily, weekly, intraday):
        """
        daily:    (B, 12, 5, 8)
        weekly:   (B, 1, 12, 8)
        intraday: (B, 60, 13, 8)
        → 返回 (B, per_scale_dim)
        """
        B = daily.size(0)

        # === Stage 1: 尺度内特征融合（保留完整 60 步时序） ===
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)  # (B, 60, 8)
        intra_daily = self.intra_compressor(intraday)              # (B, 60, intra_hidden)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)   # (B, 60, d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)  # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)  # (B, 60, d)

        # === Stage 2: 逐时间步跨尺度融合 + 时序编码 ===
        # 每个时间步 t: [week_repr(t) | day_repr(t) | intra_repr(t)] → 融合向量
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)  # (B, 60, 3*d)
        fused = self.scale_fusion(stacked)                               # (B, 60, d)

        fused_out, _ = self.temporal_encoder(fused)   # (B, 60, d)
        return fused_out[:, -1, :]                        # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# WeekOnly: 仅使用周频 (1, 12, 9) 的消融模型
# ============================================================
class WeekOnly(nn.Module):
    """仅使用周频数据的单尺度消融模型。

    参考 ScaleCentricSimple 的 Stage 1 设计，但只保留周频编码器。
    周频数据 (B, 1, 12, 8) → squeeze → (B, 12, 8) → 时序编码器 → last → 分类头。

    兼容三频输入接口：forward(daily, weekly, intraday) — 仅使用 weekly 参数。
    """

    def __init__(self, hidden_dim=128, num_classes=5, dropout_rate=0.3,
                 num_layers=1, encoder_type='gru', kernel_size=3, num_levels=4):
        super().__init__()
        self.encoder = _make_encoder(
            encoder_type, 8, hidden_dim, num_layers,
            dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, daily, weekly, intraday):
        """weekly: (B, 1, 12, 8) → (B, 12, 8)"""
        B = weekly.size(0)
        x = weekly.squeeze(1)              # (B, 12, 8)
        out, _ = self.encoder(x)           # (B, 12, hidden_dim)
        return self.head(out[:, -1, :])     # (B, num_classes)


# ============================================================
# DayIntra: 日频 + 分时（双尺度融合）
# ============================================================
class DayIntra(nn.Module):
    """日频 + 分时的双尺度消融模型。

    参考 ScaleCentricSimple 的两阶段设计：
    Stage 1 — 日内压缩 + 日频编码，各自输出 (B, 60, d)
    Stage 2 — 逐时间步 cat → MLP 融合 → 时序编码 → last → 分类头。

    数据流:
      daily:    (B, 12, 5, 8) → reshape → (B, 60, 9)
      intraday: (B, 60, 13, 8) → IntradayCompressor → (B, 60, intra_hidden) → encoder → (B, 60, d)
      day_seq:  (B, 60, 9)    → encoder → (B, 60, d)
      融合:     cat([day_repr, intra_repr]) → MLP → temporal_encoder → last → head

    兼容三频输入接口：forward(daily, weekly, intraday) — 忽略 weekly。
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：13 bar × 9 特征 → 日级摘要
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 日频 + 分时的特征内编码器
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)              # (B,60,9)  → (B,60,d)
        self.intra_encoder = mk(intra_hidden)   # (B,60,intra_hidden) → (B,60,d)

        # Stage 2: 逐时间步双尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头（与 ScaleCentricSimple 一致）
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, intraday):
        """daily: (B, 12, 5, 8), intraday: (B, 60, 13, 8) → (B, per_scale_dim)"""
        B = daily.size(0)

        # Stage 1: 尺度内特征融合
        day_seq     = daily.reshape(B, 60, 8)                  # (B, 60, 9)
        intra_daily = self.intra_compressor(intraday)          # (B, 60, intra_hidden)

        day_repr   = self._encode_full(day_seq,   self.day_encoder)   # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)  # (B, 60, d)

        # Stage 2: 逐时间步双尺度融合
        stacked = torch.cat([day_repr, intra_repr], dim=-1)    # (B, 60, 2*d)
        fused = self.scale_fusion(stacked)                     # (B, 60, d)
        fused_out, _ = self.temporal_encoder(fused)            # (B, 60, d)
        return fused_out[:, -1, :]                             # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, intraday))


# ============================================================
# DayWeek: 日频 + 周频（双尺度融合）
# ============================================================
class DayWeek(nn.Module):
    """日频 + 周频的双尺度消融模型。

    参考 ScaleCentricSimple 的两阶段设计：
    Stage 1 — 日频与周频各自编码，输出 (B, 60, d)
    Stage 2 — 逐时间步 cat → MLP 融合 → 时序编码 → last → 分类头。

    数据流:
      daily:  (B, 12, 5, 8) → reshape → (B, 60, 9) → encoder → (B, 60, d)
      weekly: (B, 1, 12, 8) → squeeze → repeat_interleave(5) → (B, 60, 9) → encoder → (B, 60, d)
      融合:   cat([day_repr, week_repr]) → MLP → temporal_encoder → last → head

    兼容三频输入接口：forward(daily, weekly, intraday) — 忽略 intraday。
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4):
        super().__init__()

        # Stage 1: 日频 + 周频的特征内编码器
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder  = mk(8)   # (B,60,9) → (B,60,d)
        self.week_encoder = mk(8)   # (B,60,9) → (B,60,d)

        # Stage 2: 逐时间步双尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头（与 ScaleCentricSimple 一致）
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly):
        """daily: (B, 12, 5, 8), weekly: (B, 1, 12, 8) → (B, per_scale_dim)"""
        B = daily.size(0)

        # Stage 1: 尺度内特征融合
        day_seq  = daily.reshape(B, 60, 8)                          # (B, 60, 9)
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)   # (B, 60, 9)

        day_repr  = self._encode_full(day_seq,  self.day_encoder)   # (B, 60, d)
        week_repr = self._encode_full(week_seq, self.week_encoder)  # (B, 60, d)

        # Stage 2: 逐时间步双尺度融合
        stacked = torch.cat([day_repr, week_repr], dim=-1)   # (B, 60, 2*d)
        fused = self.scale_fusion(stacked)                   # (B, 60, d)
        fused_out, _ = self.temporal_encoder(fused)          # (B, 60, d)
        return fused_out[:, -1, :]                           # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly))


class TemporalRelation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        x : (B,T,D)
        """
        # similarity
        A = torch.matmul(x, x.transpose(1,2))
        A = A / math.sqrt(x.size(-1))
        A = torch.softmax(A, dim=-1)
        out = torch.matmul(A, x)
        return x + out


# ============================================================
# Baseline 3 + Macro: 在 Stage 2 加入宏观指数信息
# 3 个变体：cat(沪深300+中证500) / 仅沪深300 / 仅中证500
# 宏观编码器 → (B, 60, d) 作为第 4 个频率，与日/周/分时拼入 scale_fusion
# ============================================================
class _ScaleCentricMacroBase(nn.Module):
    """
    Baseline3 基础 + macro 输入的基类。

    与 ScaleCentricSimpleTuned 的结构一致，区别：
    - __init__ 接受 macro_dim 参数（实际输入维度由子类确定）
    - forward 额外接收 macro: (B, 60, macro_dim)
    - Stage 2 融合 4 个频率（日/周/分时/macro）而非 3 个
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14, slice_cols=None):
        """
        slice_cols: 若不为 None，则从全量 macro(B,60,14) 中切片出需要的列。
                    None 表示全部 14 列都给 macro_encoder。
        """
        super().__init__()
        self.macro_dim = macro_dim
        self._slice_cols = slice_cols
        # 日内压缩：13 bar × 9 特征 → 日级摘要
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（输出完整时序）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)              # (B,60,9)  → (B,60,d)
        self.week_encoder  = mk(8)              # (B,60,9)  → (B,60,d)
        self.intra_encoder = mk(intra_hidden)   # (B,60,intra_hidden) → (B,60,d)
        self.macro_encoder = mk(macro_dim)      # (B,60,macro_dim) → (B,60,d)

        # Stage 2: 逐时间步 4 频率跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(4 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)

        # Stage 1: 尺度内特征融合
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        # macro 切片：若 slice_cols 指定了列索引则切取，否则全取
        if self._slice_cols is not None:
            macro = macro[:, :, self._slice_cols]

        day_repr   = self._encode_full(day_seq,   self.day_encoder)     # (B,60,d)
        week_repr  = self._encode_full(week_seq,  self.week_encoder)    # (B,60,d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder) # (B,60,d)
        macro_repr = self._encode_full(macro,       self.macro_encoder) # (B,60,d)

        # Stage 2: 4 频率逐时间步融合
        stacked = torch.cat([week_repr, day_repr, intra_repr, macro_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro=None):
        """
        macro: (B, 60, 14) — 始终是全量 14 维（沪深300 0:7 + 中证500 7:14）
        子类通过 slice_cols 指定取哪些列。
        """
        if macro is None:
            B = daily.size(0)
            macro = torch.randn(B, 60, self.macro_dim, device=daily.device)
        return self.head(self._compute_repr(daily, weekly, intraday, macro))


class ScaleCentricMacroCat(_ScaleCentricMacroBase):
    """
    沪深300 + 中证500 都加入（取全部列）。
    model_type: 'scale_centric_macro_cat'
    输入: daily, weekly, intraday, macro=(B,60,macro_dim)
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim, slice_cols=None)


class ScaleCentricMacroDualEnc(_ScaleCentricMacroBase):
    """
    沪深300 + 中证500 分开独立编码，共享一个 encoder 权重。

    与 ScaleCentricMacroCat 的区别：
    - 不再把两个指数拼成 (B,60,14) 输入单一 encoder
    - 而是各自切片为 (B,60,8) 后分别通过共享的 `self.macro_encoder` 编码
    - Stage 2 融合 5 个频率：日/周/分时/沪深300_repr/中证500_repr
    - scale_fusion 输入维度 = 5 * per_scale_dim

    model_type: 'scale_centric_macro_dual_enc'
    输入: daily, weekly, intraday, macro=(B,60,16) — 内部自动切前 8/后 8 列
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        # encoder 只对单指数 8 维编码
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=8, slice_cols=None)
        # 覆盖父类的 scale_fusion：5 个频率（日/周/分时/沪深300/中证500）
        self.scale_fusion = nn.Sequential(
            nn.Linear(5 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)

        # Stage 1: 尺度内特征融合
        day_seq     = daily.reshape(B, 60, 8)
        week_seq    = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        # 两个指数分开：前 8 列 = 沪深300，后 8 列 = 中证500
        # macro 是 (B,60,16) — 沪深300(0:8) + 中证500(8:16)，每指数 8 列含振幅
        idx300 = macro[:, :, :8]   # (B,60,8)
        idx500 = macro[:, :, 8:]   # (B,60,8)

        day_repr    = self._encode_full(day_seq,     self.day_encoder)     # (B,60,d)
        week_repr   = self._encode_full(week_seq,    self.week_encoder)    # (B,60,d)
        intra_repr  = self._encode_full(intra_daily, self.intra_encoder)   # (B,60,d)
        idx300_repr = self._encode_full(idx300,      self.macro_encoder)   # (B,60,d)
        idx500_repr = self._encode_full(idx500,      self.macro_encoder)   # (B,60,d)

        # Stage 2: 5 频率逐时间步融合
        stacked = torch.cat(
            [week_repr, day_repr, intra_repr, idx300_repr, idx500_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro=None):
        if macro is None:
            B = daily.size(0)
            macro = torch.randn(B, 60, 16, device=daily.device)
        return self.head(self._compute_repr(daily, weekly, intraday, macro))


class ScaleCentricMacro300(_ScaleCentricMacroBase):
    """
    仅沪深300（macro 取前 7 列: open/high/low/close/pct_chg/volume/amt）。
    model_type: 'scale_centric_macro_300'
    输入: daily, weekly, intraday, macro=(B,60,14) — 内部自动切前 7 列
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=7,
            slice_cols=slice(0, 7))   # 前 7 列 = 沪深300


class ScaleCentricMacro500(_ScaleCentricMacroBase):
    """
    仅中证500（macro 取后 7 列: open/high/low/close/pct_chg/volume/amt）。
    model_type: 'scale_centric_macro_500'
    输入: daily, weekly, intraday, macro=(B,60,14) — 内部自动切后 7 列
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=7,
            slice_cols=slice(7, 14))  # 后 7 列 = 中证500


# ============================================================
# Baseline 3 + 相对市场特征（模型内计算）
# 2 个变体：
#   - ScaleCentricRelative: Stock + Relative Features (daily 9→15，3 频融合)
#   - ScaleCentricRelativeMacro: Stock + Index + Relative (daily 9→15 + macro 第 4 频)
# ============================================================

# 通用辅助函数：从 macro 中提取 pct_chg / volume / amp（自适应 14/16 维）
# macro 列布局（16 维）：open(0),high(1),low(2),close(3),pct_chg(4),volume(5),amt(6),amp(7) × 2 指数
# macro 列布局（14 维，无振幅）：open(0),high(1),low(2),close(3),pct_chg(4),volume(5),amt(6) × 2 指数
def _extract_macro_cols(macro):
    """返回 (idx300_pct_chg, idx300_volume, idx300_amp_or_none, idx500_pct_chg, idx500_volume, idx500_amp_or_none)"""
    idx300_pct_chg = macro[:, :, 4]
    idx300_volume  = macro[:, :, 5]
    idx500_pct_chg = macro[:, :, 12 - (1 if macro.size(-1) == 14 else 0)]  # 12 or 11
    idx500_volume  = macro[:, :, 13 - (1 if macro.size(-1) == 14 else 0)]  # 13 or 12
    if macro.size(-1) == 16:
        idx300_amp = macro[:, :, 7]
        idx500_amp = macro[:, :, 15]
    else:  # 14 维无振幅
        idx300_amp = None
        idx500_amp = None
    return idx300_pct_chg, idx300_volume, idx300_amp, idx500_pct_chg, idx500_volume, idx500_amp


def _has_amp(macro):
    """macro 是否有振幅列（16 维有，14 维无）"""
    return macro.size(-1) == 16


class ScaleCentricRelative(nn.Module):
    """
    ScaleCentricSimple + 模型内动态计算相对市场特征。

    从 daily (B,12,5,9) 和 macro (B,60,16) 中动态计算 6 个相对特征：
      return_rel300 / return_rel500 / amp_rel300 / amp_rel500
      volume_rel300 / volume_rel500
    将 daily 从 8 维扩展到 15 维，其余结构与 ScaleCentricSimple 完全相同。

    model_type: 'scale_centric_relative'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        self.macro_dim = macro_dim
        _rel_dim = 6 if macro_dim == 16 else 4  # 有振幅 6 维，无振幅 4 维
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # Stage 2: 3 频跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        """
        daily_flat: (B, 60, 8)  列: open,high,low,close,pct_chg,volume,turn,amplitude
        macro:      (B, 60, 16|14)  16=含振幅, 14=无振幅
        → (B, 60, 6) 当 macro=16 维
        → (B, 60, 4) 当 macro=14 维（仅收益率+成交量）
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        parts = [
            torch.clamp(stock_pct_chg - p300, -20.0, 20.0),
            torch.clamp(stock_pct_chg - p500, -20.0, 20.0),
        ]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 动态拼接相对特征 → (B, 60, 15)
        rel_feats = self._compute_relative_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, rel_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# 特征贡献分析：Relative 子集消融实验
# rel_mask 控制加入哪几组相对特征（每组 2 维，对应沪深300/中证500）：
#   mask[0]=1 → 相对收益率 (return_rel300, return_rel500)   2 维
#   mask[1]=1 → 相对振幅     (amp_rel300, amp_rel500)       2 维
#   mask[2]=1 → 相对成交量   (volume_rel300, volume_rel500) 2 维
#
# 3 个 model_type:
#   scale_centric_relative_return — 仅相对收益率
#   scale_centric_relative_amp    — 收益率 + 振幅
#   scale_centric_relative_vol    — 收益率 + 振幅 + 成交量（=全量 relative）
# ============================================================
class ScaleCentricRelativeAbl(ScaleCentricRelative):
    """
    ScaleCentricRelative 的消融变体，通过 rel_mask 控制相对特征子集。

    rel_mask: (3,) 的 list/tuple，如 [1,0,0] 仅收益率、[1,1,0] 收益率+振幅等
    daily_expanded_dim = 8 + 2 * sum(rel_mask)
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 rel_mask=(1, 0, 0), macro_dim=14):
        # 先计算 expanded_dim 再传给父类——但父类__init__中已经硬编码了 day_encoder = mk(15)
        # 所以不能直接 super().__init__，需要自己重新构建
        super(ScaleCentricRelative, self).__init__()
        self.rel_mask = rel_mask
        self.macro_dim = macro_dim
        # 无振幅时 mask[1] 强制为 0
        if macro_dim == 14:
            rel_mask = (rel_mask[0], 0, rel_mask[2])
            self.rel_mask = rel_mask
        self._num_rel = sum(rel_mask) * 2  # 每组 2 维（300/500）
        daily_expanded_dim = 9 + self._num_rel

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        """根据 rel_mask 只计算被选中的子集。"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        parts = []
        # mask[0]: 相对收益率（带符号比例）
        if self.rel_mask[0]:
            sign300 = torch.sign(stock_pct_chg * p300)
            sign500 = torch.sign(stock_pct_chg * p500)
            return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
            return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)
            parts.extend([
                return_ratio300,
                return_ratio500,
            ])
        # mask[1]: 相对振幅（仅当 macro 有振幅列时）
        if self.rel_mask[1] and _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        # mask[2]: 相对成交量
        if self.rel_mask[2]:
            parts.extend([
                torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
            ])

        return torch.stack(parts, dim=-1)  # (B, 60, num_rel)


class ScaleCentricRelativeMacro(nn.Module):
    """
    ScaleCentricSimple + 相对市场特征 + Index 作为第 4 频。

    与 ScaleCentricRelative 的区别：
    - daily 同样 9→15 维（拼接 6 个相对特征）
    - 额外将 macro 作为第 4 个频率（如 _ScaleCentricMacroBase）
    - Stage 2 融合 4 频（日/周/分时/macro）

    model_type: 'scale_centric_relative_macro'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        self.macro_dim = macro_dim
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)
        self.macro_encoder = mk(macro_dim)

        # Stage 2: 4 频跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(4 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        parts = [
            torch.clamp(stock_pct_chg - p300, -20.0, 20.0),
            torch.clamp(stock_pct_chg - p500, -20.0, 20.0),
        ]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        rel_feats = self._compute_relative_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, rel_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)
        macro_repr = self._encode_full(macro,        self.macro_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr, macro_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ScaleCentricRelativeMacroDualEnc(nn.Module):
    """
    ScaleCentricRelativeMacro 的 dual-encoder 版本。
    - daily 同样 9→15 维（拼接 6 个相对特征）
    - 两个指数分开为 (B,60,8) 各自通过共享的 macro_encoder 编码
    - Stage 2 融合 5 频：日/周/分时/沪深300_repr/中证500_repr

    model_type: 'scale_centric_relative_macro_dual_enc'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder    = mk(daily_expanded_dim)
        self.week_encoder   = mk(8)
        self.intra_encoder  = mk(intra_hidden)
        self.macro_encoder  = mk(8)       # 单指数 8 维（14维无振幅时是7维？但dual_enc按前8列拆分）
                                          # 注意：_noamp 时 macro 14 维，前 7 列/后 7 列，但 macro_encoder=8 会报错
                                          # 需要改为按 macro_dim/2 动态

        # 单指数维度：macro_dim/2
        _single_macro_dim = macro_dim // 2
        self.macro_encoder  = mk(_single_macro_dim)

        # Stage 2: 5 频跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(5 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        parts = [
            torch.clamp(stock_pct_chg - p300, -20.0, 20.0),
            torch.clamp(stock_pct_chg - p500, -20.0, 20.0),
        ]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        rel_feats = self._compute_relative_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, rel_feats], dim=-1)

        # 两个指数分开
        _half = macro.size(-1) // 2
        idx300 = macro[:, :, :_half]
        idx500 = macro[:, :, _half:]

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr    = self._encode_full(day_seq,     self.day_encoder)
        week_repr   = self._encode_full(week_seq,    self.week_encoder)
        intra_repr  = self._encode_full(intra_daily, self.intra_encoder)
        idx300_repr = self._encode_full(idx300,      self.macro_encoder)
        idx500_repr = self._encode_full(idx500,      self.macro_encoder)

        stacked = torch.cat(
            [week_repr, day_repr, intra_repr, idx300_repr, idx500_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Level 2: Ratio-based Relative（比例表示）
# 将差值改为比例，对不同市场环境有更好的尺度不变性。
# 3 组比例特征（每组 2 维）：
#   return_ratio = pct_chg_stock / pct_chg_index    (有符号比例)
#   amp_ratio    = amp_stock / amp_index              (无量纲比值)
#   volume_ratio = vol_stock / vol_index              (相对活跃度)
# 拼接后 daily 9→15，3 频融合，结构同 ScaleCentricRelative。
# model_type: 'scale_centric_ratio'
# ============================================================
class ScaleCentricRatio(nn.Module):
    """
    Level 2 — Ratio-based Relative。

    与 ScaleCentricRelative 的唯一区别是：
    - 相对收益率 = pct_chg_stock / pct_chg_index（有符号比例，比差值更稳健）
    - 相对振幅   = amp_stock / amp_index（同前）
    - 相对成交量 = vol_stock / vol_index（同前）
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        # 子类可通过设置 self._override_rel_dim 覆盖 ratio 维度
        _rel_dim = getattr(self, '_override_rel_dim',
                           6 if macro_dim == 16 else 4)
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """
        daily_flat: (B, 60, 8)
        macro:      (B, 60, 16|14)
        → (B, 60, 6|4)
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        # 有符号比例
        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Level 2a: ScaleCentricRatio + 轻量级跨尺度交互 (CSI)
# 在 ScaleCentricRatio 基础上，三个尺度独立 TCN 编码后，
# 通过共享两层 MLP 实现跨尺度残差信息传递（无注意力），
# 再进入原有的 concat+MLP 融合和 TCN 时序建模。
# model_type: 'scale_centric_ratio_csi'
# ============================================================
class ScaleCentricRatioCSI(nn.Module):
    """
    ScaleCentricRatio + Cross-Scale Interaction (CSI)。

    与 ScaleCentricRatio 的唯一区别：
    在三个尺度独立 TCN 编码后、concat 融合之前，增加一次轻量级跨尺度交互：
      对每个尺度，将另外两个尺度的特征拼接后通过共享两层 MLP 映射到当前维度，
      以残差连接方式更新当前尺度特征。

    整体结构：
      daily(9+rel) ─→ DayTCN   ──┐
      weekly(9)    ─→ WeekTCN  ──┤  ─→ CSI交互层 ─→ concat ─→ MLP融合 ─→ TCN ─→ head
      intra(h)     ─→ IntraTCN ──┘        ↑
                            共享两层 MLP (2d → d → d)
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # ---- 新增：CSI 共享跨尺度交互 MLP ----
        # 输入：另外两个尺度拼接 (2 * per_scale_dim) → 输出：per_scale_dim
        self.csi_mlp = nn.Sequential(
            nn.Linear(2 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """同 ScaleCentricRatio。"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _cross_scale_interact(self, day_repr, week_repr, intra_repr):
        """
        CSI 跨尺度交互：三个尺度共享同一个 MLP，残差更新。
        输入/输出均为 (B, 60, d)。
        """
        # 日频更新：自身 + MLP(cat(周频, 分时))
        other_for_day = torch.cat([week_repr, intra_repr], dim=-1)  # (B, 60, 2d)
        day_delta = self.csi_mlp(other_for_day)                     # (B, 60, d)
        day_new = day_repr + day_delta

        # 周频更新：自身 + MLP(cat(日频, 分时))
        other_for_week = torch.cat([day_repr, intra_repr], dim=-1)
        week_delta = self.csi_mlp(other_for_week)
        week_new = week_repr + week_delta

        # 分时更新：自身 + MLP(cat(日频, 周频))
        other_for_intra = torch.cat([day_repr, week_repr], dim=-1)
        intra_delta = self.csi_mlp(other_for_intra)
        intra_new = intra_repr + intra_delta

        return day_new, week_new, intra_new

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)   # (B, 60, d)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)  # (B, 60, d)

        # ---- CSI 跨尺度交互 ----
        day_repr, week_repr, intra_repr = self._cross_scale_interact(
            day_repr, week_repr, intra_repr)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Level 2b: ScaleCentricRatio + Scale Mixing（基于 MLP-Mixer 思想）
# 在 ScaleCentricRatio 基础上，用共享 MLP 独立处理每个尺度，
# 配合残差连接实现隐式跨尺度信息交换（无显式拼接其他尺度）。
# model_type: 'scale_centric_ratio_scale_mix'
# ============================================================
class ScaleCentricRatioScaleMix(nn.Module):
    """
    ScaleCentricRatio + Scale Mixing（MLP-Mixer 风格）。

    与 ScaleCentricRatioCSI 的唯一区别：
    将原来的 concat(其它两个尺度) → MLP → 残差 替换为：
      对每个尺度独立应用 共享 MLP(d→d) → 残差。

    结构：
      daily(9+rel) ─→ DayTCN   ──┐
      weekly(9)    ─→ WeekTCN  ──┤  ─→ ScaleMix交互 ─→ concat ─→ MLP融合 ─→ TCN ─→ head
      intra(h)     ─→ IntraTCN ──┘        ↑
                              共享 MLP(d→d) 独立作用于每个尺度
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # ---- Scale Mixing：共享 MLP 独立作用于每个尺度 ----
        # 输入/输出均为 (B, 60, d)，残差连接
        self.scale_mix = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """同 ScaleCentricRatio。"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _cross_scale_interact(self, day_repr, week_repr, intra_repr):
        """
        Scale Mixing：共享 MLP 独立作用于每个尺度，残差连接。
        输入/输出均为 (B, 60, d)。
        """
        day_new   = day_repr   + self.scale_mix(day_repr)
        week_new  = week_repr  + self.scale_mix(week_repr)
        intra_new = intra_repr + self.scale_mix(intra_repr)
        return day_new, week_new, intra_new

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)   # (B, 60, d)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)  # (B, 60, d)

        # ---- Scale Mixing 交互 ----
        day_repr, week_repr, intra_repr = self._cross_scale_interact(
            day_repr, week_repr, intra_repr)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Level 1b: 全部用差值（统一表示）
# 对比 Level 1（原 relative，收益率差值+振幅/成交量比例）：
# 这里收益率/振幅/成交量全部用差值，排除"比例vs差值"这个混杂因素。
# model_type: 'scale_centric_diff_all'
# ============================================================
class ScaleCentricDiffAll(nn.Module):
    """
    全部用差值表示：
      return_diff = pct_chg_stock - pct_chg_index（同原）
      amp_diff    = amp_stock - amp_index
      volume_diff = vol_stock - vol_index
    daily 9→15，3 频融合。
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_diff_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        parts = [
            torch.clamp(stock_pct_chg - p300, -20.0, 20.0),
            torch.clamp(stock_pct_chg - p500, -20.0, 20.0),
        ]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp - a300, -20.0, 20.0),
                torch.clamp(stock_amp - a500, -20.0, 20.0),
            ])
        parts.extend([
            torch.clamp(stock_volume - v300, -20.0, 20.0),
            torch.clamp(stock_volume - v500, -20.0, 20.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        diff_feats = self._compute_diff_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, diff_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Level 3: State-based Relative（状态表示）
# 将股票与指数的数值比较转化为离散状态，使模型学习"状态差异"而非具体数值。
#
# 状态划分（通过简单阈值）：
#   收益率状态:  -1(大跌) / 0(震荡) / 1(大涨)    基于 pct_chg 绝对值阈值 1.5%
#   振幅状态:    0(低波动) / 1(高波动)            基于振幅中位数阈值
#   成交量状态:  0(缩量) / 1(正常) / 2(放量)      基于 vol/index_vol 比值阈值
#
# 股票状态和指数状态分别计算，然后组合编码：
#   return_state:  (stock_state << 2) | index_state  → 0..8 共 9 种组合
#   amp_state:     (stock_state << 1) | index_state  → 0..3 共 4 种组合
#   volume_state:  (stock_state << 2) | index_state  → 0..8 共 9 种组合
# 组合后 embedding 为 4/2/4 维，拼接得到 daily 9+10=18 维。
# model_type: 'scale_centric_state'
# ============================================================
class ScaleCentricState(nn.Module):
    """
    Level 3 — State-based Relative。

    将连续数值转为离散状态组合，输入 embedding 层学习状态关系。
    不引入复杂网络，仅增加少量 embedding 参数。
    """
    _EPS = 1e-8
    _RET_THRESH = 1.5    # 收益率阈值 %
    _VOL_RATIO_LO = 0.5  # 缩量阈值
    _VOL_RATIO_HI = 2.0  # 放量阈值

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14):
        super().__init__()
        self.has_amp = (macro_dim == 16)
        # 状态组合 embedding
        self.return_embed = nn.Embedding(9, 4)    # 3×3 种状态组合
        if self.has_amp:
            self.amp_embed    = nn.Embedding(4, 2)    # 2×2 种状态组合
        self.volume_embed = nn.Embedding(9, 4)    # 3×3 种状态组合
        state_dim = 4 + (2 if self.has_amp else 0) + 4  # 10 or 8
        daily_expanded_dim = 8 + state_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    @staticmethod
    def _to_return_state(x):
        """x: (B,60), pct_chg 值 → -1/0/1"""
        return torch.where(x.abs() > ScaleCentricState._RET_THRESH,
                           torch.sign(x).long(), torch.zeros_like(x).long())

    @staticmethod
    def _to_amp_state(x, dim=1):
        """x: (B,60), 振幅值 → 0(低) / 1(高)，基于样本内中位数阈值"""
        med = x.median(dim=dim, keepdim=True).values
        return (x > med).long()

    @staticmethod
    def _to_volume_state(x):
        """x: (B,60), vol/index_vol 比值 → 0(缩量) / 1(正常) / 2(放量)"""
        return torch.where(
            x > ScaleCentricState._VOL_RATIO_HI,
            torch.full_like(x, 2, dtype=torch.long),
            torch.where(
                x < ScaleCentricState._VOL_RATIO_LO,
                torch.zeros_like(x, dtype=torch.long),
                torch.ones_like(x, dtype=torch.long),
            )
        )

    def _compute_state_features(self, daily_flat, macro):
        """
        → (B, 60, 10|8)  state embedding 拼接
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        B, T = stock_pct_chg.shape
        device = stock_pct_chg.device

        # === 收益率状态组合（沪深300）===
        stk_ret_s = self._to_return_state(stock_pct_chg)       # -1/0/1
        idx_ret_s = self._to_return_state(p300)                # -1/0/1
        comb_ret = (stk_ret_s + 1) * 3 + (idx_ret_s + 1)       # 0..8
        ret_emb = self.return_embed(comb_ret)                  # (B,60,4)

        parts = [ret_emb]

        # === 振幅状态组合（沪深300）（仅当 macro 有振幅列时）===
        if self.has_amp:
            stk_amp_s = self._to_amp_state(stock_amp)          # 0/1
            idx_amp_s = self._to_amp_state(a300)               # 0/1
            comb_amp = stk_amp_s * 2 + idx_amp_s               # 0..3
            amp_emb = self.amp_embed(comb_amp)                 # (B,60,2)
            parts.append(amp_emb)

        # === 成交量状态组合（沪深300）===
        vol_ratio300 = stock_volume / (v300 + self._EPS)
        stk_vol_s = self._to_volume_state(vol_ratio300)        # 0/1/2
        idx_vol_s = self._to_volume_state(
            v300 / (v300.mean(dim=1, keepdim=True) + self._EPS))
        comb_vol = stk_vol_s * 3 + idx_vol_s                   # 0..8
        vol_emb = self.volume_embed(comb_vol)                  # (B,60,4)
        parts.append(vol_emb)

        return torch.cat(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        state_feats = self._compute_state_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, state_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Baseline 3-微调: 添加Relation Module
# ============================================================
class ScaleCentricSimpleRelation(nn.Module):

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32):
        super().__init__()
        # 日内压缩：13 bar × 9 特征 → 日级摘要
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # Stage 1: 三尺度各自的「特征内融合」编码器（输出完整时序）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)              # (B,60,9)  → (B,60,d)
        self.week_encoder  = mk(8)              # (B,60,9)  → (B,60,d)
        self.intra_encoder = mk(intra_hidden)   # (B,60,intra_hidden) → (B,60,d)
        self.relation = TemporalRelation()
        # Stage 2: 逐时间步跨尺度融合
        # 在每个交易日 t，三尺度表示拼接后过 MLP → 一个 t 时刻的融合向量
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Stage 2 续: 在 60 天时序上编码融合后的序列
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # 分类头
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _encode_full(self, x, encoder):
        """编码完整时序，返回 (B, T, hidden)，不截断 last。"""
        out, _ = encoder(x)
        return out                                  # (B, T, per_scale_dim)

    def _compute_repr(self, daily, weekly, intraday):
        """
        daily:    (B, 12, 5, 8)
        weekly:   (B, 1, 12, 8)
        intraday: (B, 60, 13, 8)
        → 返回 (B, per_scale_dim)
        """
        B = daily.size(0)

        # === Stage 1: 尺度内特征融合（保留完整 60 步时序） ===
        day_seq   = daily.reshape(B, 60, 8)
        week_seq  = weekly.squeeze(1).repeat_interleave(5, dim=1)  # (B, 60, 8)
        intra_daily = self.intra_compressor(intraday)              # (B, 60, intra_hidden)

        day_repr = self.relation(self._encode_full(day_seq, self.day_encoder))
        week_repr = self.relation(self._encode_full(week_seq, self.week_encoder))
        intra_repr = self.relation(self._encode_full(intra_daily, self.intra_encoder))

        # === Stage 2: 逐时间步跨尺度融合 + 时序编码 ===
        # 每个时间步 t: [week_repr(t) | day_repr(t) | intra_repr(t)] → 融合向量
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)  # (B, 60, 3*d)
        fused = self.scale_fusion(stacked)                               # (B, 60, d)

        fused_out, _ = self.temporal_encoder(fused)   # (B, 60, d)
        return fused_out[:, -1, :]                        # (B, d)

    def forward(self, daily, weekly, intraday):
        return self.head(self._compute_repr(daily, weekly, intraday))


# ============================================================
# FFT 频域过滤辅助函数
# ============================================================

def _fft_low_freq_trend(signal, n_keep=5):
    """
    对信号做 rfft，保留前 n_keep 个最低频分量，其余置零，irfft 回时域。

    signal: (B, T) 或 (B, T, F) — 对最后一维做 FFT
    n_keep: 保留的低频分量数（在 rfft 后的频点数中从 DC 开始计）

    return: (B, T) 或 (B, T, F) — 市场低频趋势信号
    """
    orig_shape = signal.shape
    if signal.dim() == 3:
        B, T, F = signal.shape
        # 逐特征做 FFT
        trends = []
        for f in range(F):
            spec = torch.fft.rfft(signal[:, :, f], dim=-1)  # (B, T//2+1)
            mask = torch.zeros_like(spec)
            mask[:, :n_keep] = spec[:, :n_keep]
            trend = torch.fft.irfft(mask, n=T, dim=-1)      # (B, T)
            trends.append(trend)
        return torch.stack(trends, dim=-1)  # (B, T, F)
    else:
        # (B, T)
        spec = torch.fft.rfft(signal, dim=-1)
        mask = torch.zeros_like(spec)
        mask[:, :n_keep] = spec[:, :n_keep]
        return torch.fft.irfft(mask, n=signal.size(-1), dim=-1)


# ============================================================
# 实验一 — FFT 频域残差过滤
# ============================================================
# 变体 A: 股票 + 市场双 FFT，可学习 α × spectrum_market 残差
# 变体 B: 仅市场 FFT，提取低频趋势，时域相减
# 基类: ScaleCentricRelative（3 频融合，relative features）
# 两个变体都在 _compute_repr 中对 daily 所有 8 维特征做 FFT 过滤，
# 然后用过滤后的信号替换原始 daily，再走相对特征拼接逻辑。
# ============================================================

class ScaleCentricFFTResidual(nn.Module):
    """
    实验一变体 A — 频域残差过滤（可学习 α × market_spectrum）。

    思路：对每只个股和对应指数做 FFT，在频域中用可学习系数 α 乘市场频谱
    作为 Common，残差 = stock_spectrum - α * market_spectrum，IFFT 回时域
    得到过滤后的股票信号。

    继承 ScaleCentricRelative 的结构，但在 _compute_repr 中先对 daily 所有
    8 维特征做 FFT 过滤，再用过滤后的信号走原有的 relative 拼接和 3 频融合。

    model_type: 'scale_centric_fft_residual'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 n_keep=5, macro_dim=14):
        """
        n_keep: 保留的低频分量数（用于低频截断，变体 A 的 α 是频域全局系数，
                但为了简化实现，n_keep 仅在低频相减时用——这里是保留所有频点，
                用可学习 α 缩放后相减，不截断。）
        """
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # 可学习的 α 系数（每个特征一个）：8 维特征各一个全局 α
        self.alpha = nn.Parameter(torch.ones(8) * 0.5)

        # Stage 2: 3 频跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_rel300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_rel500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_rel300, return_rel500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _fft_filter_residual(self, daily_flat, macro):
        """
        变体 A: 可学习 α × market_spectrum 残差。

        daily_flat: (B, 60, 8)
        macro:      (B, 60, 16)  沪深300(0:8) + 中证500(8:16)
        → (B, 60, 9)  过滤后的信号
        """
        B, T, F = daily_flat.shape
        # 市场信号 = 沪深300 + 中证500 平均（自适应 16 或 14 维）
        _half = macro.size(-1) // 2
        market_avg = (macro[:, :, :_half] + macro[:, :, _half:]) / 2.0  # (B, 60, _half)

        filtered = []
        for f in range(F):
            stock_sig = daily_flat[:, :, f]       # (B, 60)
            # 对应的市场特征（前 8 个特征直接对应，amplitude=9 用第 8 个市场振幅）
            mkt_idx = min(f, _half - 1)            # 自适应：16维用7，14维用6
            mkt_sig = market_avg[:, :, mkt_idx]    # (B, 60)

            # 双 FFT
            stock_spec = torch.fft.rfft(stock_sig, dim=-1)  # (B, 31)
            mkt_spec   = torch.fft.rfft(mkt_sig, dim=-1)    # (B, 31)

            # 残差 = stock_spec - alpha[f] * mkt_spec
            alpha_f = torch.clamp(self.alpha[f], 0.0, 1.0)
            residual_spec = stock_spec - alpha_f * mkt_spec

            # IFFT 回时域
            residual = torch.fft.irfft(residual_spec, n=T, dim=-1)  # (B, 60)
            filtered.append(residual)

        return torch.stack(filtered, dim=-1)  # (B, 60, 9)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # Step 1: FFT 过滤 → 用过滤后的信号替换原始 daily
        daily_filtered = self._fft_filter_residual(daily_flat, macro)

        # Step 2: 在过滤后的信号上计算相对特征
        rel_feats = self._compute_relative_features(daily_filtered, macro)
        day_seq = torch.cat([daily_filtered, rel_feats], dim=-1)  # (B,60,15)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ScaleCentricFFTLowCut(nn.Module):
    """
    实验一变体 B — 市场低频趋势过滤。

    思路：只对市场做 FFT，提取前 k 个低频分量（市场低频趋势），IFFT 回时域，
    然后股票原始值减去这个市场低频趋势，得到过滤后的股票信号。

    继承 ScaleCentricRelative 的结构，但在 _compute_repr 中先对 daily 所有
    8 维特征做低频趋势过滤，再用过滤后的信号走原有的 relative 拼接和 3 频融合。

    model_type: 'scale_centric_fft_lowcut'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 n_keep=5, macro_dim=14):
        """
        n_keep: 保留的低频分量数（在 rfft 后的频点数中，从 DC 开始计）。
                60 天的 rfft 有 31 个频点，n_keep=5 保留前 5 个最低频。
        """
        super().__init__()
        self.n_keep = n_keep
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # Stage 2: 3 频跨尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_relative_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_rel300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_rel500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_rel300, return_rel500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _fft_lowcut_filter(self, daily_flat, macro):
        """
        变体 B: 市场低频趋势过滤。

        daily_flat: (B, 60, 8)
        macro:      (B, 60, 16|14)
        → (B, 60, 9)  过滤后的信号
        """
        B, T, F = daily_flat.shape
        _half = macro.size(-1) // 2
        # 市场平均
        market_avg = (macro[:, :, :_half] + macro[:, :, _half:]) / 2.0  # (B, 60, _half)

        filtered = []
        for f in range(F):
            stock_sig = daily_flat[:, :, f]       # (B, 60)
            mkt_idx = min(f, _half - 1)
            mkt_sig = market_avg[:, :, mkt_idx]    # (B, 60)

            # 对市场做 FFT → 提取低频 → IFFT → 市场低频趋势
            mkt_trend = _fft_low_freq_trend(mkt_sig, n_keep=self.n_keep)  # (B, 60)

            # 股票原始值减去市场低频趋势
            cleaned = stock_sig - mkt_trend
            filtered.append(cleaned)

        return torch.stack(filtered, dim=-1)  # (B, 60, 9)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # Step 1: 低频趋势过滤
        daily_filtered = self._fft_lowcut_filter(daily_flat, macro)

        # Step 2: 在过滤后的信号上计算相对特征
        rel_feats = self._compute_relative_features(daily_filtered, macro)
        day_seq = torch.cat([daily_filtered, rel_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# 实验二 — 宏观调控器门控注入
# ============================================================
# 方案 A: 全局池化门控注入 (ScaleCentricMacroGate)
#   指数编码器输出 → GAP → (B,d) 状态向量 → FC_scale/FC_shift → broadcast
#   到日/周/分时三个编码器的中间特征上：f' = f * (1 + scale) + shift
#
# 方案 B: 逐日调节门控注入 (ScaleCentricMacroGateDaily)
#   指数编码器保留 (B,60,d) 不池化
#   两个小型 FC 生成 scale 和 shift 系数
#   对日/周/分时：f' = f * (1 + scale) + shift
#
# 基类: ScaleCentricRelativeMacro（4 频融合 + relative features + macro 第 4 频）
# ============================================================

class _MacroGateBase(nn.Module):
    """
    实验二的基类 — 宏观调控器门控注入。

    继承 ScaleCentricRelativeMacro 的结构，但改造指数编码器的使用方式：
    - macro_encoder 保留，输出宏调控信号
    - 宏调控信号不参与最后的 concat 融合
    - 而是通过 scale/shift 门控作用到日/周/分时三个编码器的输出上
    - 被调节后的三个尺度输出再 concat + MLP 融合

    子类需实现 _get_macro_control_signal(macro_repr) → (B, d) 或 (B, 60, d)
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=16, gate_hidden=32):
        """
        gate_hidden: 门控生成器中间维度
        """
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)
        self.macro_encoder = mk(macro_dim)

        # 门控生成器：宏调控信号 → scale 和 shift
        # 子类在 _build_gate 中根据具体方案构造
        self._build_gate(per_scale_dim, gate_hidden, dropout_rate)

        # Stage 2: 3 频跨尺度融合（去掉 macro 频率，用 gate 替代）
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _build_gate(self, per_scale_dim, gate_hidden, dropout_rate):
        """子类实现：构建 scale_generator 和 shift_generator。"""
        raise NotImplementedError

    def _get_macro_control_signal(self, macro_repr):
        """子类实现：从 macro_repr 提取门控信号。"""
        raise NotImplementedError

    def _compute_relative_features(self, daily_flat, macro):
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume   = daily_flat[:, :, 5]
        stock_amp      = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_rel300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_rel500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_rel300, return_rel500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 相对特征
        rel_feats = self._compute_relative_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, rel_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        # 编码四路
        day_repr   = self._encode_full(day_seq,      self.day_encoder)      # (B,60,d)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)     # (B,60,d)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)    # (B,60,d)
        macro_repr = self._encode_full(macro,        self.macro_encoder)    # (B,60,d)

        # 宏调控信号 → scale/shift
        scale, shift = self._get_macro_control_signal(macro_repr)
        # scale, shift: (B, 1, d) 或 (B, 60, d)，广播到三尺度

        # 门控注入到日/周/分时
        day_repr   = day_repr   * (1.0 + scale) + shift
        week_repr  = week_repr  * (1.0 + scale) + shift
        intra_repr = intra_repr * (1.0 + scale) + shift

        # 3 频融合（无 macro）
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ScaleCentricMacroGate(_MacroGateBase):
    """
    实验二方案 A — 全局池化门控注入。

    指数编码器输出 (B,60,d) → GAP → (B,d) 状态向量 v
    v → FC_scale → (B,d) → reshape → (B,1,d) → broadcast 到 (B,60,d)
    v → FC_shift → (B,d) → reshape → (B,1,d) → broadcast 到 (B,60,d)
    对日/周/分时：f' = f * (1 + scale) + shift

    model_type: 'scale_centric_macro_gate'
    """

    def _build_gate(self, per_scale_dim, gate_hidden, dropout_rate):
        self.scale_generator = nn.Sequential(
            nn.Linear(per_scale_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, per_scale_dim),
            nn.Tanh(),
        )
        self.shift_generator = nn.Sequential(
            nn.Linear(per_scale_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, per_scale_dim),
            nn.Tanh(),
        )

    def _get_macro_control_signal(self, macro_repr):
        """
        macro_repr: (B, 60, d)
        → scale: (B, 1, d), shift: (B, 1, d)
        """
        # 全局平均池化
        v = macro_repr.mean(dim=1)                     # (B, d)
        scale = self.scale_generator(v).unsqueeze(1)   # (B, 1, d)
        shift = self.shift_generator(v).unsqueeze(1)   # (B, 1, d)
        return scale, shift


class ScaleCentricMacroGateDaily(_MacroGateBase):
    """
    实验二方案 B — 逐日调节门控注入。

    指数编码器保留 (B,60,d) 不池化。
    两个小型 FC：
      scale_generator: (B,60,d) → (B,60,d) — 逐日缩放系数
      shift_generator: (B,60,d) → (B,60,d) — 逐日偏移系数
    对日/周/分时：f' = f * (1 + scale) + shift

    model_type: 'scale_centric_macro_gate_daily'
    """

    def _build_gate(self, per_scale_dim, gate_hidden, dropout_rate):
        self.scale_generator = nn.Sequential(
            nn.Linear(per_scale_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, per_scale_dim),
            nn.Tanh(),
        )
        self.shift_generator = nn.Sequential(
            nn.Linear(per_scale_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, per_scale_dim),
            nn.Tanh(),
        )

    def _get_macro_control_signal(self, macro_repr):
        """
        macro_repr: (B, 60, d)
        → scale: (B, 60, d), shift: (B, 60, d)
        """
        scale = self.scale_generator(macro_repr)   # (B, 60, d)
        shift = self.shift_generator(macro_repr)   # (B, 60, d)
        return scale, shift

# ============================================================
# 多任务学习: Baseline3 + 未来收益率序列辅助监督 + 统计量辅助监督
# ============================================================
class ScaleCentricFutureMTL(_ScaleCentricMacroBase):
    """
    多任务框架：_ScaleCentricMacroBase 主干 + 三个辅助头

    继承 macro 基类，将 macro 作为第 4 个频率编码并融合。
    增加三个头：
    1. main_head: (B,d) → (B,3)  三分类（主任务）
    2. seq_head:  (B,d) → (B,T)  未来累计收益曲线预测（实验1）
    3. stat_head: (B,d) → (B,8)  未来统计量预测（实验2）

    forward 返回:
        logits, proj, aux_seq_pred, aux_stat_pred
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=14, future_days=50):
        # 基类初始化：4 频率融合（日/周/分时/macro），macro_dim 传入全量
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim, slice_cols=None)

        # 覆盖基类的 self.head，替换为自己的四个头
        # === 主分类头 ===
        self.main_head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # === 对比投影头（同 Baseline3） ===
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # === 实验1：未来收益率序列解码头 ===
        self.seq_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, future_days),
        )

        # === 实验2：统计量预测头（8维） ===
        self.stat_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 8),  # [max_ret, min_ret, max_pos, min_pos, cum_max_ret, max_drawdown, volatility, breakthrough_day]
        )

    def forward(self, daily, weekly, intraday, macro=None):
        if macro is None:
            B = daily.size(0)
            macro = torch.randn(B, 60, self.macro_dim, device=daily.device)
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.main_head(repr_vec)
        proj = self.projection(repr_vec)
        aux_seq = self.seq_head(repr_vec)    # (B, T)
        aux_stat = self.stat_head(repr_vec)  # (B, 4)
        return logits, proj, aux_seq, aux_stat


# ============================================================
# 多任务学习 V2: 基于 ScaleCentricRelative + 任务解耦投影 + 时序平滑
# 两个变体：
#   v1: cls_head 沿用 ScaleCentricRelative 的结构（BN → Linear→ReLU→Dropout→...）
#   v2: cls_head 简化为 Linear→ReLU→Dropout→Linear（无 BN，减少深度）
# ============================================================
class _ScaleCentricMTLRelativeBase(ScaleCentricRatio):
    """
    多任务基类：继承 ScaleCentricRatio（3频+ratio特征注入），
    增加三个任务投影层 + 三个任务 head。
    
    子类只需覆盖 _build_cls_head() 来自定义分类头结构。
    
    forward 返回: logits, proj, aux_seq_pred, aux_stat_pred
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=14):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        # 覆盖父类的 self.head 和 self.projection
        # 改为任务专用投影 + 各任务 head
        
        # === 任务解耦投影 ===
        self.cls_proj = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # === 分类头（由子类定义）===
        self.main_head = self._build_cls_head(
            per_scale_dim, hidden_dim, num_classes, dropout_rate)

        # === 对比投影头（用分类特征）===
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # === 未来序列头：Linear + 1D Conv 时序平滑 ===
        self.seq_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, future_days),
        )
        self.seq_smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

        # === 统计量头（8维）===
        self.stat_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 8),
        )

    def _build_cls_head(self, per_scale_dim, hidden_dim, num_classes, dropout_rate):
        """子类重写此方法以定义不同的分类头结构。"""
        raise NotImplementedError

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)

        # 任务解耦
        cls_feat  = self.cls_proj(repr_vec)
        seq_feat  = self.seq_proj(repr_vec)
        stat_feat = self.stat_proj(repr_vec)

        logits   = self.main_head(cls_feat)
        proj     = self.projection(cls_feat)
        aux_seq  = self.seq_head(seq_feat)                          # (B, 50)
        aux_seq  = self.seq_smooth(aux_seq.unsqueeze(1)).squeeze(1)  # 时序平滑
        aux_stat = self.stat_head(stat_feat)                        # (B, 8)

        return logits, proj, aux_seq, aux_stat


class ScaleCentricMTLRelative(_ScaleCentricMTLRelativeBase):
    """
    多任务 V1：分类头沿用 ScaleCentricRelative 原始结构。
    
    cls_head: BN → Linear(128→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→3)
    
    model_type: 'scale_centric_mtl_relative'
    """
    def _build_cls_head(self, per_scale_dim, hidden_dim, num_classes, dropout_rate):
        return nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )


class ScaleCentricMTLRelativeV2(_ScaleCentricMTLRelativeBase):
    """
    多任务 V2：简化版分类头，无 BN，减少深度。
    
    cls_head: Linear(128→128) → ReLU → Dropout → Linear(128→3)
    
    model_type: 'scale_centric_mtl_relative_v2'
    """
    def _build_cls_head(self, per_scale_dim, hidden_dim, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )


class ScaleCentricMTLRelativeV2Ranking(_ScaleCentricMTLRelativeBase):
    """
    多任务 V3：用排序损失 + 趋势方向监督替代 MSE 序列预测。

    与 V2 的区别：
    - seq_head 改为输出 50 维排序分数（用于 RankingLoss）
    - 新增 trend_head: (128) → (3)，预测未来累计收益趋势方向
    - forward 返回 5 元组: (logits, proj, aux_rank_pred, aux_stat_pred, aux_trend_pred)
    
    RankingLoss 不要求精确数值预测，只要求预测值与真实值排序一致，
    避免了 MSE 下日收益率量级过小导致梯度消失的问题。
    TrendDirectionLoss 直接预测未来趋势方向，与主任务同量级。

    model_type: 'scale_centric_mtl_relative_v2_ranking'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=14):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # 覆盖 seq_head：输出排序分数（与 future_days 同维度，但不要求数值匹配）
        # 结构不变，但语义变为"排序分数"而非"数值预测"
        self.seq_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, future_days),
        )
        # 去掉 seq_smooth，因为排序不需要平滑

        # === 新增：趋势方向头 ===
        # 预测未来累计收益最终符号：下跌(0) / 震荡(1) / 上涨(2)
        self.trend_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),  # 3 分类
        )

    def _build_cls_head(self, per_scale_dim, hidden_dim, num_classes, dropout_rate):
        """与 V2 相同的简化分类头"""
        return nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)

        # 任务解耦
        cls_feat  = self.cls_proj(repr_vec)
        seq_feat  = self.seq_proj(repr_vec)
        stat_feat = self.stat_proj(repr_vec)

        logits      = self.main_head(cls_feat)
        proj        = self.projection(cls_feat)
        aux_rank    = self.seq_head(seq_feat)       # (B, 50) 排序分数
        aux_stat    = self.stat_head(stat_feat)      # (B, 8)
        aux_trend   = self.trend_head(seq_feat)      # (B, 3) 趋势方向 logits

        # 返回 5 元组，train.py 需适配
        return logits, proj, aux_rank, aux_stat, aux_trend

# 在 ScaleCentricRatio 的基础上，对所有 Ratio 列做 FFT 频域分解
# （低频趋势 + 高频波动），扩展输入特征维度。
# model_type: 'scale_centric_ratio_fft'
# ============================================================
class ScaleCentricRatioFFT(nn.Module):
    """
    方向三：Frequency-guided Ratio Decomposition。

    与 ScaleCentricRatio 的唯一区别是：
    - 在 _compute_ratio_features 计算完所有 Ratio 列后，
    - 对每一列在时间维度（60天）上做 FFT → 低频趋势 + 高频波动
    - 每列 Ratio 扩展为 2 列，daily_expanded_dim 变为 8 + N_ratio*2

    低频趋势：保留前 k 个频率分量重构（反映持续跑赢/跑输的趋势）
    高频波动：原始信号 - 低频趋势（反映短期扰动/反转信号）
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=16, fft_k=5):
        super().__init__()
        self.fft_k = fft_k  # 保留前 k 个频率
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim * 2  # 每列 Ratio 扩展为 low + high

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _fft_decompose(self, x, k):
        """
        x: (B, T, N) — 对最后一维的每列在时间维做 FFT 分解
        返回: (B, T, N*2) — low, high 交替排列
        """
        B, T, N = x.shape
        # (B, N, T) → FFT
        X = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # (B, N, T//2+1)
        # 低频：保留前 k 个频率
        X_low = torch.zeros_like(X)
        X_low[..., :k] = X[..., :k]
        x_low = torch.fft.irfft(X_low, n=T, dim=-1)      # (B, N, T)
        x_high = x.permute(0, 2, 1) - x_low               # (B, N, T)
        # 交替排列: [col0_low, col0_high, col1_low, col1_high, ...]
        result = torch.stack([x_low, x_high], dim=2)      # (B, N, 2, T)
        result = result.permute(0, 3, 1, 2).reshape(B, T, N * 2)  # (B, T, N*2)
        return result

    def _compute_ratio_features(self, daily_flat, macro):
        """同 ScaleCentricRatio，但输出后做 FFT 分解"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        # 有符号比例
        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        ratio_feats = torch.stack(parts, dim=-1)  # (B, 60, N_ratio)
        # 对每一列做 FFT 分解 → (B, 60, N_ratio*2)
        return self._fft_decompose(ratio_feats, self.fft_k)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_decomp = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_decomp], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# 方向一：Market-guided Cross Attention
# 在 ScaleCentricRatio 的基础上，增加 Cross Attention 模块。
# 三频融合后的表示作为 Q，宏观指数编码后的表示作为 K/V，
# 通过 Cross Attention 按需检索市场信息，残差连接融合。
# model_type: 'scale_centric_ratio_cross_attn'
# ============================================================
class ScaleCentricRatioCrossAttn(nn.Module):
    """
    方向一：Market-guided Cross Attention。

    与 ScaleCentricRatio 的区别：
    - 新增独立的 macro 编码器（与 day_encoder 结构一致）
    - 三频融合后的 fused 表示作为 Q，macro_repr 作为 K/V
    - Cross Attention 输出通过残差连接与 fused 相加
    - 之后再送入 temporal_encoder

    这样市场信息始终作为"被查询的外部知识"，不会与个股特征竞争表示空间。
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=16, cross_attn_heads=4):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)
        self.macro_encoder = mk(macro_dim)  # 独立编码 macro 序列

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # Cross Attention: fused→Q, macro_repr→K/V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=per_scale_dim,
            num_heads=cross_attn_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(per_scale_dim)

        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """同 ScaleCentricRatio"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)
        macro_repr = self._encode_full(macro,        self.macro_encoder)  # (B, 60, D)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)  # (B, 60, D)

        # Cross Attention: fused→Q, macro_repr→K/V
        attn_out, _ = self.cross_attn(
            query=fused,
            key=macro_repr,
            value=macro_repr,
        )
        fused = self.cross_norm(fused + attn_out)  # 残差连接

        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ScaleCentricRatioRelationMatrix(nn.Module):
    """
    Dynamic Market Relation Matrix（动态市场关系矩阵）。

    核心思想：
    将股票和市场视为两个独立的多维时间序列：
      stock_seq: (B, 60, 9)   — 个股日频特征
      macro_seq: (B, 60, 16)  — 市场指数特征
    不直接 concat，而是分别编码后计算二者的关系矩阵：
      M = stock_repr @ macro_repr^T   — (B, 60, 60)
    其中行=股票历史某一天，列=市场历史某一天，
    元素表示当前股票状态与过去市场状态之间的关联程度。

    然后使用 M 加权聚合市场信息：
      macro_agg = softmax(M) @ macro_repr   — (B, 60, D)
    
    最后将聚合后的市场信息与三频融合特征 + ratio 特征结合。

    与 ScaleCentricRatioCrossAttn 的区别：
    - CrossAttn 是对齐时间步的查询（fused→Q, macro→K/V）
    - Relation Matrix 是全时间双向交互（60×60 全连接矩阵），
      允许股票某一天关联市场所有天。
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=16):
        super().__init__()
        _rel_dim = 6 if macro_dim == 16 else 4
        daily_expanded_dim = 8 + _rel_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)
        self.macro_encoder = mk(macro_dim)  # 独立编码 macro 序列 → (B, 60, D)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 关系矩阵聚合后的市场信息投影层
        self.macro_proj = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 融合后的时序编码器
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """同 ScaleCentricRatio"""
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300, v300, a300, p500, v500, a500 = _extract_macro_cols(macro)

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS),
            0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS),
            0.01, 100.0)

        parts = [return_ratio300, return_ratio500]
        if _has_amp(macro):
            parts.extend([
                torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
                torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)

        # --- Step 1: 编码 ratio 特征并拼接 ---
        daily_flat = daily.reshape(B, 60, 8)
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        # --- Step 2: 三频分别编码 ---
        day_repr   = self._encode_full(day_seq,      self.day_encoder)   # (B, 60, D)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)  # (B, 60, D)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder) # (B, 60, D)
        macro_repr = self._encode_full(macro,        self.macro_encoder) # (B, 60, D)

        # --- Step 3: 三频融合 ---
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)  # (B, 60, D)

        # --- Step 4: 动态市场关系矩阵 ---
        # fused: (B, 60, D) 作为股票表示 Q
        # macro_repr: (B, 60, D) 作为市场表示 K
        # M = Q @ K^T / sqrt(D)  — (B, 60, 60)
        scale = math.sqrt(fused.size(-1))
        relation_matrix = torch.bmm(fused, macro_repr.transpose(1, 2)) / scale  # (B, 60, 60)

        # softmax 按行（股票时间步维度）归一化
        attn_weights = F.softmax(relation_matrix, dim=-1)  # (B, 60, 60)

        # 用关系矩阵加权聚合市场信息
        # attn_weights @ macro_repr: (B, 60, 60) @ (B, 60, D) → (B, 60, D)
        macro_agg = torch.bmm(attn_weights, macro_repr)  # (B, 60, D)

        # 投影层
        macro_agg = self.macro_proj(macro_agg)  # (B, 60, D)

        # --- Step 5: 融合市场信息 ---
        fused = fused + macro_agg  # 残差连接

        # --- Step 6: 时序编码 + 取最后时间步 ---
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ScaleCentricRelationMatrix(nn.Module):
    """
    Dynamic Market Relation Matrix（动态市场关系矩阵）—— 无 Ratio 版本。

    核心思想：
    将股票和市场视为两个独立的多维时间序列：
      stock_seq: (B, 60, 9)   — 个股日频特征（原始 8 维，不使用 Ratio）
      macro_seq: (B, 60, 16)  — 市场指数特征
    不直接 concat，而是分别编码后计算二者的关系矩阵：
      M = stock_repr @ macro_repr^T   — (B, 60, 60)
    其中行=股票历史某一天，列=市场历史某一天，
    元素表示当前股票状态与过去市场状态之间的关联程度。

    然后使用 M 加权聚合市场信息：
      macro_agg = softmax(M) @ macro_repr   — (B, 60, D)

    最后将聚合后的市场信息与三频融合特征结合。

    与 ScaleCentricRatioRelationMatrix 的区别：
    - 不使用 Ratio 特征，day_encoder 输入为原始 8 维
    - 日频数据直接编码为 stock_repr，不拼接 ratio 特征
    
    model_type: 'scale_centric_relation_matrix'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=5,
                 dropout_rate=0.3, num_layers=1, encoder_type='gru',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=16):
        super().__init__()

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(8)           # 原始 8 维，无 Ratio
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)
        self.macro_encoder = mk(macro_dim)   # 独立编码 macro 序列 → (B, 60, D)

        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 关系矩阵聚合后的市场信息投影层
        self.macro_proj = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 融合后的时序编码器
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)

        # --- Step 1: 三频分别编码（日频直接用原始 8 维） ---
        daily_flat = daily.reshape(B, 60, 8)  # (B, 60, 9)，原始数据，无 Ratio

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(daily_flat,   self.day_encoder)   # (B, 60, D)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)  # (B, 60, D)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder) # (B, 60, D)
        macro_repr = self._encode_full(macro,        self.macro_encoder) # (B, 60, D)

        # --- Step 2: 三频融合 ---
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)  # (B, 60, D)

        # --- Step 3: 动态市场关系矩阵 ---
        # fused: (B, 60, D) 作为股票表示 Q
        # macro_repr: (B, 60, D) 作为市场表示 K
        # M = Q @ K^T / sqrt(D)  — (B, 60, 60)
        scale = math.sqrt(fused.size(-1))
        relation_matrix = torch.bmm(fused, macro_repr.transpose(1, 2)) / scale  # (B, 60, 60)

        # softmax 按行（股票时间步维度）归一化
        attn_weights = F.softmax(relation_matrix, dim=-1)  # (B, 60, 60)

        # 用关系矩阵加权聚合市场信息
        macro_agg = torch.bmm(attn_weights, macro_repr)  # (B, 60, D)

        # 投影层
        macro_agg = self.macro_proj(macro_agg)  # (B, 60, D)

        # --- Step 4: 融合市场信息 ---
        fused = fused + macro_agg  # 残差连接

        # --- Step 5: 时序编码 + 取最后时间步 ---
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# 快速自测
# ============================================================
if __name__ == '__main__':
    daily = torch.randn(4, 12, 5, 9)
    weekly = torch.randn(4, 1, 12, 9)
    intra = torch.randn(4, 60, 8, 9)
    macro = torch.randn(4, 60, 16)

    for name, cls in [('FeatureCentricSimple', FeatureCentricSimple),
                       ('ScaleCentricSimple', ScaleCentricSimple),
                       ('BothCentricSimple', BothCentricSimple),
                       ('IntradayOnlyFlatten', IntradayOnlyFlatten),
                       ('ScaleCentricSimpleRelation', ScaleCentricSimpleRelation),
                       ('ScaleCentricFFTResidual', ScaleCentricFFTResidual),
                       ('ScaleCentricFFTLowCut', ScaleCentricFFTLowCut),
                       ('ScaleCentricMacroGate', ScaleCentricMacroGate),
                       ('ScaleCentricMacroGateDaily', ScaleCentricMacroGateDaily)]:
        model = cls(num_classes=5)
        # 新模型需要 macro 参数
        if name.startswith('ScaleCentricFFT') or name.startswith('ScaleCentricMacro'):
            out = model(daily, weekly, intra, macro)
        else:
            out = model(daily, weekly, intra)
        if isinstance(out, tuple):
            logits, proj = out
            n = sum(p.numel() for p in model.parameters())
            print(f'{name:>32s}: logits={list(logits.shape)}, proj={list(proj.shape)}, params={n:>10,}')
        else:
            n = sum(p.numel() for p in model.parameters())
            print(f'{name:>32s}: output={list(out.shape)}, params={n:>10,}')
