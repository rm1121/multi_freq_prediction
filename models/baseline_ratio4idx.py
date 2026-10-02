"""
4 指数宏观特征实验模型（macro4idx 56维）。

包含三个模型：
  1. ScaleCentricRatio4Idx        — 4 指数 ratio（实验 A）
  2. ScaleCentricRatio4IdxWide    — 4 指数 ratio + 广度信息（实验 B）
  3. ScaleCentricRatio4IdxCrossAttn — 4 指数 ratio + TCN市场编码 + 交叉注意力（实验 C）

macro4idx 列布局（每指数 14 维）：
  [0:open, 1:high, 2:low, 3:close, 4:pct_chg, 5:volume, 6:amt, 7:turn, 8:amplitude,
   9:up_ratio, 10:down_ratio, 11:up_limit_ratio, 12:down_limit_ratio, 13:rel_advance_decline]

4 指数拼接顺序：中证全指(0~13) / 中证1000(14~27) / 沪深300(28~41) / 中证500(42~55)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seqnets import TCNEncoder
from .baseline_simple import ScaleCentricMTLRelativeV2Ranking


# ============================================================
# 日内压缩器（复用，与 baseline_simple 一致）
# ============================================================
class IntradayCompressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim,
                          num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 3:
            x = x.reshape(B * 60, 13, 1)
        else:
            x = x.reshape(B * 60, 13, x.size(-1))
        out, _ = self.gru(x)          # (B*60, 8, hidden_dim)
        out = out[:, -1, :]            # (B*60, hidden_dim)
        out = self.dropout(out)
        return out.reshape(B, 60, -1)  # (B, 60, hidden_dim)


# ============================================================
# 编码器工厂
# ============================================================
def _make_encoder(encoder_type, input_size, hidden_size, num_layers,
                  dropout, kernel_size=3, num_levels=4):
    if encoder_type in ('gru', 'lstm'):
        rnn_cls = nn.GRU if encoder_type == 'gru' else nn.LSTM
        return _RNNWrapper(rnn_cls, input_size, hidden_size,
                           num_layers, dropout)
    elif encoder_type == 'tcn':
        return _TCNWrapper(input_size, hidden_size, kernel_size,
                           num_levels, dropout)
    else:
        raise ValueError(f'Unknown encoder: {encoder_type}')


class _RNNWrapper(nn.Module):
    def __init__(self, rnn_cls, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = rnn_cls(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out, None


class _TCNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_levels, dropout):
        super().__init__()
        self.tcn = TCNEncoder(input_size, hidden_size,
                              kernel_size=kernel_size,
                              num_levels=num_levels,
                              dropout_rate=dropout)

    def forward(self, x):
        # TCNEncoder 返回 (B, hidden_dim, T)，转置为 (B, T, hidden_dim)
        out = self.tcn(x)
        return out.transpose(1, 2), None


# ============================================================
# 实验 A：4 指数 ratio
# ============================================================
class ScaleCentricRatio4Idx(nn.Module):
    """
    实验 A：4 指数 ratio 特征。

    从 macro4idx (56维) 中提取 4 指数的 pct_chg/volume/amplitude，
    计算有符号比例，共 12 维 ratio 特征。
    daily_expanded_dim = 8 + 12 = 20

    无多任务辅助头，返回 (logits, proj)。
    多任务版本见 _MTL 子类。

    model_type: 'scale_centric_ratio_4idx'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        super().__init__()
        # 子类可覆盖 _ratio_dim 来扩展 ratio 特征维度
        if not hasattr(self, '_ratio_dim'):
            self._ratio_dim = 12  # 4指数 × 3 ratio = 12

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)

        # 线性投影分组：stock 特征和 ratio 特征分别映射再拼接
        self.stock_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.ratio_proj = nn.Sequential(
            nn.Linear(self._ratio_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # day_encoder 输入 = 64(stock) + 32(ratio) = 96
        self.day_encoder = mk(96)

        if not getattr(self, '_skip_week_encoder', False):
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

        # head 加 BatchNorm 稳定训练
        self.head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

    # ---- ratio 计算 ----
    def _compute_ratio_features(self, daily_flat, macro):
        """
        daily_flat: (B, 60, 8)
        macro:      (B, 60, 56) — 4指数 × 14维
        → (B, 60, 12)
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        parts = []
        for idx_offset in [0, 14, 28, 42]:
            idx_pct_chg = macro[:, :, idx_offset + 4]
            idx_volume  = macro[:, :, idx_offset + 5]
            idx_amp     = macro[:, :, idx_offset + 8]

            # 有符号比例：pct_chg
            sign = torch.sign(stock_pct_chg * idx_pct_chg)
            return_ratio = sign * torch.clamp(
                (stock_pct_chg.abs() + self._EPS) / (idx_pct_chg.abs() + self._EPS),
                0.01, 100.0)
            parts.append(return_ratio)

            # amplitude ratio（纯正数）
            parts.append(torch.clamp(
                stock_amp / (idx_amp + self._EPS), 0.01, 100.0))

            # volume ratio（纯正数）
            parts.append(torch.clamp(
                stock_volume / (idx_volume + self._EPS), 0.01, 100.0))

        # log1p 处理：正数部分 log(1+x)，负数部分保留符号*log(1+|x|)
        # 避免 sign=-1 时 log1p(负数) 产生 NaN
        ratio = torch.stack(parts, dim=-1)  # (B, 60, 12)
        return torch.sign(ratio) * torch.log1p(ratio.abs())

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)

        # 线性投影分组：stock 和 ratio 各自独立映射再拼接
        stock_feat = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)  # (B, 60, 96)

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
# 实验 A-WeekRatio：日频 ratio + 周频 ratio 双投影分组
# ============================================================
class ScaleCentricRatio4IdxWeekRatio(ScaleCentricRatio4Idx):
    """
    在 ScaleCentricRatio4Idx（日频 ratio 投影分组）基础上，
    对周频数据也做同样的 ratio 投影分组处理。

    日频：stock_proj(9→64) + ratio_proj(12→32) = 96 → day_encoder
    周频：week_stock_proj(9→32) + week_ratio_proj(12→16) = 48 → week_encoder

    周频 macro 从日频 macro 降采样（每 5 天取最后一天）。
    周频 stock 来自 weekly (B, 12, 8)。

    model_type: 'scale_centric_ratio_4idx_week_ratio'
    """
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

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)

        # 周频投影分组
        self.week_stock_proj = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.week_ratio_proj = nn.Sequential(
            nn.Linear(12, 16),  # 4指数 × 3 ratio = 12
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # 覆盖父类的 week_encoder：输入 = 32(stock) + 16(ratio) = 48
        self.week_encoder = mk(48)

    def _compute_week_macro(self, macro):
        """
        从日频 macro 计算周频统计量 macro。
        macro: (B, 60, 56) → (B, 12, 56)

        每 5 天一组，对每指数 14 维做不同聚合：
          - pct_chg(4): 周累计收益（prod(1+x) - 1）
          - volume(5)/amt(6): 周均值
          - turn(7): 周均值
          - amplitude(8): 周均值
          - open/high/low/close(0~3): 周最后一天（价格已在预处理归一化）
          - up_ratio(9)/down_ratio(10): 周均值
          - up_limit_ratio(11)/down_limit_ratio(12): 周均值
          - rel_advance_decline(13): 周均值
        """
        B, T, D = macro.shape
        macro_5day = macro.reshape(B, 12, 5, D)  # (B, 12, 5, 56)

        # 每指数 14 维的聚合方式
        # 价格类(0~3): 取最后一天
        price = macro_5day[:, :, -1, :14]  # (B, 12, 14)

        # pct_chg(4): 周累计 (1+x1)*(1+x2)*... - 1
        pct_chg = macro_5day[:, :, :, 4]  # (B, 12, 5)
        cum_pct = torch.prod(1 + pct_chg, dim=-1, keepdim=True) - 1  # (B, 12, 1)

        # volume(5), amt(6), turn(7), amplitude(8): 周均值
        vol_amt = macro_5day[:, :, :, [5, 6, 7, 8]].mean(dim=2)  # (B, 12, 4)

        # up_ratio(9), down_ratio(10), up_limit_ratio(11), down_limit_ratio(12): 周均值
        breadth = macro_5day[:, :, :, [9, 10, 11, 12]].mean(dim=2)  # (B, 12, 4)

        # rel_advance_decline(13): 周均值
        rel_adv = macro_5day[:, :, :, 13].mean(dim=-1, keepdim=True)  # (B, 12, 1)

        week_feats = torch.cat([
            price[:, :, :4],            # 0~3: open,high,low,close（最后一天）
            cum_pct,                     # 4: 周累计收益
            vol_amt,                     # 5~8: volume,amt,turn,amp（周均值）
            breadth,                     # 9~12: up/down_ratio,up/down_limit（周均值）
            rel_adv,                     # 13: rel_advance_decline（周均值）
        ], dim=-1)  # (B, 12, 14)

        # 对 4 指数重复
        # 由于 4 指数结构一致，可以直接用 view 扩展
        # macro 的 56 维 = 4 × 14
        week_macro_parts = []
        for idx_offset in range(0, 56, 14):
            idx_data = macro_5day[:, :, :, idx_offset:idx_offset + 14]
            # 对该指数做同样的聚合
            p = idx_data[:, :, -1, :4]                                    # price
            cp = torch.prod(1 + idx_data[:, :, :, 4], dim=-1, keepdim=True) - 1  # cum pct
            va = idx_data[:, :, :, [5, 6, 7, 8]].mean(dim=2)             # vol/amt/turn/amp
            br = idx_data[:, :, :, [9, 10, 11, 12]].mean(dim=2)          # breadth
            ra = idx_data[:, :, :, 13].mean(dim=-1, keepdim=True)        # rel_adv
            week_macro_parts.append(torch.cat([p, cp, va, br, ra], dim=-1))

        return torch.cat(week_macro_parts, dim=-1)  # (B, 12, 56)

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # === 日频处理（与父类一致） ===
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        stock_feat = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)  # (B, 60, 96)

        # === 周频处理 ===
        week_stock = weekly.squeeze(1)  # (B, 12, 8)
        week_macro = self._compute_week_macro(macro)  # (B, 12, 56)

        # 周频 ratio：使用周频专用的列索引
        week_ratio_feats = self._compute_weekly_ratio_features(week_stock, week_macro)  # (B, 12, 12)

        week_stock_feat = self.week_stock_proj(week_stock)     # (B, 12, 32)
        week_ratio_feat = self.week_ratio_proj(week_ratio_feats)  # (B, 12, 16)
        week_seq = torch.cat([week_stock_feat, week_ratio_feat], dim=-1)  # (B, 12, 48)

        # 周频上采样到日频（重复 5 次）
        week_seq = week_seq.repeat_interleave(5, dim=1)  # (B, 60, 48)

        # === 日内处理 ===
        intra_daily = self.intra_compressor(intraday)

        # === 三频编码 ===
        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def _compute_weekly_ratio_features(self, week_stock, week_macro):
        """
        周频专用的 ratio 计算（列布局与日频不同）。

        周 stock(12,9): [open_avg, high_avg, low_avg, close_avg,
                         volume_avg, amt_avg, cum_return, turn_avg, amplitude_avg]
          - cum_return(6) 对应日频 pct_chg(4)
          - volume_avg(4)  对应日频 volume(5)
          - amplitude_avg(8) 对应日频 amplitude(8)（索引相同）

        周 macro(12,56): 与日频 macro 列布局一致（4指数×14维）
          每指数 14 维：open,high,low,close, pct_chg,volume,amt,turn,amplitude,
                        up_ratio,down_ratio,up_limit_ratio,down_limit_ratio,rel_advance_decline

        输出: (B, 12, 12) — 4指数 × 3 ratio
        """
        stock_cum_return = week_stock[:, :, 5]  # cum_return
        stock_volume     = week_stock[:, :, 4]  # volume_avg
        stock_amp        = week_stock[:, :, 7]  # amplitude_avg

        parts = []
        for idx_offset in [0, 14, 28, 42]:
            idx_cum_ret = week_macro[:, :, idx_offset + 4]
            idx_volume  = week_macro[:, :, idx_offset + 5]
            idx_amp     = week_macro[:, :, idx_offset + 8]

            sign = torch.sign(stock_cum_return * idx_cum_ret)
            return_ratio = sign * torch.clamp(
                (stock_cum_return.abs() + self._EPS) / (idx_cum_ret.abs() + self._EPS),
                0.01, 100.0)
            parts.append(return_ratio)
            parts.append(torch.clamp(
                stock_amp / (idx_amp + self._EPS), 0.01, 100.0))
            parts.append(torch.clamp(
                stock_volume / (idx_volume + self._EPS), 0.01, 100.0))

        ratio = torch.stack(parts, dim=-1)  # (B, 12, 12)
        return torch.sign(ratio) * torch.log1p(ratio.abs())


class ScaleCentricRatio4IdxWeekRatioV2(ScaleCentricRatio4Idx):
    """
    改进版：周频 ratio 直接使用预处理好的周频 macro（macro_weekly），
    不再从日频 macro 降采样。

    forward 接受 (daily, weekly, intraday, macro, macro_weekly)。

    day_seq = stock(64) + day_ratio(32) = 96 → day_encoder
    week_seq = week_stock(32) + week_ratio(16) = 48 → week_encoder（不重复上采样）
    day_repr(60,128) → 每5天池化 → (12,128) → 与 week_repr(12,128) concat → 取最后一周

    model_type: 'scale_centric_ratio_4idx_week_ratio_v2'
    """
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

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)

        # 周频投影分组
        self.week_stock_proj = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.week_ratio_proj = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.week_encoder = mk(48)

        # 覆盖父类的 week_encoder 和 day_encoder（day 不变）
        # 删除父类创建的 week_encoder（mk(8)），改用上面的 mk(48)
        # 注意：父类 __init__ 已创建 week_encoder=mk(8)，我们覆盖它
        self.week_encoder = mk(48)

    def forward(self, daily, weekly, intraday, macro, macro_weekly=None):
        """接受 macro_weekly 作为第 5 个参数。"""
        repr_vec = self._compute_repr(daily, weekly, intraday, macro, macro_weekly)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj

    def _compute_repr(self, daily, weekly, intraday, macro, macro_weekly=None):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # === 日频处理 ===
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        stock_feat = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)  # (B, 60, 96)

        # === 周频处理（使用预计算的周频 macro） ===
        week_stock = weekly.squeeze(1)  # (B, 12, 8)
        if macro_weekly is not None:
            week_macro = macro_weekly  # (B, 12, 56) — 直接用预处理好的周统计量
        else:
            # 降级：简单取每5天最后一天（兼容旧配置，但不会很好）
            week_macro = macro.reshape(B, 12, 5, -1)[:, :, -1, :]  # (B, 12, 56)

        # 周频 ratio：使用周频专用的索引（列布局不同）
        week_ratio_feats = self._compute_weekly_ratio_features(week_stock, week_macro)  # (B, 12, 12)
        week_stock_feat = self.week_stock_proj(week_stock)     # (B, 12, 32)
        week_ratio_feat = self.week_ratio_proj(week_ratio_feats)  # (B, 12, 16)
        week_seq = torch.cat([week_stock_feat, week_ratio_feat], dim=-1)  # (B, 12, 48)

        # === 日内处理 ===
        intra_daily = self.intra_compressor(intraday)

        # === 三频编码（周频不上采样，日频下采样对齐） ===
        day_repr   = self._encode_full(day_seq,      self.day_encoder)   # (B, 60, D)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)  # (B, 12, D)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder) # (B, 60, D)

        # 日频/日内下采样到 12 步（每 5 天池化）
        day_repr_pooled   = day_repr.reshape(B, 12, 5, -1).mean(dim=2)    # (B, 12, D)
        intra_repr_pooled = intra_repr.reshape(B, 12, 5, -1).mean(dim=2)  # (B, 12, D)

        # 在 12 周维度上拼接
        stacked = torch.cat([week_repr, day_repr_pooled, intra_repr_pooled], dim=-1)  # (B, 12, 3D)
        fused = self.scale_fusion(stacked)  # (B, 12, D)

        # 时序编码（12 步）
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]  # 取最后一周

    def _compute_weekly_ratio_features(self, week_stock, week_macro):
        """
        周频专用的 ratio 计算（列布局与日频不同）。

        周 stock(12,9): [open_avg, high_avg, low_avg, close_avg,
                         volume_avg, amt_avg, cum_return, turn_avg, amplitude_avg]
          - cum_return(6) 对应日频 pct_chg(4)
          - volume_avg(4)  对应日频 volume(5)
          - amplitude_avg(8) 对应日频 amplitude(8)（索引相同）

        周 macro(12,56): 与日频 macro 列布局一致（4指数×14维）
          每指数 14 维：open,high,low,close, pct_chg,volume,amt,turn,amplitude,
                        up_ratio,down_ratio,up_limit_ratio,down_limit_ratio,rel_advance_decline

        输出: (B, 12, 12) — 4指数 × 3 ratio
        """
        # 周 stock 特征（列索引不同）
        stock_cum_return = week_stock[:, :, 5]  # cum_return
        stock_volume     = week_stock[:, :, 4]  # volume_avg
        stock_amp        = week_stock[:, :, 7]  # amplitude_avg

        parts = []
        for idx_offset in [0, 14, 28, 42]:
            # 周 macro 的 pct_chg(4) 已经是周累计收益（预处理中计算）
            idx_cum_ret = week_macro[:, :, idx_offset + 4]
            idx_volume  = week_macro[:, :, idx_offset + 5]
            idx_amp     = week_macro[:, :, idx_offset + 8]

            # cum_return ratio（有符号比例）
            sign = torch.sign(stock_cum_return * idx_cum_ret)
            return_ratio = sign * torch.clamp(
                (stock_cum_return.abs() + self._EPS) / (idx_cum_ret.abs() + self._EPS),
                0.01, 100.0)
            parts.append(return_ratio)

            # amplitude ratio
            parts.append(torch.clamp(
                stock_amp / (idx_amp + self._EPS), 0.01, 100.0))

            # volume ratio
            parts.append(torch.clamp(
                stock_volume / (idx_volume + self._EPS), 0.01, 100.0))

        ratio = torch.stack(parts, dim=-1)  # (B, 12, 12)
        return torch.sign(ratio) * torch.log1p(ratio.abs())


# ============================================================
# 实验 A-WeekRatio V3：周 ratio 上采样到 60 步，按 baseline3 方式三频融合
# ============================================================
class ScaleCentricRatio4IdxWeekRatioV3(ScaleCentricRatio4Idx):
    """
    修正版：周 ratio 在 12 步空间计算并投影拼接后，**上采样到 60 步**，
    再与日 ratio 版本、分时数据一起按 baseline3 的 ScaleCentricSimple 方式处理。

    关键区别 vs V1/V2：
      - V1: 周上采样到 60 → 编码 → 融合（正确思路，但实现不彻底）
      - V2: 周 12 步编码，日/日内下采样到 12 步融合（改变了 baseline3 对齐方式）
      - V3: 周在 12 步空间做 stock+ratio 投影 → 上采样到 60 → 编码 → 与日/日内 60 步融合

    数据流：
      day:    stock(9→64) + ratio(12→32) = (B,60,96)  → day_encoder → (B,60,d)
      week:   stock(9→32) + ratio(12→16) = (B,12,48)  → repeat_interleave(5) → (B,60,48) → week_encoder → (B,60,d)
      intra:  (B,60,13,8) → compressor → (B,60,intra_hidden) → intra_encoder → (B,60,d)

      cat([week, day, intra], dim=-1) → (B,60,3d)
      → scale_fusion → (B,60,d) → temporal_encoder → last → head

    forward 接受 (daily, weekly, intraday, macro, macro_weekly)。

    model_type: 'scale_centric_ratio_4idx_week_ratio_v3'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        # 告诉父类不要创建默认的 week_encoder（我们自定义）
        self._skip_week_encoder = True
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)

        # 周频投影分组
        self.week_stock_proj = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.week_ratio_proj = nn.Sequential(
            nn.Linear(12, 16),  # 4指数 × 3 ratio = 12
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # 自定义 week_encoder：输入 = 32(stock) + 16(ratio) = 48
        self.week_encoder = mk(48)

    def _compute_week_macro(self, macro):
        """
        从日频 macro 计算周频统计量 macro。
        macro: (B, 60, 56) → (B, 12, 56)

        每 5 天一组，对每指数 14 维做不同聚合：
          - pct_chg(4): 周累计收益（prod(1+x) - 1）
          - volume(5)/amt(6): 周均值
          - turn(7): 周均值
          - amplitude(8): 周均值
          - open/high/low/close(0~3): 周最后一天（价格已在预处理归一化）
          - up_ratio(9)/down_ratio(10): 周均值
          - up_limit_ratio(11)/down_limit_ratio(12): 周均值
          - rel_advance_decline(13): 周均值
        """
        B, T, D = macro.shape
        macro_5day = macro.reshape(B, 12, 5, D)  # (B, 12, 5, 56)

        week_macro_parts = []
        for idx_offset in range(0, 56, 14):
            idx_data = macro_5day[:, :, :, idx_offset:idx_offset + 14]
            p = idx_data[:, :, -1, :4]                                    # price
            cp = torch.prod(1 + idx_data[:, :, :, 4], dim=-1, keepdim=True) - 1  # cum pct
            va = idx_data[:, :, :, [5, 6, 7, 8]].mean(dim=2)             # vol/amt/turn/amp
            br = idx_data[:, :, :, [9, 10, 11, 12]].mean(dim=2)          # breadth
            ra = idx_data[:, :, :, 13].mean(dim=-1, keepdim=True)        # rel_adv
            week_macro_parts.append(torch.cat([p, cp, va, br, ra], dim=-1))

        return torch.cat(week_macro_parts, dim=-1)  # (B, 12, 56)

    def forward(self, daily, weekly, intraday, macro, macro_weekly=None):
        """接受 macro_weekly 作为第 5 个参数。"""
        repr_vec = self._compute_repr(daily, weekly, intraday, macro, macro_weekly)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj

    def _compute_repr(self, daily, weekly, intraday, macro, macro_weekly=None):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # === 日频处理（同父类 ScaleCentricRatio4Idx） ===
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        stock_feat = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)  # (B, 60, 96)

        # === 周频处理 ===
        week_stock = weekly.squeeze(1)  # (B, 12, 8)
        if macro_weekly is not None:
            week_macro = macro_weekly  # (B, 12, 56) — 直接用预处理好的周统计量
        else:
            # 降级：从日频 macro 降采样
            week_macro = self._compute_week_macro(macro)  # (B, 12, 56)

        # 周频 ratio：在 12 步空间计算
        week_ratio_feats = self._compute_weekly_ratio_features(week_stock, week_macro)  # (B, 12, 12)

        # 周频投影拼接（在 12 步空间）
        week_stock_feat = self.week_stock_proj(week_stock)       # (B, 12, 32)
        week_ratio_feat = self.week_ratio_proj(week_ratio_feats) # (B, 12, 16)
        week_seq_12 = torch.cat([week_stock_feat, week_ratio_feat], dim=-1)  # (B, 12, 48)

        # ★ 关键：在 12 步空间拼接后，上采样到 60 步（而不是先编码再上采样）
        week_seq = week_seq_12.repeat_interleave(5, dim=1)  # (B, 60, 48)

        # === 日内处理 ===
        intra_daily = self.intra_compressor(intraday)

        # === 三频编码（全部在 60 步空间） — 与 baseline3 一致 ===
        day_repr   = self._encode_full(day_seq,     self.day_encoder)    # (B, 60, d)
        week_repr  = self._encode_full(week_seq,    self.week_encoder)   # (B, 60, d)
        intra_repr = self._encode_full(intra_daily, self.intra_encoder)  # (B, 60, d)

        # === 逐时间步跨尺度融合 — 与 baseline3 一致 ===
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)  # (B, 60, 3*d)
        fused = self.scale_fusion(stacked)                               # (B, 60, d)
        fused_out, _ = self.temporal_encoder(fused)   # (B, 60, d)
        return fused_out[:, -1, :]                     # (B, d)

    def _compute_weekly_ratio_features(self, week_stock, week_macro):
        """
        周频 ratio 计算（列布局与日频不同）。

        周 stock(12,9): [open_avg, high_avg, low_avg, close_avg,
                         volume_avg, amt_avg, cum_return, turn_avg, amplitude_avg]
          - cum_return(6) 对应日频 pct_chg(4)
          - volume_avg(4)  对应日频 volume(5)
          - amplitude_avg(8) 对应日频 amplitude(8)

        周 macro(12,56): 与日频 macro 列布局一致（4指数×14维）

        输出: (B, 12, 12) — 4指数 × 3 ratio
        """
        stock_cum_return = week_stock[:, :, 5]  # cum_return
        stock_volume     = week_stock[:, :, 4]  # volume_avg
        stock_amp        = week_stock[:, :, 7]  # amplitude_avg

        parts = []
        for idx_offset in [0, 14, 28, 42]:
            idx_cum_ret = week_macro[:, :, idx_offset + 4]
            idx_volume  = week_macro[:, :, idx_offset + 5]
            idx_amp     = week_macro[:, :, idx_offset + 8]

            # cum_return ratio（有符号比例）
            sign = torch.sign(stock_cum_return * idx_cum_ret)
            return_ratio = sign * torch.clamp(
                (stock_cum_return.abs() + self._EPS) / (idx_cum_ret.abs() + self._EPS),
                0.01, 100.0)
            parts.append(return_ratio)

            # amplitude ratio
            parts.append(torch.clamp(
                stock_amp / (idx_amp + self._EPS), 0.01, 100.0))

            # volume ratio
            parts.append(torch.clamp(
                stock_volume / (idx_volume + self._EPS), 0.01, 100.0))

        ratio = torch.stack(parts, dim=-1)  # (B, 12, 12)
        return torch.sign(ratio) * torch.log1p(ratio.abs())


# ============================================================
# 实验 A-MTL：4 指数 ratio + 多任务损失（ranking + trend）
# ============================================================
class ScaleCentricRatio4IdxMTL(ScaleCentricRatio4Idx):
    """
    在 ScaleCentricRatio4Idx（新版本 4 指数 ratio + 投影分组 + log1p + BN）基础上，
    增加多任务辅助头（排序 + 趋势方向），forward 返回 5 元组。

    与 train.py 中的 ranking/trend 损失兼容。

    model_type: 'scale_centric_ratio_4idx_mtl'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56, future_days=50):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        # 任务解耦投影
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

        # 分类头（加 BN，复用父类风格）
        self.main_head = nn.Sequential(
            nn.BatchNorm1d(per_scale_dim),
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # 对比投影头
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 排序头（用于 RankingLoss）
        self.seq_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, future_days),
        )

        # 统计量头
        self.stat_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 8),
        )

        # 趋势方向头
        self.trend_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)

        # 任务解耦
        cls_feat  = self.cls_proj(repr_vec)
        seq_feat  = self.seq_proj(repr_vec)
        stat_feat = self.stat_proj(repr_vec)

        logits      = self.main_head(cls_feat)
        proj        = self.projection(cls_feat)
        aux_rank    = self.seq_head(seq_feat)       # (B, 50)
        aux_stat    = self.stat_head(stat_feat)      # (B, 8)
        aux_trend   = self.trend_head(seq_feat)      # (B, 3)

        return logits, proj, aux_rank, aux_stat, aux_trend


# ============================================================
# 实验 B：4 指数 ratio + 市场广度信息
# ============================================================
class ScaleCentricRatio4IdxWide(ScaleCentricRatio4Idx):
    """
    实验 B：4 指数 ratio + 市场广度信息。

    在 ratio 基础上，额外将每指数的 up_ratio(9), down_ratio(10),
    rel_advance_decline(13) 提取为 breadth 特征（12维），通过独立的
    breadth_proj(12→16) 投影，再与 stock_feat(64) + ratio_feat(32) 拼接。

    day_seq = 64(stock) + 32(ratio) + 16(breadth) = 112

    model_type: 'scale_centric_ratio_4idx_wide'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        # 设置 _ratio_dim=12（仅 ratio，不含 breadth）
        self._ratio_dim = 12
        # 先调父类，父类会创建 stock_proj(9→64), ratio_proj(12→32),
        # day_encoder(96) 等。我们之后再覆盖 day_encoder
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        # 新增 breadth 投影层
        self.breadth_proj = nn.Sequential(
            nn.Linear(12, 16),  # 4指数 × 3 breadth = 12
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        # 重新设置 day_encoder：输入 = 64(stock) + 32(ratio) + 16(breadth) = 112
        self.day_encoder = mk(112)

    def _compute_breadth_features(self, macro):
        """
        提取 12 维广度信息并做 log1p 压缩。

        macro: (B, 60, 56) — 4指数 × 14维
        → (B, 60, 12): [up_ratio×4, down_ratio×4, rel_advance_decline×4] 带 log1p
        """
        parts = []
        for idx_offset in [0, 14, 28, 42]:
            parts.append(macro[:, :, idx_offset + 9])   # up_ratio
            parts.append(macro[:, :, idx_offset + 10])  # down_ratio
            parts.append(macro[:, :, idx_offset + 13])  # rel_advance_decline

        breadth = torch.stack(parts, dim=-1)  # (B, 60, 12)
        return torch.sign(breadth) * torch.log1p(breadth.abs())

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征（12维）+ 广度特征（12维）
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        breadth_feats = self._compute_breadth_features(macro)

        # 2. 三路独立投影
        stock_feat   = self.stock_proj(daily_flat)     # (B, 60, 64)
        ratio_feat   = self.ratio_proj(ratio_feats)     # (B, 60, 32)
        breadth_feat = self.breadth_proj(breadth_feats)  # (B, 60, 16)
        day_seq = torch.cat([stock_feat, ratio_feat, breadth_feat], dim=-1)  # (B, 60, 112)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 实验 B-对照：ratio + breadth 合并投影（方案 A）
# ============================================================
class ScaleCentricRatio4IdxWideConcat(ScaleCentricRatio4IdxWide):
    """
    实验 B 的对照实验：将 ratio(12) 与 breadth(12) 拼接后一起投影。

    与 Wide 版本的区别：
    - Wide:  stock(9→64) + ratio(12→32) + breadth(12→16) = 112
    - Concat: stock(9→64) + ratio_breadth(24→32)          = 96

    day_encoder 输入 = 96，与父类 ScaleCentricRatio4Idx 一致，无需重设。

    model_type: 'scale_centric_ratio_4idx_wide_concat'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        # 调父类 ScaleCentricRatio4IdxWide，它会设置 self._ratio_dim=12、
        # 创建 stock_proj(9→64)、breadth_proj(12→16) 等，并重设 day_encoder(112)。
        # 我们随后再覆盖这些组件。
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        # 覆盖 breadth_proj：改为 ratio(12) + breadth(12) = 24 → 32
        self.breadth_proj = nn.Sequential(
            nn.Linear(24, 32),  # ratio(12) + breadth(12) = 24
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # day_encoder 恢复为 96 维（64 stock + 32 ratio_breadth）
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder = mk(96)

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征 + 广度特征
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        breadth_feats = self._compute_breadth_features(macro)

        # 2. ratio(12) 与 breadth(12) 拼接后一起投影
        stock_feat    = self.stock_proj(daily_flat)                    # (B, 60, 64)
        ratio_breadth = torch.cat([ratio_feats, breadth_feats], dim=-1) # (B, 60, 24)
        ratio_breadth_feat = self.breadth_proj(ratio_breadth)           # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_breadth_feat], dim=-1)  # (B, 60, 96)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 实验 B-消融：只用 breadth（wide），无 ratio
# ============================================================
class ScaleCentricRatio4IdxWideOnly(ScaleCentricRatio4Idx):
    """
    实验 B 的消融对照：去掉 ratio，只用 stock + breadth。

    与 Wide 版本的区别：
    - Wide:      stock(9→64) + ratio(12→32) + breadth(12→16) = 112
    - WideOnly:  stock(9→64) + breadth(12→16)                = 80

    用于消融实验，验证：breadth（市场广度）本身是否已包含 ratio 的信息？

    day_encoder 输入 = 80。

    注意：继承 ScaleCentricRatio4Idx，ratio_proj 仍会被父类创建但不会使用
    （WideOnly 的 _compute_repr 完全覆盖了父类逻辑，不涉及 ratio）。
    model_type: 'scale_centric_ratio_4idx_wide_only'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56):
        # 设置 _ratio_dim=12（父类默认值），让父类正常创建 ratio_proj。
        # WideOnly 的 _compute_repr 完全覆盖了父类逻辑，不会使用 ratio_proj，
        # 但保持父类初始化的一致性，避免 Linear(0, 32) 的边界情况。
        self._ratio_dim = 12
        # 先调父类，父类会创建 stock_proj(9→64)，但不创建 ratio_proj
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, macro_dim=macro_dim)

        # 新增 breadth 投影层
        self.breadth_proj = nn.Sequential(
            nn.Linear(12, 16),  # 4指数 × 3 breadth = 12
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 重新设置 day_encoder：输入 = 64(stock) + 16(breadth) = 80
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder = mk(80)

    def _compute_breadth_features(self, macro):
        """
        提取 12 维广度信息并做 log1p 压缩。

        macro: (B, 60, 56) — 4指数 × 14维
        → (B, 60, 12): [up_ratio×4, down_ratio×4, rel_advance_decline×4] 带 log1p
        """
        parts = []
        for idx_offset in [0, 14, 28, 42]:
            parts.append(macro[:, :, idx_offset + 9])   # up_ratio
            parts.append(macro[:, :, idx_offset + 10])  # down_ratio
            parts.append(macro[:, :, idx_offset + 13])  # rel_advance_decline

        breadth = torch.stack(parts, dim=-1)  # (B, 60, 12)
        return torch.sign(breadth) * torch.log1p(breadth.abs())

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. 广度特征
        breadth_feats = self._compute_breadth_features(macro)

        # 2. 两路独立投影：stock + breadth（无 ratio）
        stock_feat    = self.stock_proj(daily_flat)       # (B, 60, 64)
        breadth_feat  = self.breadth_proj(breadth_feats)   # (B, 60, 16)
        day_seq = torch.cat([stock_feat, breadth_feat], dim=-1)  # (B, 60, 80)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 实验 C：4 指数 ratio + TCN 市场状态编码 + 交叉注意力融合
# ============================================================
class ScaleCentricRatio4IdxCrossAttn(ScaleCentricRatio4Idx):
    """
    实验 C：引入市场状态编码器，通过交叉注意力融合市场与个股信息。

    设计方案：
    ┌─────────────────────────────────────────────────────────┐
    │  市场状态编码（Market State Encoder）                       │
    │                                                         │
    │  macro (B,60,56) → TCN → (B,60,D_market)                │
    │   其中 D_market = per_scale_dim                          │
    │                                                         │
    │  每时间步输出是一个 128 维的市场状态向量，代表该时刻            │
    │  全市场（4 指数）的综合状态。                                │
    └─────────────────────────────────────────────────────────┘

    融合方式：
      Q = 三频融合后的 fused (B,60,D)
      K/V = 市场状态编码序列 (B,60,D)
      CrossAttn(Q, K, V) → attn_out (B,60,D)
      fused = LayerNorm(fused + attn_out)  # 残差连接
      → temporal_encoder → repr_vec

    设计思路：
    1. 市场状态编码器将 56 维原始宏观数据压缩为 128 维市场状态向量
    2. 交叉注意力让个股特征（fused）"查询"最相关的市场状态
    3. 残差连接保证个股信息不会丢失
    4. 市场信息始终以"被查询的外部知识"角色参与，不会干扰个股表示空间

    model_type: 'scale_centric_ratio_4idx_cross_attn'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 macro_dim=56, cross_attn_heads=4):
        # 先调父类初始化（会设置 day_encoder/week_encoder/intra_compressor 等）
        # 但我们需要额外加 market_encoder，且覆盖 scale_fusion 后的逻辑
        # 因此调完父类后需要修改后续部分
        # 最简单：手动设置，不调 super().__init__
        # 直接复制父类初始化逻辑 + 新增模块
        self._EPS = 1e-8

        nn.Module.__init__(self)

        self.macro_dim = macro_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)

        # 线性投影分组：stock 特征和 ratio 特征分别映射再拼接
        self.stock_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.ratio_proj = nn.Sequential(
            nn.Linear(12, 32),   # CrossAttn 固定 12 维 ratio
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.day_encoder = mk(96)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # === 市场状态编码器 ===
        # 输入: (B, 60, 56) 原始宏观数据
        # 输出: (B, 60, per_scale_dim) 市场状态序列
        self.market_encoder = _make_encoder(
            encoder_type, macro_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # === 三频融合 ===
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # === 交叉注意力：fused→Q, market_state→K/V ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=per_scale_dim,
            num_heads=cross_attn_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(per_scale_dim)

        # === 时序编码器（融合后） ===
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # === 分类头 ===
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

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征 + 线性投影分组
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        stock_feat = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)  # (B, 60, 96)

        # 2. 三频编码
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)   # (B, 60, D)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)  # (B, 60, D)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder) # (B, 60, D)

        # 3. 市场状态编码
        #    macro (B, 60, 56) → TCN → market_state (B, 60, D)
        market_state, _ = self.market_encoder(macro)

        # 4. 三频融合
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)  # (B, 60, 3D)
        fused = self.scale_fusion(stacked)  # (B, 60, D)

        # 5. 交叉注意力：个股 fused → Q, 市场状态 → K/V
        attn_out, _ = self.cross_attn(
            query=fused,
            key=market_state,
            value=market_state,
        )
        fused = self.cross_norm(fused + attn_out)  # 残差连接 (B, 60, D)

        # 6. 时序编码 → 取最后时间步
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 市场风格编码器：从 4 指数宏观数据中提取市场状态序列
# 方案1：MultiHeadAttention 关系学习
# 方案2：Softmax 关系矩阵
# ============================================================

class _MarketStyleEncoderMHA(nn.Module):
    """
    方案1：精简版 MultiHeadAttention 市场风格编码器。

    输入:  macro (B, 60, 56) — 4指数 × 14维
    流程:
      1. Linear(14→32): 每个指数编码为 32 维 token
      2. MHA(4 heads, 32d): 4 个 token 自注意力
      3. mean: 全局平均池化 → (B, 60, 32)
      4. Linear(32→hidden_dim): 投影到交叉注意力维度
    输出:  market_state (B, 60, hidden_dim)
    """
    def __init__(self, macro_dim=56, hidden_dim=128, num_heads=4, dropout_rate=0.3):
        super().__init__()
        self.num_indices = 4
        index_dim = macro_dim // 4  # 14
        self.index_dim = index_dim
        self.token_dim = 32

        # 1. 线性编码: 14 → 32
        self.index_linear = nn.Linear(index_dim, self.token_dim)

        # 2. MHA: 4 token 自注意力（32维）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.token_dim)

        # 3. 投影到 hidden_dim
        self.out_proj = nn.Linear(self.token_dim, hidden_dim)

    def forward(self, macro):
        """
        macro: (B, 60, 56)
        → market_state: (B, 60, hidden_dim)
        """
        B, T, _ = macro.shape

        # 1. 拆分 4 指数并编码: (B, T, 4, 14) → (B, T, 4, 32)
        idx_tokens = macro.view(B, T, 4, self.index_dim)
        idx_tokens = self.index_linear(idx_tokens)

        # 2. MHA: 合并 B,T 维度，4 token 自注意力
        idx_tokens_flat = idx_tokens.reshape(B * T, 4, self.token_dim)
        attn_out, _ = self.self_attn(idx_tokens_flat, idx_tokens_flat, idx_tokens_flat)
        attn_out = self.attn_norm(idx_tokens_flat + attn_out)

        # 3. 全局平均池化: (B*T, 4, 32) → (B*T, 32)
        market_day = attn_out.mean(dim=1)

        # 4. 投影到 hidden_dim: (B*T, 32) → (B*T, hidden_dim)
        market_state = self.out_proj(market_day)
        market_state = market_state.reshape(B, T, -1)  # (B, 60, hidden_dim)

        return market_state


class _MarketStyleEncoderSoftmax(nn.Module):
    """
    方案2：精简版 Softmax 关系矩阵市场风格编码器。

    输入:  macro (B, 60, 56) — 4指数 × 14维
    流程:
      1. MLP(14→32): 每个指数编码为 32 维 token
      2. Softmax 关系矩阵: QK^T/√d → (4,4) → 加权聚合 → mean → (B,T,32)
      3. Linear(32→hidden_dim): 投影到交叉注意力维度
    输出:  market_state (B, 60, hidden_dim)
    """
    def __init__(self, macro_dim=56, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.num_indices = 4
        index_dim = macro_dim // 4  # 14
        self.index_dim = index_dim
        self.token_dim = 32

        # 1. MLP 编码: 14 → 32
        self.index_mlp = nn.Sequential(
            nn.Linear(index_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # QKV 投影（32维空间内）
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)

        # 2. 投影到 hidden_dim
        self.out_proj = nn.Linear(self.token_dim, hidden_dim)

    def forward(self, macro):
        """
        macro: (B, 60, 56)
        → market_state: (B, 60, hidden_dim)
        """
        B, T, _ = macro.shape

        # 1. 拆分并编码: (B, T, 4, 14) → (B, T, 4, 32)
        idx_tokens = macro.view(B, T, 4, self.index_dim)
        idx_tokens = self.index_mlp(idx_tokens)

        # 2. 关系矩阵
        Q = self.q_proj(idx_tokens)
        K = self.k_proj(idx_tokens)
        V = self.v_proj(idx_tokens)

        # 关系矩阵: (B, T, 4, 4)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.token_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 关系加权聚合: (B, T, 4, 32)
        context = torch.matmul(attn_weights, V)

        # 池化: (B, T, 32)
        market_day = context.mean(dim=2)

        # 3. 投影到 hidden_dim: (B, T, 32) → (B, T, hidden_dim)
        market_state = self.out_proj(market_day)

        return market_state


# ============================================================
# 实验 D：2 指数 ratio + MHA 市场风格编码 + 交叉注意力融合
# 继承 ScaleCentricMTLRelativeV2Ranking，覆盖 _compute_repr
# 注意：这个模型需要多任务训练框架，forward 返回 5 元组
# model_type: 'scale_centric_mtl_v2_ranking_market_mha'
# ============================================================
class ScaleCentricMTLV2RankingMarketMHA(nn.Module):
    """
    在 ratiov2 基础上，添加方案1（MHA）市场风格编码器。

    本类不继承 ratiov2，而是完整重写以保持自包含。
    结构等同于 ScaleCentricMTLRelativeV2Ranking + 市场风格编码。

    macro: (B, 60, 56) — 4指数 × 14维
      - ratio 计算：取沪深300(28~41)和中证500(42~55)的 pct_chg/volume/amp
      - 市场风格编码：全部 56 维

    融合方式：
      fused (B, 60, D) + market_state (B, 60, D)
      → CrossAttention(Q=fused, K/V=market_state) → 残差 → temporal_encoder

    model_type: 'scale_centric_mtl_v2_ranking_market_mha'
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        super().__init__()

        # 2 指数 ratio（沪深300+中证500）= 6 维
        daily_expanded_dim = 8 + 6

        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder   = mk(daily_expanded_dim)
        self.week_encoder  = mk(8)
        self.intra_encoder = mk(intra_hidden)

        # 三频融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(3 * per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # 时序编码器（融合+注意力后）
        self.temporal_encoder = _make_encoder(
            encoder_type, per_scale_dim, per_scale_dim,
            num_layers, dropout_rate, kernel_size, num_levels)

        # === 市场风格编码器（方案1：MHA）===
        self.market_encoder = _MarketStyleEncoderMHA(
            macro_dim=macro_dim, hidden_dim=per_scale_dim,
            num_heads=4, dropout_rate=dropout_rate)

        # === 交叉注意力：fused→Q, market_state→K/V ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=per_scale_dim, num_heads=4,
            dropout=dropout_rate, batch_first=True)
        self.cross_norm = nn.LayerNorm(per_scale_dim)

        # === 多任务解耦投影 ===
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

        # === 分类头（简化版）===
        self.main_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # === 对比投影头 ===
        self.projection = nn.Sequential(
            nn.Linear(per_scale_dim, per_scale_dim),
            nn.ReLU(),
            nn.Linear(per_scale_dim, per_scale_dim),
        )

        # === 排序头（用于 RankingLoss）===
        self.seq_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, future_days),
        )

        # === 统计量头 ===
        self.stat_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 8),
        )

        # === 趋势方向头 ===
        self.trend_head = nn.Sequential(
            nn.Linear(per_scale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),
        )

    def _compute_ratio_features(self, daily_flat, macro):
        """
        从 56 维 macro4idx 中取沪深300(28~41)和中证500(42~55)做 ratio。
        → (B, 60, 6): [return_rel300, return_rel500, amp_rel300, amp_rel500, vol_rel300, vol_rel500]
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        # 沪深300: 28~41, 中证500: 42~55
        p300 = macro[:, :, 28 + 4]
        v300 = macro[:, :, 28 + 5]
        a300 = macro[:, :, 28 + 8]
        p500 = macro[:, :, 42 + 4]
        v500 = macro[:, :, 42 + 5]
        a500 = macro[:, :, 42 + 8]

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
        parts.extend([
            torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
        ])
        parts.extend([
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ])

        return torch.stack(parts, dim=-1)  # (B, 60, 6)

    def _encode_full(self, x, encoder):
        out, _ = encoder(x)
        return out

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征（沪深300+中证500）
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)  # (B, 60, 15)

        # 2. 三频编码
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        # 3. 市场风格编码
        market_state = self.market_encoder(macro)  # (B, 60, per_scale_dim)

        # 4. 三频融合
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)  # (B, 60, per_scale_dim)

        # 5. 交叉注意力：fused→Q, market_state→K/V
        attn_out, _ = self.cross_attn(
            query=fused, key=market_state, value=market_state)
        fused = self.cross_norm(fused + attn_out)  # 残差

        # 6. 时序编码
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]

    def forward(self, daily, weekly, intraday, macro):
        repr_vec = self._compute_repr(daily, weekly, intraday, macro)

        cls_feat  = self.cls_proj(repr_vec)
        seq_feat  = self.seq_proj(repr_vec)
        stat_feat = self.stat_proj(repr_vec)

        logits      = self.main_head(cls_feat)
        proj        = self.projection(cls_feat)
        aux_rank    = self.seq_head(seq_feat)       # (B, 50) 排序分数
        aux_stat    = self.stat_head(stat_feat)      # (B, 8)
        aux_trend   = self.trend_head(seq_feat)      # (B, 3) 趋势方向 logits

        return logits, proj, aux_rank, aux_stat, aux_trend


# ============================================================
# 实验 E：2 指数 ratio + Softmax 关系矩阵市场风格编码 + 交叉注意力
# model_type: 'scale_centric_mtl_v2_ranking_market_softmax'
# ============================================================
class ScaleCentricMTLV2RankingMarketSoftmax(ScaleCentricMTLV2RankingMarketMHA):
    """
    与 MarketMHA 的唯一区别：市场风格编码器使用 Softmax 关系矩阵方案。

    model_type: 'scale_centric_mtl_v2_ranking_market_softmax'
    """

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # 替换市场风格编码器为 Softmax 方案
        self.market_encoder = _MarketStyleEncoderSoftmax(
            macro_dim=macro_dim, hidden_dim=per_scale_dim,
            dropout_rate=dropout_rate)


# ============================================================
# 实验 F/G：市场风格 + 逐日加法融合 / FiLM 调制
# 基类：提取公共的 _compute_repr 中融合部分的变化
# ============================================================

class _ScaleCentricMarketAddBase(ScaleCentricMTLV2RankingMarketMHA):
    """
    基类：用 market_state 逐日调制 fused（加法或 FiLM），代替交叉注意力。

    子类只需覆盖 _fuse_market(self, fused, market_state) 方法。
    """
    _market_encoder_cls = _MarketStyleEncoderMHA  # 子类可覆盖

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        # 先调父类 __init__，但父类中会创建 cross_attn 和 cross_norm
        # 我们不需要它们，所以调完后再删掉以节省显存
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # 替换市场编码器
        self.market_encoder = self._market_encoder_cls(
            macro_dim=macro_dim, hidden_dim=per_scale_dim,
            dropout_rate=dropout_rate)

        # 删除不需要的交叉注意力模块
        del self.cross_attn
        del self.cross_norm

    def _fuse_market(self, fused, market_state):
        """子类覆盖此方法实现不同的融合策略。"""
        raise NotImplementedError

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征（沪深300+中证500）
        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        day_seq = torch.cat([daily_flat, ratio_feats], dim=-1)

        # 2. 三频编码
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        # 3. 市场风格编码
        market_state = self.market_encoder(macro)  # (B, 60, per_scale_dim)

        # 4. 三频融合
        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)

        # 5. 融合市场风格
        fused = self._fuse_market(fused, market_state)

        # 6. 时序编码
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ---- 实验 F1: MHA 市场编码 + 加法融合 ----
class ScaleCentricMarketAddMHA(_ScaleCentricMarketAddBase):
    """
    加法融合：fused = fused + market_state

    model_type: 'scale_centric_market_add_mha'
    """
    _market_encoder_cls = _MarketStyleEncoderMHA

    def _fuse_market(self, fused, market_state):
        return fused + market_state


# ---- 实验 F2: Softmax 市场编码 + 加法融合 ----
class ScaleCentricMarketAddSoftmax(_ScaleCentricMarketAddBase):
    """
    加法融合：fused = fused + market_state

    model_type: 'scale_centric_market_add_softmax'
    """
    _market_encoder_cls = _MarketStyleEncoderSoftmax

    def _fuse_market(self, fused, market_state):
        return fused + market_state


# ---- 实验 G1: MHA 市场编码 + FiLM 调制 ----
class ScaleCentricMarketFiLMMHA(_ScaleCentricMarketAddBase):
    """
    FiLM 调制：fused = fused * (1 + scale) + shift

    market_state → Linear(128→256) → split → scale(128), shift(128)

    model_type: 'scale_centric_market_film_mha'
    """
    _market_encoder_cls = _MarketStyleEncoderMHA

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # FiLM 生成器：market_state (B, 60, 128) → scale, shift (B, 60, 128)
        self.film_generator = nn.Linear(per_scale_dim, 2 * per_scale_dim)

    def _fuse_market(self, fused, market_state):
        film = self.film_generator(market_state)  # (B, 60, 256)
        scale, shift = film.chunk(2, dim=-1)       # 各 (B, 60, 128)
        return fused * (1 + scale) + shift


# ---- 实验 G2: Softmax 市场编码 + FiLM 调制 ----
class ScaleCentricMarketFiLMSoftmax(ScaleCentricMarketFiLMMHA):
    """
    FiLM 调制 + Softmax 市场编码。

    model_type: 'scale_centric_market_film_softmax'
    """
    _market_encoder_cls = _MarketStyleEncoderSoftmax


# ============================================================
# 实验 H: 2 指数 ratio + 跨指数关系特征
# 在 ratiov2 的基础上，额外从 4 指数中构造指数间相对特征
# 使用 ScaleCentricMTLV2RankingMarketMHA 的框架，但不做市场编码
# 而是把跨指数关系特征拼入 day_seq
# model_type: 'scale_centric_mtl_v2_ranking_cross_idx'
# ============================================================
class ScaleCentricMTLV2RankingCrossIdx(ScaleCentricMTLRelativeV2Ranking):
    """
    在 ratiov2 基础上，增加指数间关系特征。

    从 56 维 macro4idx 中取每指数的 5 列：
      pct_chg(4), volume(5), amplitude(8), up_ratio(9), down_ratio(10)

    4 指数代号：全指(F=0~13), 1000(A=14~27), 300(B=28~41), 500(C=42~55)

    构造指数间比率（6 组指数对 × 5 种特征 = 30 维）：
      收益比：F/B, A/B, C/B, F/C, A/C, F/A
      成交量比：同上 6 个
      风格波动比：同上 6 个
      上涨比例比：同上 6 个
      下跌比例比：同上 6 个

    daily_expanded_dim = 8 + 6(个股ratio) + 30(跨指数ratio) = 44
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        # 用 _override_rel_dim 通知父类 ScaleCentricRatio
        self._override_rel_dim = 36  # 6 个股 ratio + 30 跨指数 ratio
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

    def _compute_ratio_features(self, daily_flat, macro):
        """
        daily_flat: (B, 60, 8)
        macro: (B, 60, 56)
        → (B, 60, 36): [个股ratio(6), 跨指数ratio(30)]
        """
        # ---- Part 1: 个股 vs 沪深300/中证500 ratio（6维，同原始ratiov2） ----
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]

        p300 = macro[:, :, 28 + 4]
        v300 = macro[:, :, 28 + 5]
        a300 = macro[:, :, 28 + 8]
        p500 = macro[:, :, 42 + 4]
        v500 = macro[:, :, 42 + 5]
        a500 = macro[:, :, 42 + 8]

        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS), 0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS), 0.01, 100.0)

        stock_ratio = [
            return_ratio300, return_ratio500,
            torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
        ]

        # ---- Part 2: 跨指数关系特征（30维） ----
        # 4 指数的 5 个特征
        idx_feats = {}
        idx_names = ['full', '1000', '300', '500']
        idx_offsets = [0, 14, 28, 42]
        for name, off in zip(idx_names, idx_offsets):
            idx_feats[name] = {
                'pct_chg': macro[:, :, off + 4],      # 收益
                'volume':  macro[:, :, off + 5],       # 成交量
                'amp':     macro[:, :, off + 8],       # 振幅（风格波动）
                'up_ratio':  macro[:, :, off + 9],     # 上涨比例
                'down_ratio': macro[:, :, off + 10],   # 下跌比例
            }

        # 指数对: (分子, 分母, 名称)
        idx_pairs = [
            ('full', '300'), ('1000', '300'), ('500', '300'),
            ('full', '500'), ('1000', '500'),
            ('full', '1000'),
        ]

        cross_ratio = []
        for num_name, den_name in idx_pairs:
            num = idx_feats[num_name]
            den = idx_feats[den_name]
            # 5 种特征的比率（均用有符号比例）
            for key in ['pct_chg', 'volume', 'amp', 'up_ratio', 'down_ratio']:
                nv = num[key]
                dv = den[key]
                if key == 'pct_chg':
                    # 收益用有符号比例
                    sign = torch.sign(nv * dv)
                    ratio = sign * torch.clamp(
                        (nv.abs() + self._EPS) / (dv.abs() + self._EPS), 0.01, 100.0)
                else:
                    # 其余用纯正数比例
                    ratio = torch.clamp(nv / (dv + self._EPS), 0.01, 100.0)
                cross_ratio.append(ratio)

        return torch.stack(stock_ratio + cross_ratio, dim=-1)  # (B, 60, 36)


# ============================================================
# 实验 C-方案1：4 指数 ratio + 市场状态投影 → day_seq 层面拼接（方案1）
# 继承 ScaleCentricRatio4Idx，在市场编码之前就将市场状态投影拼入 day_seq，
# 让 TCN 天然建模市场状态与个股特征的时序交互（类似 ratio 的处理方式）
# model_type: 'scale_centric_ratio_4idx_market_proj'
# ============================================================
class ScaleCentricRatio4IdxMarketProj(ScaleCentricRatio4Idx):
    """
    实验 C-方案1：市场状态因子投影后拼入 day_seq。

    在 ScaleCentricRatio4Idx 基础上：
      1. 用 _MarketStateFactorEncoder 从 macro(56维) 提取 6 维经济学因子，
         再编码为 16 维市场特征
      2. day_seq = stock_feat(64) + ratio_feat(32) + market_feat(16) = 112

    关键设计：市场特征在 day_encoder(TCN) 之前就与 stock/ratio 拼接，
    让 TCN 在每个时间步都能看到市场状态，学习"市场→个股"的联合时序演化。

    model_type: 'scale_centric_ratio_4idx_market_proj'
    """
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

        # 市场因子编码器：56维 → 6维经济学因子 → 16 维（一步到位）
        self.market_encoder = _MarketStateFactorEncoder(
            macro_dim=macro_dim, hidden_dim=16,
            dropout_rate=dropout_rate)

        # day_encoder：输入 = 64(stock) + 32(ratio) + 16(market) = 112
        mk = lambda sz: _make_encoder(encoder_type, sz, per_scale_dim,
                                      num_layers, dropout_rate,
                                      kernel_size, num_levels)
        self.day_encoder = mk(112)

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        # 1. ratio 特征
        ratio_feats = self._compute_ratio_features(daily_flat, macro)

        # 2. 市场因子编码（56→6 经济学因子 → 16）
        market_feat = self.market_encoder(macro)  # (B, 60, 16)

        # 3. 线性投影分组 + 市场特征拼接
        stock_feat  = self.stock_proj(daily_flat)   # (B, 60, 64)
        ratio_feat  = self.ratio_proj(ratio_feats)  # (B, 60, 32)
        day_seq = torch.cat([stock_feat, ratio_feat, market_feat], dim=-1)  # (B, 60, 112)

        # 4. 三频编码
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


# ============================================================
# 实验 C-方案2：市场状态门控 ratio（所有 4 种市场编码器）
#
# 核心思路：market 不直接拼入日频，而是生成 gate 信号调节 ratio 的强度。
# 这样 market 只影响"个股 vs 市场"的相对关系权重，不污染个股编码。
#
# 架构：
#   stock(8) → stock_proj(64) ──┐
#                                ├── concat(96) → day_encoder(TCN) → ...
#   ratio(12) → ratio_proj(32) ──┘
#                         ↑
#                  × gate(32)  ← element-wise
#                         ↑
#             market_encoder → Linear(8) → Sigmoid → expand(32)
# ============================================================


# ============================================================
# 实验 I: 2 指数 ratio + 指数关系矩阵 → 市场状态 → 交叉注意力融合
# 继承 ScaleCentricMTLV2RankingMarketMHA，替换市场编码器为关系矩阵编码器
# model_type: 'scale_centric_mtl_v2_ranking_relation_matrix'
# ============================================================
class _RelationMatrixEncoder(nn.Module):
    """
    关系矩阵编码器。

    输入: macro (B, 60, 56)
    流程:
      1. 取每指数 4 特征 [pct_chg, volume, amp, turn] → (B,T,4,4)
      2. 对每个特征构建 4×4 比值矩阵 → (B,T,4,4,4) → 展平 → (B,T,4,16)
      3. MLP → (B,T,4,32)
      4. MHA(4 token) → mean → (B,T,32)
      5. Linear(32→128) → (B,T,128)
    输出: market_state (B, 60, hidden_dim)
    """
    _EPS = 1e-8

    def __init__(self, macro_dim=56, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.token_dim = 32

        # MLP: (4,16) → (4,32)
        self.relation_mlp = nn.Sequential(
            nn.Linear(16, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # MHA: 4 token 自注意力（32维）
        self.token_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=4,
            dropout=dropout_rate, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.token_dim)

        # 投影到 hidden_dim
        self.out_proj = nn.Linear(self.token_dim, hidden_dim)

    def forward(self, macro):
        B, T, _ = macro.shape

        # 1. 取 4 指数的 4 个特征：pct_chg(4), volume(5), amp(8), turn(7)
        idx_data = macro.reshape(B, T, 4, 14)
        idx_feats = idx_data[:, :, :, [4, 5, 8, 7]]  # (B, T, 4, 4)

        # 2. 构建 4 个 4×4 关系矩阵（每个特征一个）
        matrices = []
        for f in range(4):
            f_vals = idx_feats[:, :, :, f:f+1]  # (B,T,4,1)
            sign = torch.sign(f_vals * f_vals.transpose(-2, -1))
            ratio_mat = sign * torch.clamp(
                (f_vals.abs() + self._EPS) / (f_vals.transpose(-2, -1).abs() + self._EPS),
                0.01, 100.0)
            matrices.append(ratio_mat)

        # (B, T, 4, 4, 4) → (B, T, 4, 16)
        rel_matrix = torch.stack(matrices, dim=-1)
        rel_flat = rel_matrix.reshape(B, T, 4, 16)

        # 3. MLP 抽象: (B, T, 4, 16) → (B, T, 4, 32)
        token_feats = self.relation_mlp(rel_flat)

        # 4. 注意力融合: (B*T, 4, 32) → mean → (B*T, 32)
        token_flat = token_feats.reshape(B * T, 4, self.token_dim)
        attn_out, _ = self.token_attn(token_flat, token_flat, token_flat)
        attn_out = self.attn_norm(token_flat + attn_out)
        market_day = attn_out.mean(dim=1)  # (B*T, 32)

        # 5. 投影到 hidden_dim
        market_state = self.out_proj(market_day)
        market_state = market_state.reshape(B, T, -1)  # (B, 60, hidden_dim)

        return market_state


class ScaleCentricMTLV2RankingRelationMatrix(ScaleCentricMTLV2RankingMarketMHA):
    """
    在 ratiov2 + cross_attn 基础上，用关系矩阵编码器替换 MHA 市场编码器。

    关系矩阵编码器将 4 指数 × 4 特征构建为 4×4 比值矩阵，
    通过 MLP + MHA 融合为市场状态。

    其余结构与 MarketMHA 完全相同（cross_attn 融合）。

    model_type: 'scale_centric_mtl_v2_ranking_relation_matrix'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # 替换市场编码器为关系矩阵编码器
        self.market_encoder = _RelationMatrixEncoder(
            macro_dim=macro_dim, hidden_dim=per_scale_dim,
            dropout_rate=dropout_rate)


# ============================================================
# 实验 J: 2 指数 ratio + 换手率 ratio（沪深300+中证500）
# 在 ratiov2 的 6 维 ratio 基础上，增加 turn(换手率) ratio 2 维
# daily_expanded_dim = 8 + 8 = 16
# model_type: 'scale_centric_mtl_v2_ranking_turn'
# ============================================================
# ============================================================
# 实验 K: 市场状态因子编码器（价格/波动/风险偏好/市场扩散）
# ============================================================
class _MarketStateFactorEncoder(nn.Module):
    """
    市场状态因子编码器：从 4 指数宏观数据中提取经济学含义的市场状态因子。

    输入: macro (B, 60, 56) — 4指数 × 14维
    输出: market_state (B, 60, hidden_dim)

    因子设计（共 6 维原始因子）：
      1. 价格 (Price)      — 四指数平均收益: mean(pct_chg)     → 1 维
      2. 波动 (Volatility)  — 四指数平均振幅: mean(amplitude)  → 1 维
      3. 风险偏好 (Risk Pref) — 小盘/大盘收益比:
          1000/300, 500/300, 1000/全指 (有符号比例)           → 3 维
      4. 市场扩散 (Breadth)  — 四指数平均腾落指标:
          mean(rel_advance_decline)                            → 1 维

    编码流程: Raw(6) → Linear(6→32) → ReLU → Linear(32→hidden_dim)
    """
    _EPS = 1e-8

    def __init__(self, macro_dim=56, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.factor_dim = 6
        self.inner_dim = 32

        self.encoder = nn.Sequential(
            nn.Linear(self.factor_dim, self.inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.inner_dim, hidden_dim),
        )

    def forward(self, macro):
        """
        macro: (B, 60, 56)
        → market_state: (B, 60, hidden_dim)
        """
        B, T, _ = macro.shape

        # 1. 价格: 4 指数 pct_chg 均值
        #    全指(4), 1000(18), 300(32), 500(46)
        price_mean = macro[:, :, [4, 18, 32, 46]].mean(dim=-1, keepdim=True)  # (B, T, 1)

        # 2. 波动: 4 指数 amplitude 均值
        #    全指(8), 1000(22), 300(36), 500(50)
        vol_mean = macro[:, :, [8, 22, 36, 50]].mean(dim=-1, keepdim=True)   # (B, T, 1)

        # 3. 风险偏好: 小盘/大盘收益比（有符号比例）
        p1000 = macro[:, :, 18]   # 中证1000 pct_chg
        p300  = macro[:, :, 32]   # 沪深300 pct_chg
        p500  = macro[:, :, 46]   # 中证500 pct_chg
        pfull = macro[:, :, 4]    # 中证全指 pct_chg

        def signed_ratio(a, b):
            sign = torch.sign(a * b)
            return sign * torch.clamp(
                (a.abs() + self._EPS) / (b.abs() + self._EPS), 0.01, 100.0)

        risk_1000_300  = signed_ratio(p1000, p300).unsqueeze(-1)   # (B, T, 1)
        risk_500_300   = signed_ratio(p500,  p300).unsqueeze(-1)   # (B, T, 1)
        risk_1000_full = signed_ratio(p1000, pfull).unsqueeze(-1)  # (B, T, 1)

        # 4. 市场扩散: 4 指数 rel_advance_decline 均值
        #    全指(13), 1000(27), 300(41), 500(55)
        breadth_mean = macro[:, :, [13, 27, 41, 55]].mean(dim=-1, keepdim=True)  # (B, T, 1)

        # 拼接原始因子
        raw_factors = torch.cat([
            price_mean, vol_mean,
            risk_1000_300, risk_500_300, risk_1000_full,
            breadth_mean,
        ], dim=-1)  # (B, T, 6)

        # MLP 编码
        market_state = self.encoder(raw_factors)  # (B, T, hidden_dim)

        return market_state


class ScaleCentricMTLV2RankingMarketFactor(ScaleCentricMTLV2RankingMarketMHA):
    """
    在 ratiov2 + cross_attn 基础上，用市场状态因子编码器替换 MHA 市场编码器。

    市场状态因子编码器提取 6 维经济学含义的因子（价格/波动/风险偏好/市场扩散），
    通过 MLP 编码为市场状态序列，经交叉注意力与个股特征融合。

    其余结构与 MarketMHA 完全相同。

    model_type: 'scale_centric_mtl_v2_ranking_market_factor'
    """
    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

        # 替换市场编码器为因子编码器
        self.market_encoder = _MarketStateFactorEncoder(
            macro_dim=macro_dim, hidden_dim=per_scale_dim,
            dropout_rate=dropout_rate)


# ============================================================
# 实验 J: 2 指数 ratio + 换手率 ratio（沪深300+中证500）
# ============================================================
class ScaleCentricMTLV2RankingTurn(ScaleCentricMTLRelativeV2Ranking):
    """
    在 ratiov2 基础上，增加换手率 ratio。

    从 macro4idx 中取沪深300(28~41)和中证500(42~55)的 turn(7)，
    与股票的 turn(7) 做有符号比例。

    ratio 维度: [pct_chg_300, pct_chg_500, amp_300, amp_500,
                 vol_300, vol_500, turn_300, turn_500] = 8 维

    daily_expanded_dim = 8 + 8 = 16
    """
    _EPS = 1e-8

    def __init__(self, per_scale_dim=128, hidden_dim=128, num_classes=3,
                 dropout_rate=0.3, num_layers=1, encoder_type='tcn',
                 kernel_size=3, num_levels=4, intra_hidden=32,
                 future_days=50, macro_dim=56):
        self._override_rel_dim = 8
        super().__init__(
            per_scale_dim=per_scale_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout_rate=dropout_rate,
            num_layers=num_layers, encoder_type=encoder_type,
            kernel_size=kernel_size, num_levels=num_levels,
            intra_hidden=intra_hidden, future_days=future_days,
            macro_dim=macro_dim)

    def _compute_ratio_features(self, daily_flat, macro):
        """
        daily_flat: (B, 60, 8)
        macro: (B, 60, 56)
        → (B, 60, 8): [pct_chg_300, pct_chg_500, amp_300, amp_500,
                        vol_300, vol_500, turn_300, turn_500]
        """
        stock_pct_chg = daily_flat[:, :, 4]
        stock_volume  = daily_flat[:, :, 5]
        stock_amp     = daily_flat[:, :, 7]
        stock_turn    = daily_flat[:, :, 7]  # 换手率

        # 沪深300: 28~41, 中证500: 42~55
        p300 = macro[:, :, 28 + 4]
        v300 = macro[:, :, 28 + 5]
        a300 = macro[:, :, 28 + 8]
        t300 = macro[:, :, 28 + 7]  # turn
        p500 = macro[:, :, 42 + 4]
        v500 = macro[:, :, 42 + 5]
        a500 = macro[:, :, 42 + 8]
        t500 = macro[:, :, 42 + 7]  # turn

        # pct_chg 有符号比例
        sign300 = torch.sign(stock_pct_chg * p300)
        sign500 = torch.sign(stock_pct_chg * p500)
        return_ratio300 = sign300 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p300.abs() + self._EPS), 0.01, 100.0)
        return_ratio500 = sign500 * torch.clamp(
            (stock_pct_chg.abs() + self._EPS) / (p500.abs() + self._EPS), 0.01, 100.0)

        # turn 也是正数，用纯正数比例
        turn_ratio300 = torch.clamp(stock_turn / (t300 + self._EPS), 0.01, 100.0)
        turn_ratio500 = torch.clamp(stock_turn / (t500 + self._EPS), 0.01, 100.0)

        return torch.stack([
            return_ratio300, return_ratio500,
            torch.clamp(stock_amp / (a300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_amp / (a500 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v300 + self._EPS), 0.01, 100.0),
            torch.clamp(stock_volume / (v500 + self._EPS), 0.01, 100.0),
            turn_ratio300, turn_ratio500,
        ], dim=-1)  # (B, 60, 8)


# ============================================================
# 实验 C-方案2：市场状态门控 ratio（所有 4 种市场编码器）
# ============================================================

class _ScaleCentricRatioGateBase(ScaleCentricRatio4Idx):
    """
    基类：市场状态门控/FiLM 调节 ratio。

    子类设置 _market_encoder_key 和 _use_film。
    - gate:   ratio * sigmoid(market)          → [0, 1]，仅抑制
    - FiLM:   ratio * (1 + scale) + shift       → 可放大/抑制 + 偏置
    """
    _market_encoder_key = None
    _use_film = False

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

        if self._use_film:
            # FiLM: market_dim → 2 × 32 = scale(32) + shift(32)
            self.film_net = nn.Linear(market_dim, 2 * 32)
        else:
            # Gate: market_dim → 8 → Sigmoid → repeat(4) → 32
            self.gate_net = nn.Sequential(
                nn.Linear(market_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.Sigmoid(),
            )

    def _modulate_ratio(self, ratio_feat, market_state):
        """根据 _use_film 选择 gate 或 FiLM 调制 ratio。"""
        if self._use_film:
            film = self.film_net(market_state)        # (B, 60, 64)
            scale, shift = film.chunk(2, dim=-1)      # 各 (B, 60, 32)
            return ratio_feat * (1 + scale) + shift
        else:
            gate_8 = self.gate_net(market_state)      # (B, 60, 8)
            gate_32 = gate_8.repeat_interleave(4, dim=-1)  # (B, 60, 32)
            return ratio_feat * gate_32

    def _compute_repr(self, daily, weekly, intraday, macro):
        B = daily.size(0)
        daily_flat = daily.reshape(B, 60, 8)

        ratio_feats = self._compute_ratio_features(daily_flat, macro)
        ratio_feat = self.ratio_proj(ratio_feats)  # (B, 60, 32)

        market_state = self.market_encoder(macro)
        ratio_feat = self._modulate_ratio(ratio_feat, market_state)

        stock_feat = self.stock_proj(daily_flat)
        day_seq = torch.cat([stock_feat, ratio_feat], dim=-1)

        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)
        intra_daily = self.intra_compressor(intraday)

        day_repr   = self._encode_full(day_seq,      self.day_encoder)
        week_repr  = self._encode_full(week_seq,     self.week_encoder)
        intra_repr = self._encode_full(intra_daily,  self.intra_encoder)

        stacked = torch.cat([week_repr, day_repr, intra_repr], dim=-1)
        fused = self.scale_fusion(stacked)
        fused_out, _ = self.temporal_encoder(fused)
        return fused_out[:, -1, :]


class ScaleCentricRatioGateFactor(_ScaleCentricRatioGateBase):
    """市场因子门控 ratio。model_type: 'scale_centric_ratio_gate_factor'"""
    _market_encoder_key = 'factor'


class ScaleCentricRatioGateMHA(_ScaleCentricRatioGateBase):
    """MHA 市场编码门控 ratio。model_type: 'scale_centric_ratio_gate_mha'"""
    _market_encoder_key = 'mha'


class ScaleCentricRatioGateSoftmax(_ScaleCentricRatioGateBase):
    """Softmax 关系矩阵门控 ratio。model_type: 'scale_centric_ratio_gate_softmax'"""
    _market_encoder_key = 'softmax'


class ScaleCentricRatioGateRelationMatrix(_ScaleCentricRatioGateBase):
    """关系矩阵门控 ratio。model_type: 'scale_centric_ratio_gate_relation_matrix'"""
    _market_encoder_key = 'relation_matrix'


# ---- 方案2-FiLM：FiLM 调制 ratio（4 种编码器）----

class ScaleCentricRatioFiLMFactor(_ScaleCentricRatioGateBase):
    """市场因子 FiLM 调制 ratio。model_type: 'scale_centric_ratio_film_factor'"""
    _market_encoder_key = 'factor'
    _use_film = True


class ScaleCentricRatioFiLMMHA(_ScaleCentricRatioGateBase):
    """MHA 市场编码 FiLM 调制 ratio。model_type: 'scale_centric_ratio_film_mha'"""
    _market_encoder_key = 'mha'
    _use_film = True


class ScaleCentricRatioFiLMSoftmax(_ScaleCentricRatioGateBase):
    """Softmax 关系矩阵 FiLM 调制 ratio。model_type: 'scale_centric_ratio_film_softmax'"""
    _market_encoder_key = 'softmax'
    _use_film = True


class ScaleCentricRatioFiLMRelationMatrix(_ScaleCentricRatioGateBase):
    """关系矩阵 FiLM 调制 ratio。model_type: 'scale_centric_ratio_film_relation_matrix'"""
    _market_encoder_key = 'relation_matrix'
    _use_film = True


