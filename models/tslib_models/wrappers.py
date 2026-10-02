"""
对比实验：统一分类头 + 三尺度对齐拼接的时序模型封装。

这些模型作为最终模型 ScaleCentricFusionGateMHA 的对比 baseline：
  - 无市场信息模型（LSTM/GRU/Transformer/Informer/Autoformer/PatchTST/
    TimesNet/iTransformer/TimeMixer/ModernTCN）：
      输入 = 三尺度对齐后拼接的序列，不独立编码。
  - 有市场信息模型（MASTER，单独文件实现）。

统一约定：
  - 输入：model(daily, weekly, intraday) -> (logits, proj)
  - daily: (B, 12, 5, 8) -> reshape (B, 60, 8)
  - weekly: (B, 1, 12, 8) -> repeat_interleave(5) -> (B, 60, 8)
  - intraday: (B, 60, 13, 8) -> IntradayCompressor -> (B, 60, intra_hidden)
  - 三尺度拼接 -> (B, 60, 8+8+intra_hidden)
  - 经过各自的序列编码器提取表示，统一过 ClassificationHead 分类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding


# ============================================================
# 统一分类头（与最终模型一致：BN -> MLP -> Dropout -> 分类）
# ============================================================
class ClassificationHead(nn.Module):
    """统一分类头，与 ScaleCentricFusionGateMHA 的 head 一致。"""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ProjectionHead(nn.Module):
    """对比投影头。"""

    def __init__(self, in_dim, dropout_rate):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, x):
        return self.projection(x)


# ============================================================
# 数据对齐：三尺度 -> 拼接序列 (B, 60, C)
# ============================================================
class IntradayCompressor(nn.Module):
    """分时压缩：13 bar × 8 -> 日级摘要 (B, 60, intra_hidden)。"""

    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B * 60, 13, x.size(-1))
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return out.reshape(B, 60, -1)


def align_scales(daily, weekly, intraday, intra_compressor):
    """将三尺度对齐为拼接序列 (B, 60, 8+8+intra_hidden)。"""
    B = daily.size(0)
    day_seq = daily.reshape(B, 60, 8)
    week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)  # (B, 60, 8)
    intra_seq = intra_compressor(intraday)                    # (B, 60, intra_hidden)
    return torch.cat([day_seq, week_seq, intra_seq], dim=-1)


# ============================================================
# 通用包装基类
# ============================================================
class TSLibWrapper(nn.Module):
    """将任意序列编码器包装为三尺度分类模型。

    子类需实现 self.seq_encoder(x) -> (B, D) 表示向量。
    """

    def __init__(self, hidden_dim=128, num_classes=3, dropout_rate=0.3,
                 intra_hidden=32, **kwargs):
        super().__init__()
        self.intra_compressor = IntradayCompressor(
            input_dim=8, hidden_dim=intra_hidden, dropout_rate=dropout_rate)
        self.head = ClassificationHead(hidden_dim, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(hidden_dim, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self.seq_encoder(x)  # (B, D)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# LSTM / GRU（手写实现）
# ============================================================
class _RecurrentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate,
                 cell='lstm'):
        super().__init__()
        rnn_cls = nn.LSTM if cell == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.rnn(x)  # (B, T, D)
        return self.norm(out[:, -1, :])


class LSTMClassifier(TSLibWrapper):
    """LSTM baseline。model_type: 'baseline_lstm'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        # 先构建 encoder 再调父类（父类只建 compressor + head，无 input_dim）
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        self.seq_encoder = _RecurrentEncoder(enc_in, hidden_dim, num_layers,
                                             dropout_rate, 'lstm')
        self.head = ClassificationHead(hidden_dim, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(hidden_dim, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self.seq_encoder(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class GRUClassifier(TSLibWrapper):
    """GRU baseline。model_type: 'baseline_gru'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        self.seq_encoder = _RecurrentEncoder(enc_in, hidden_dim, num_layers,
                                             dropout_rate, 'gru')
        self.head = ClassificationHead(hidden_dim, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(hidden_dim, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self.seq_encoder(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class Chomp1d(nn.Module):
    """裁剪掉右侧多余 padding，保证因果性。"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class _TCNBlock(nn.Module):
    """TCN 因果膨胀卷积块（与项目 TCNEncoder 一致：残差 + 双卷积 + Chomp）。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = torch.relu(self.bn1(self.chomp1(self.conv1(x))))
        out = self.dropout1(out)
        out = torch.relu(self.bn2(self.chomp2(self.conv2(out))))
        out = self.dropout2(out)
        return torch.relu(out + residual)


class _TCNEncoder(nn.Module):
    """TCN 编码器：因果膨胀卷积，输出最后一个时间步的表示。

    input_dim -> hidden_dim，num_levels 层，每层膨胀系数 2^(i)。
    使用与项目 TCNEncoder 一致的因果 TCNBlock（残差 + Chomp），
    避免对称 padding 导致的信息泄漏。
    """

    def __init__(self, input_dim, hidden_dim, num_levels=4,
                 kernel_size=3, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, hidden_dim, kernel_size,
                                    dilation, dropout_rate))
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)        # (B, F, T)
        out = self.net(x)             # (B, hidden, T)
        return out.mean(dim=2)        # (B, hidden) 时间平均池化


class TCNClassifier(TSLibWrapper):
    """TCN baseline。model_type: 'baseline_tcn'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, num_levels=4,
                 kernel_size=3, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        self.seq_encoder = _TCNEncoder(enc_in, hidden_dim, num_levels=num_levels,
                                       kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.head = ClassificationHead(hidden_dim, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(hidden_dim, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self.seq_encoder(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


# ============================================================
# Transformer 系（TSLib 模型包装）
# ============================================================
class _Configs:
    """TSLib 模型所需的 config 对象。"""

    def __init__(self, task_name='long_term_forecast', seq_len=60, pred_len=1,
                 label_len=12, enc_in=None, dec_in=None, c_out=None,
                 d_model=64, n_heads=4, d_ff=128, e_layers=2, d_layers=1,
                 dropout=0.3, activation='gelu', factor=5, output_attention=False,
                 embed='timeF', freq='h', num_class=3, **kwargs):
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.dec_in = dec_in if dec_in is not None else enc_in
        self.c_out = c_out if c_out is not None else enc_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.output_attention = output_attention
        self.embed = embed
        self.freq = freq
        self.num_class = num_class
        self.distil = kwargs.get('distil', False)
        self.decomposition = kwargs.get('decomposition', False)
        self.individual = kwargs.get('individual', False)
        self.embed_type = kwargs.get('embed_type', 0)
        # 各模型特有参数
        self.moving_avg = kwargs.get('moving_avg', 25)
        self.num_kernels = kwargs.get('num_kernels', 6)
        self.top_k = kwargs.get('top_k', 5)
        self.channel_independence = kwargs.get('channel_independence', 1)
        self.decomp_method = kwargs.get('decomp_method', 'moving_avg')
        self.down_sampling_layers = kwargs.get('down_sampling_layers', 3)
        self.down_sampling_method = kwargs.get('down_sampling_method', 'avg')
        self.down_sampling_window = kwargs.get('down_sampling_window', 2)
        self.use_norm = kwargs.get('use_norm', 1)


def _build_dec_input(x_enc, model):
    """构造 decoder 输入 x_dec（用于预测类任务），x_mark 置 None 跳过时间编码。"""
    B, L, C = x_enc.shape
    label_len = getattr(model, 'label_len', 12)
    pred_len = getattr(model, 'pred_len', 1)
    x_dec = torch.cat([x_enc[:, -label_len:, :],
                       torch.zeros(B, pred_len, C, device=x_enc.device)], dim=1)
    return x_enc, None, x_dec, None


class TransformerClassifier(TSLibWrapper):
    """Transformer baseline。model_type: 'baseline_transformer'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .Transformer import Model as TSLTransformer
        self.tslib = TSLTransformer(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)  # (B, c_out)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class InformerClassifier(TSLibWrapper):
    """Informer baseline。model_type: 'baseline_informer'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate,
                           factor=5)
        from .Informer import Model as TSLInformer
        self.tslib = TSLInformer(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class AutoformerClassifier(TSLibWrapper):
    """Autoformer baseline。model_type: 'baseline_autoformer'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .Autoformer import Model as TSLAutoformer
        self.tslib = TSLAutoformer(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class PatchTSTClassifier(TSLibWrapper):
    """PatchTST baseline。model_type: 'baseline_patchtst'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4,
                 patch_size=5, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .PatchTST import Model as TSLPatchTST
        patch_len = max(1, patch_size)
        self.tslib = TSLPatchTST(configs, patch_len=patch_len, stride=patch_len // 2 if patch_len > 1 else 1)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class TimesNetClassifier(TSLibWrapper):
    """TimesNet baseline。model_type: 'baseline_timesnet'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .TimesNet import Model as TSLTimesNet
        self.tslib = TSLTimesNet(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class iTransformerClassifier(TSLibWrapper):
    """iTransformer baseline。model_type: 'baseline_itransformer'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .iTransformer import Model as TSLiTrans
        self.tslib = TSLiTrans(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class TimeMixerClassifier(TSLibWrapper):
    """TimeMixer baseline。model_type: 'baseline_timemixer'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)
        configs = _Configs(task_name='long_term_forecast', enc_in=enc_in, c_out=enc_in,
                           d_model=d_model, n_heads=4, d_ff=hidden_dim,
                           e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .TimeMixer import Model as TSLTimeMixer
        self.tslib = TSLTimeMixer(configs)
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        x_enc, x_mark_enc, x_dec, x_mark_dec = _build_dec_input(x, self.tslib)
        dec_out = self.tslib.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.tslib.pred_len:, :].mean(dim=1)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj


class ModernTCNClassifier(TSLibWrapper):
    """ModernTCN baseline。model_type: 'baseline_moderntcn'

    使用 ModernTCN 的多尺度深度可重参数化卷积骨干提取序列表示，
    统一过 ClassificationHead 分类。
    """

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=4,
                 kernel_size=3, **kwargs):
        enc_in = 8 + 8 + intra_hidden
        nn.Module.__init__(self)
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)

        from .moderntcn.ModernTCN import ModernTCN as MTNCore

        # seq_len=60，patch_size/patch_stride 需能整除 60
        patch_stride = 5
        patch_size = 5
        seq_len = 60
        # 4 个 stage（dims/num_blocks 长度必须为 4，因为 downsample 层固定 3 个）
        # downsample_ratio=1 避免短序列被过度压缩
        dims = [d_model] * 4
        dw_dims = [d_model] * 4
        num_blocks = [2] * 4
        large_size = [51] * 4
        small_size = [5] * 4
        self.tslib = MTNCore(
            patch_size=patch_size, patch_stride=patch_stride,
            stem_ratio=4, downsample_ratio=1, ffn_ratio=2,
            num_blocks=num_blocks, large_size=large_size,
            small_size=small_size, dims=dims, dw_dims=dw_dims,
            nvars=enc_in, small_kernel_merged=False,
            backbone_dropout=dropout_rate, head_dropout=dropout_rate,
            use_multi_scale=True, revin=True, affine=True,
            subtract_last=False, freq='h', seq_len=seq_len,
            c_in=enc_in, individual=False, target_window=1,
        )
        self.head = ClassificationHead(enc_in, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(enc_in, dropout_rate)

    def _encode(self, x):
        # 直接调用 forward_feature 提取多尺度特征，避开官方 forward 的 channel 混乱。
        # 输入 (B, M, L) = (B, C, L)，特征 (B, M, D, N)，池化为 (B, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        feat = self.tslib.forward_feature(x, te=None)  # (B, M, D, N)
        return feat.mean(dim=(-1, -2))  # (B, C)

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj
