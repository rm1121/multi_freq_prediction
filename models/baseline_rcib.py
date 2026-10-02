"""
Baseline: RevIN-CNN-iTransformer-BiLSTM（模块化金融时序预测框架，
Huang & Yang, Expert Systems with Applications 2026）。model_type: 'baseline_rcib'

官方仓库: https://github.com/DyeDye-diyung/Framework_for_Financial_Time-Series_Forecasting
原论文框架: RevIN -> CNN -> 可插拔 Transformer 变体 -> BiLSTM（最优组合为
RevIN_CNN_iTransformer_BiLSTM），原任务为单变量价格回归，此处替换为统一分类头。

适配方式（与其他对比 baseline 一致）：
  三尺度输入 -> align_scales 拼接 (B, 60, 8+8+intra_hidden)
  -> RevIN（逐样本实例归一化；分类任务无需 denorm，仅保留 normalize 半边，
     用于缓解分布漂移/体制切换）
  -> CNN（2 层 Conv1d，局部短期微观模式）
  -> iTransformer（变量维注意力，可插拔核心，直接复用仓库现有实现；
     pred_len=seq_len=60 重建序列以保留时间维度）
  -> BiLSTM（双向时序动态，取末步前向+后向隐状态）
  -> 统一分类头 -> (logits, proj)
"""

import torch
import torch.nn as nn

from .tslib_models.wrappers import (IntradayCompressor, align_scales,
                                     ClassificationHead, ProjectionHead,
                                     _Configs)


class RevIN(nn.Module):
    """可逆实例归一化（RevIN）。分类任务无需还原量纲，
    仅保留 normalize 半边（这正是 RevIN 缓解非平稳性的核心部分）。
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """x: (B, L, C) -> (B, L, C)"""
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class LocalCNNEncoder(nn.Module):
    """CNN 模块：提取局部短期微观模式。"""

    def __init__(self, in_channels, out_channels=64, num_layers=2,
                 kernel_size=3, dropout_rate=0.5):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(c_in, out_channels, kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ]
            c_in = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, L, C) -> (B, L, out_channels)"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class RevINCNNiTransformerBiLSTMClassifier(nn.Module):
    """RevIN-CNN-iTransformer-BiLSTM baseline。model_type: 'baseline_rcib'"""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.5, intra_hidden=32, d_model=64, depth=2,
                 cnn_channels=64, bilstm_hidden=128, **kwargs):
        super().__init__()
        # 美股数据管线的日/周/分时输入均为 8 维；align_scales 同样按 8 维对齐。
        enc_in = 8 + 8 + intra_hidden
        self.intra_compressor = IntradayCompressor(8, intra_hidden, dropout_rate)

        # 模块 1: RevIN（实例归一化）
        self.revin = RevIN(enc_in)
        # 模块 2: CNN（局部模式）
        self.cnn = LocalCNNEncoder(enc_in, cnn_channels,
                                   num_layers=2, dropout_rate=dropout_rate)
        # 模块 3: iTransformer（可插拔 Transformer 变体，变量维注意力）
        configs = _Configs(
            task_name='long_term_forecast', seq_len=60, pred_len=60,
            label_len=12, enc_in=cnn_channels, c_out=cnn_channels,
            d_model=d_model, n_heads=4, d_ff=hidden_dim,
            e_layers=depth, d_layers=1, dropout=dropout_rate)
        from .tslib_models.iTransformer import Model as TSLiTransformer
        self.itransformer = TSLiTransformer(configs)
        # 模块 4: BiLSTM（双向时序动态）
        self.bilstm = nn.LSTM(cnn_channels, bilstm_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)

        repr_dim = 2 * bilstm_hidden
        self.head = ClassificationHead(repr_dim, hidden_dim,
                                       num_classes, dropout_rate)
        self.projection = ProjectionHead(repr_dim, dropout_rate)

    def _encode(self, x):
        """x: (B, 60, enc_in) -> (B, 2*bilstm_hidden)"""
        x = self.revin(x)                            # (B, 60, enc_in)
        x = self.cnn(x)                              # (B, 60, cnn_channels)
        x = self.itransformer(x, None, None, None)   # (B, 60, cnn_channels)
        out, _ = self.bilstm(x)                      # (B, 60, 2*bilstm_hidden)
        return self.dropout(out[:, -1, :])           # 末步前向+后向隐状态拼接

    def forward(self, daily, weekly, intraday):
        x = align_scales(daily, weekly, intraday, self.intra_compressor)
        repr_vec = self._encode(x)
        logits = self.head(repr_vec)
        proj = self.projection(repr_vec)
        return logits, proj
