"""
MASTER（Market-Guided Stock Transformer, AAAI/KDD 2024）对比实验。

不直接跑官方版本，而是：
  - 输入：三尺度对齐后的日频股票特征 + 市场信息（macro）
  - 使用 MASTER 的核心 encoder 结构：
      1. 市场引导门控（Market-Guided Gate）：用市场信息生成特征权重，
         动态调整股票特征的重要性
      2. 时间注意力（TAttention）：捕捉个股时序模式
      3. 空间注意力（SAttention）：建模特征/变量间的相关性
      4. 时序池化（TemporalAttention）：聚合时间步
  - 输出：统一分类头（与最终模型一致：BN -> MLP -> Dropout -> 分类）

model_type: 'baseline_master'
"""

import math
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm


# ============================================================
# MASTER 核心组件（来自官方实现）
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    """空间注意力（inter-feature aggregation），特征维注意力。"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model), nn.ReLU(), Dropout(p=dropout),
            Linear(d_model, d_model), Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output


class TAttention(nn.Module):
    """时间注意力（intra-stock aggregation），时间维注意力。"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model), nn.ReLU(), Dropout(p=dropout),
            Linear(d_model, d_model), Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output


class Gate(nn.Module):
    """市场引导门控：用市场信息生成特征权重。"""

    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    """时序池化：用最后一个时间步作为 query，加权聚合时间步。"""

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, D]
        return output


# ============================================================
# MASTER 分类封装
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


class ClassificationHead(nn.Module):
    """统一分类头，与最终模型一致。"""

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


class MASTERClassifier(nn.Module):
    """
    MASTER 市场引导 encoder + 统一分类头。

    输入：daily, weekly, intraday（三尺度股票信息）+ macro（市场信息）
    流程：
      1. 三尺度对齐拼接 -> src (B, 60, enc_in)
      2. 市场引导门控：macro 最后一天生成特征权重，调制 src
      3. TAttention（时序模式）
      4. SAttention（特征相关性）
      5. TemporalAttention（时序池化）-> (B, d_model)
      6. 统一分类头

    model_type: 'baseline_master'
    """

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=1, num_classes=3,
                 dropout_rate=0.3, intra_hidden=32, d_model=64, depth=2,
                 macro_dim=56, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        enc_in = input_dim + input_dim + intra_hidden
        self.enc_in = enc_in
        self.macro_dim = macro_dim

        self.intra_compressor = IntradayCompressor(
            input_dim=input_dim, hidden_dim=intra_hidden, dropout_rate=dropout_rate)

        # 市场引导门控：market_dim -> enc_in（特征权重）
        self.feature_gate = Gate(macro_dim, enc_in, beta=1.0)

        # MASTER encoder
        t_nhead = 4
        s_nhead = 4
        self.layers = nn.Sequential(
            nn.Linear(enc_in, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=dropout_rate),
            TemporalAttention(d_model=d_model),
        )

        # 统一分类头
        self.head = ClassificationHead(d_model, hidden_dim, num_classes, dropout_rate)
        self.projection = ProjectionHead(d_model, dropout_rate)

    def _align_scales(self, daily, weekly, intraday):
        B = daily.size(0)
        day_seq = daily.reshape(B, 60, self.input_dim)
        week_seq = weekly.squeeze(1).repeat_interleave(5, dim=1)  # (B, 60, input_dim)
        intra_seq = self.intra_compressor(intraday)               # (B, 60, intra_hidden)
        return torch.cat([day_seq, week_seq, intra_seq], dim=-1)  # (B, 60, enc_in)

    def forward(self, daily, weekly, intraday, macro):
        src = self._align_scales(daily, weekly, intraday)  # (B, 60, enc_in)

        # 市场引导门控：用最后一天的市场信息生成特征权重
        gate_input = macro[:, -1, :]  # (B, market_dim)
        gate = self.feature_gate(gate_input).unsqueeze(1)  # (B, 1, enc_in)
        src = src * gate

        feat = self.layers(src)  # (B, d_model)

        logits = self.head(feat)
        proj = self.projection(feat)
        return logits, proj
