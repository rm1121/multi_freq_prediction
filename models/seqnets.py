import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 多频率融合工具
# ============================================================
def _fuse_frequencies(daily, weekly, intraday):
    """
    将日频/周频/分时三路信号统一到 (B, 60, 90) 的时间粒度。

    daily:    (B, 12, 5, 9)   → (B, 60, 9)
    weekly:   (B, 1, 12, 9)   → (B, 60, 9)
    intraday: (B, 60, 8, 9)   → (B, 60, 72)
    拼接: (B, 60, 9+9+72) → (B, 60, 90)
    """
    B = daily.size(0)

    day_feat = daily.reshape(B, 60, 9)

    week_feat = weekly.squeeze(1)
    week_feat = week_feat.repeat_interleave(5, dim=1)

    intra_feat = intraday.reshape(B, 60, -1)

    return torch.cat([day_feat, week_feat, intra_feat], dim=-1)


# ============================================================
# 多频率 Baseline：日频 + 周频 + 分时 → GRU
# ============================================================
class MultiFreqGRU(nn.Module):
    """
    三频率融合 GRU Baseline。
    """

    def __init__(self, hidden_dim=128, num_layers=2, num_classes=5,
                 dropout_rate=0.3):
        super().__init__()

        input_dim = 9 + 9 + 8 * 9  # 90

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, daily, weekly, intraday):
        x = _fuse_frequencies(daily, weekly, intraday)     # (B, 60, 90)
        out, _ = self.gru(x)                                # (B, 60, hidden_dim)
        out = out[:, -1, :]                                 # 取最后时刻
        out = self.dropout(out)
        return self.fc(out)


# ============================================================
# 多频率 Baseline：日频 + 周频 + 分时 → MLP
# ============================================================
class MultiFreqMLP(nn.Module):
    """
    三频率融合 MLP Baseline。
    融合后展平为 (B, 5400)，经多层全连接后分类。
    """

    def __init__(self, hidden_dim=512, num_classes=5, dropout_rate=0.3):
        super().__init__()

        input_dim = 9 + 9 + 8 * 9  # 90
        flat_dim = 60 * input_dim  # 5400

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(flat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, daily, weekly, intraday):
        x = _fuse_frequencies(daily, weekly, intraday)     # (B, 60, 90)
        x = x.reshape(x.size(0), -1)                        # (B, 5400)
        return self.mlp(x)


# ============================================================
# 多频率 Baseline：日频 + 周频 + 分时 → TCN
# ============================================================
class MultiFreqTCN(nn.Module):
    """
    三频率融合 TCN Baseline。
    融合后送入 TCNEncoder，时间平均池化后分类。
    """

    def __init__(self, hidden_dim=128, num_classes=5, dropout_rate=0.3,
                 kernel_size=3, num_levels=4):
        super().__init__()

        input_dim = 9 + 9 + 8 * 9  # 90

        self.encoder = TCNEncoder(input_dim, hidden_dim, num_levels,
                                  kernel_size, dropout_rate)
        self.classifier = ClassifierHead(hidden_dim, hidden_dim,
                                         num_classes, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = _fuse_frequencies(daily, weekly, intraday)     # (B, 60, 90)
        feat = self.encoder(x)                              # (B, hidden_dim, T)
        feat = torch.mean(feat, dim=2)                      # 时间平均池化
        return self.classifier(feat)


# ============================================================
# 多频率 Baseline：日频 + 周频 + 分时 → LSTM
# ============================================================
class MultiFreqLSTM(nn.Module):
    """
    三频率融合 LSTM Baseline（双向）。
    融合后送入 LSTMEncoder，时间平均池化后分类。
    """

    def __init__(self, hidden_dim=128, num_layers=2, num_classes=5,
                 dropout_rate=0.3):
        super().__init__()

        input_dim = 9 + 9 + 8 * 9  # 90

        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers,
                                   dropout_rate)
        # 双向 LSTM 输出维度翻倍
        self.classifier = ClassifierHead(hidden_dim * 2, hidden_dim,
                                         num_classes, dropout_rate)

    def forward(self, daily, weekly, intraday):
        x = _fuse_frequencies(daily, weekly, intraday)     # (B, 60, 90)
        feat = self.encoder(x)                              # (B, 2*hidden_dim, T)
        feat = torch.mean(feat, dim=2)                      # 时间平均池化
        return self.classifier(feat)


# ============================================================
# TCN 残差块
# ============================================================
class Chomp1d(nn.Module):
    """保证因果卷积：去掉右侧 padding 产生的多余时间步"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
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
        out = F.relu(self.bn1(self.chomp1(self.conv1(x))))
        out = self.dropout1(out)
        out = F.relu(self.bn2(self.chomp2(self.conv2(out))))
        out = self.dropout2(out)
        return F.relu(out + residual)


# ============================================================
# 时序编码器
# ============================================================
class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_levels=4, kernel_size=3, dropout_rate=0.3):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout_rate))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)
        return self.network(x)  # (B, hidden_dim, T)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)          # (B, T, 2*hidden_dim)
        return out.transpose(1, 2)     # (B, 2*hidden_dim, T)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout_rate if num_layers > 1 else 0)

    def forward(self, x):
        out, _ = self.gru(x)           # (B, T, 2*hidden_dim)
        return out.transpose(1, 2)     # (B, 2*hidden_dim, T)


# ============================================================
# 分类头部
# ============================================================
class ClassifierHead(nn.Module):
    def __init__(self, encoder_out_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(encoder_out_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, C)
        return self.head(x)


# ============================================================
# 统一分类器
# ============================================================
class SeqClassifier(nn.Module):
    """
    时序分类模型，支持 tcn / lstm / gru 三种编码器。
    - tcn:       encoder_out_dim = hidden_dim
    - lstm/gru:  encoder_out_dim = hidden_dim * 2（双向）
    """

    def __init__(self, model_type='lstm', input_dim=9, hidden_dim=128,
                 num_layers=2, num_classes=5, dropout_rate=0.3,
                 kernel_size=3, num_levels=4):
        super().__init__()
        self.model_type = model_type

        if model_type == 'tcn':
            self.encoder = TCNEncoder(input_dim, hidden_dim, num_levels,
                                      kernel_size, dropout_rate)
            encoder_out_dim = hidden_dim
        elif model_type == 'lstm':
            self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout_rate)
            encoder_out_dim = hidden_dim * 2
        elif model_type == 'gru':
            self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout_rate)
            encoder_out_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.classifier = ClassifierHead(encoder_out_dim, hidden_dim, num_classes, dropout_rate)

    def forward(self, x):
        """
        x: (B, T, F)
        """
        feat = self.encoder(x)                       # (B, C, T)
        feat = torch.mean(feat, dim=2)               # 时间维平均池化 → (B, C)
        logits = self.classifier(feat)               # (B, num_classes)
        return logits


if __name__ == '__main__':
    for mtype in ['tcn', 'lstm', 'gru']:
        model = SeqClassifier(model_type=mtype, input_dim=9, hidden_dim=128,
                              num_layers=2, num_classes=5, dropout_rate=0.3)
        x = torch.randn(8, 60, 9)  # (B=8, T=60, F=9)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{mtype.upper():>6}: input={x.shape} → output={out.shape}, params={n_params:,}")
