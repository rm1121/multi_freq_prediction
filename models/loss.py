import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(loss_type, alpha=None, gamma=2.0, label_smoothing=0.0,
                      reduction='mean'):
    """Construct the configured primary classification loss.

    ``label_smoothing`` applies to cross entropy only because the current
    FocalLoss implementation does not define label-smoothing behavior.
    """
    loss_type = str(loss_type).lower()
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    if loss_type == 'ce':
        weight = None
        if alpha is not None:
            weight = torch.as_tensor(alpha, dtype=torch.float32)
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    raise ValueError(
        f"Unsupported loss type: {loss_type!r}. Supported types are 'focal' and 'ce'."
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: 类别权重（Tensor 类型），例如 [1.0, 0.25, 1.0] 表示对类别 1 降低权重
        gamma: 难易样本调节因子，默认 2.0
        reduction: 'mean' or 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        inputs: shape [B, C]，经过 softmax 之前的 logits
        targets: shape [B]，真实标签，int 类型
        """
        # 计算 softmax 概率
        probs = F.softmax(inputs, dim=1)
        # 获取每个样本对应的概率
        probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # shape: [B]

        # log(p)
        log_p = torch.log(probs + 1e-9)

        # alpha 权重
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            log_p = log_p * at

        loss = -1 * (1 - probs) ** self.gamma * log_p

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class RankingLoss(nn.Module):
    """
    排序损失（Ranking Loss）：基于 Spearman 秩相关系数的可微近似。

    核心思想：
    不要求模型精确预测未来收益率的数值，而是要求预测值与真实值的
    相对排序一致。即：未来收益率高的时间步，预测分数也应当更高。

    L = 1 - τ_spearman(pred, target)

    使用可微的 ranking 近似，基于 pairwise comparison：
    L = mean( max(0, (sign(t_i - t_j) - sign(p_i - p_j))^2 ) )

    但更稳定的做法是直接优化 Spearman 的近似版本：
    将 target 和 pred 都转为排序分数（rank），然后计算 MSE。

    Args:
        temperature: soft_rank 的平滑温度（越大越接近硬排序）
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def _soft_rank(self, x, dim=-1):
        """可微的排序近似：通过 softmax + 累计求和模拟排序。"""
        # 对每个元素，计算与其他元素的 pairwise 比较
        # x: (B, T)
        B, T = x.shape
        # pairwise difference: (B, T, T)
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # x_i - x_j
        # 符号函数近似: sigmoid(temp * diff)
        sign = torch.sigmoid(self.temperature * diff)  # (B, T, T)
        # 每个元素的 soft rank = 比它小的元素个数 + 1
        # 沿 dim=2 求和（对每个 i，sum over j of sign(x_i > x_j)）
        # 但实际上我们想要的是"比 x_i 小的元素个数"
        # sign(x_i - x_j) > 0.5 表示 x_i > x_j
        ranks = sign.sum(dim=2)  # (B, T)
        return ranks

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, T) — 模型预测的排序分数
            target: (B, T) — 真实未来收益率序列
        Returns:
            scalar loss
        """
        # 计算 soft rank
        pred_rank = self._soft_rank(pred)
        target_rank = self._soft_rank(target)

        # 归一化到 [0, 1]
        T = pred.size(1)
        pred_rank = pred_rank / T
        target_rank = target_rank / T

        # MSE 损失（让预测的排序与真实排序一致）
        loss = F.mse_loss(pred_rank, target_rank)
        return loss


class TrendDirectionLoss(nn.Module):
    """
    趋势方向损失：预测未来累计收益的最终符号。

    将未来累计收益曲线的最終值离散化为 3 类：
      0: 下跌（cum_ret[-1] < -threshold）
      1: 震荡（|cum_ret[-1]| <= threshold）
      2: 上涨（cum_ret[-1] > threshold）

    使用 CE 损失，与主任务一致的量级。
    """
    def __init__(self, threshold=0.05, label_smoothing=0.0):
        super().__init__()
        self.threshold = threshold
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _get_trend_labels(self, cum_seq):
        """从累计收益曲线生成趋势方向标签 (B,)"""
        final_ret = cum_seq[:, -1]  # (B,) 最后一天的累计收益
        labels = torch.zeros_like(final_ret, dtype=torch.long)
        labels[final_ret > self.threshold] = 2
        labels[final_ret < -self.threshold] = 0
        labels[(final_ret >= -self.threshold) & (final_ret <= self.threshold)] = 1
        return labels

    def forward(self, trend_logits, cum_seq):
        """
        Args:
            trend_logits: (B, 3) — 趋势方向 logits
            cum_seq:      (B, 50) — 未来累计收益曲线（真实值）
        Returns:
            scalar loss
        """
        labels = self._get_trend_labels(cum_seq)
        return self.ce(trend_logits, labels)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    对每个 anchor，正样本 = 同 batch 中同 label 的其他样本，
    负样本 = 同 batch 中不同 label 的样本。

    L_i = -1/|P(i)| * sum_{p∈P(i)} log( exp(z_i·z_p / τ) / sum_{a∈A(i)} exp(z_i·z_a / τ) )

    Args:
        temperature: 温度参数 τ，默认 0.07
        base_temperature: 基础温度，用于缩放 loss，默认 0.07
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) — L2 归一化后的特征向量
            labels:   (B,)  — 真实标签
        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # L2 normalize
        features = F.normalize(features, dim=1)

        # 相似度矩阵 (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # 正样本 mask: (B, B)，同 label 且不是自己
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)  # (B, B)
        mask_pos = mask_pos - torch.eye(batch_size, device=device)  # 去掉自己

        # 去掉正样本数 = 0 的行（该 label 只出现一次）
        n_pos = mask_pos.sum(dim=1)  # (B,)
        valid = n_pos > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        # 数值稳定：减去每行最大值
        sim_max = sim.detach().max(dim=1, keepdim=True)[0]
        sim = sim - sim_max

        # exp(sim)
        exp_sim = torch.exp(sim)

        # 分母：所有负样本的 exp 之和 + 所有正样本的 exp 之和
        # mask_neg = 1 - mask_pos - eye
        mask_neg = (1 - mask_pos) - torch.eye(batch_size, device=device)
        mask_neg = mask_neg.clamp(min=0)

        # 对每个 anchor: sum_{a≠i} exp(sim[i,a])
        denom = (exp_sim * (mask_pos + mask_neg)).sum(dim=1, keepdim=True)  # (B, 1)

        # 分子：sum_{p∈P(i)} exp(sim[i,p])
        # 然后取 log
        log_prob = sim - torch.log(denom + 1e-9)  # (B, B)，每行是 log(exp(sim)/denom)

        # 只保留正样本的 log_prob 并求和
        log_prob_pos = (log_prob * mask_pos).sum(dim=1)  # (B,)

        # 除以正样本数
        loss = - (self.base_temperature / self.temperature) * log_prob_pos / n_pos.clamp(min=1)

        # 只对有正样本的 anchor 计算 loss
        loss = loss[valid].mean()

        return loss
