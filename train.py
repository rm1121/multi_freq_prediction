import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import inspect
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import get_model, FocalLoss
from data.dataset import get_dataloaders
from utils.utils import EarlyStopping, set_seed


def _model_accepts_macro(model, n_params=4):
    """检查模型 forward 是否接受 macro 参数（兼容 DataParallel 包装）。
    n_params=4: 接受 (daily, weekly, intraday, macro)
    n_params=5: 接受 (daily, weekly, intraday, macro, macro_weekly)
    """
    if isinstance(model, nn.DataParallel):
        m = model.module
    else:
        m = model
    sig = inspect.signature(m.forward)
    params = list(sig.parameters.values())
    non_self = [p for p in params if p.name != 'self']
    return len(non_self) >= n_params


def get_loss_function(loss_name='ce', alpha=0.25, gamma=2.0, label_smoothing=0.0):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def plot_metrics(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, values in metrics_dict.items():
        plt.figure(figsize=(8, 5))
        if values and isinstance(values[0], list):
            values_t = list(zip(*values))
            for class_idx, class_values in enumerate(values_t):
                plt.plot(class_values, label=f'{metric_name}_class{class_idx}')
        else:
            plt.plot(values, label=metric_name)
        plt.title(metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}.png"))
        plt.close()


def train(model, train_loader, criterion, optimizer, device,
          cl_criterion=None, cl_weight=0.0,
          aux_seq_weight=0.0, aux_stat_weight=0.0,
          seq_loss_fn=None, stat_loss_fn=None,
          use_cum_seq=False,
          rank_loss_fn=None, trend_loss_fn=None,
          aux_rank_weight=0.0, aux_trend_weight=0.0):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_cl = 0.0
    running_seq = 0.0
    running_stat = 0.0
    running_rank = 0.0
    running_trend = 0.0
    for batch in train_loader:
        daily, weekly, intraday, y_batch = batch[:4]
        daily, weekly, intraday = daily.to(device), weekly.to(device), intraday.to(device)
        y_batch = y_batch.to(device).long()
        optimizer.zero_grad()

        # 解析 batch 元素：固定顺序为
        # [daily, weekly, intraday, label, macro?, macro_weekly?, future_ret_seq?, future_cum_seq?, aux_stats?]
        # 根据 len(batch) 判断哪些元素存在
        n = len(batch)
        macro = batch[4].to(device) if n > 4 else None
        macro_weekly = batch[5].to(device) if n > 5 else None
        has_ret_seq = n > 6
        has_cum_seq = n > 7
        has_aux_stats = n > 8

        if macro_weekly is not None and _model_accepts_macro(model, n_params=5):
            outputs = model(daily, weekly, intraday, macro, macro_weekly)
        elif macro is not None and _model_accepts_macro(model):
            outputs = model(daily, weekly, intraday, macro)
        else:
            outputs = model(daily, weekly, intraday)

        # 多任务 V3 模型返回 5 元组 (logits, proj, aux_rank, aux_stat, aux_trend)
        if isinstance(outputs, tuple) and len(outputs) == 5:
            logits, proj, aux_rank_pred, aux_stat_pred, aux_trend_pred = outputs
            aux_seq_pred = None  # V3 没有 MSE 序列预测
        # 多任务 V1/V2 模型返回 4 元组 (logits, proj, aux_seq, aux_stat)
        elif isinstance(outputs, tuple) and len(outputs) == 4:
            logits, proj, aux_seq_pred, aux_stat_pred = outputs
            aux_rank_pred, aux_trend_pred = None, None
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            logits, proj = outputs
            aux_seq_pred, aux_stat_pred = None, None
            aux_rank_pred, aux_trend_pred = None, None
        else:
            logits = outputs
            proj = aux_seq_pred = aux_stat_pred = None
            aux_rank_pred, aux_trend_pred = None, None

        # CE loss
        ce_loss = criterion(logits, y_batch)

        # SupCon loss（仅在训练时计算）
        cl_loss = torch.tensor(0.0, device=device)
        if cl_criterion is not None and proj is not None and cl_weight > 0:
            cl_loss = cl_criterion(proj, y_batch)

        # 实验1 loss：未来序列 MSE（V1/V2 模型使用）
        seq_loss = torch.tensor(0.0, device=device)
        if aux_seq_weight > 0 and aux_seq_pred is not None and seq_loss_fn is not None:
            if use_cum_seq and has_cum_seq:
                future_seq = batch[6].to(device)
            elif has_ret_seq:
                future_seq = batch[5].to(device)
            else:
                future_seq = None
            if future_seq is not None:
                seq_loss = seq_loss_fn(aux_seq_pred, future_seq)

        # 实验2 loss：统计量 MSE
        stat_loss = torch.tensor(0.0, device=device)
        if aux_stat_weight > 0 and aux_stat_pred is not None and stat_loss_fn is not None and has_aux_stats:
            aux_stats = batch[7].to(device)
            stat_loss = stat_loss_fn(aux_stat_pred, aux_stats)

        # 排序损失（V3 模型使用）
        rank_loss = torch.tensor(0.0, device=device)
        if aux_rank_weight > 0 and aux_rank_pred is not None and rank_loss_fn is not None and has_cum_seq:
            future_cum_seq = batch[6].to(device)
            rank_loss = rank_loss_fn(aux_rank_pred, future_cum_seq)

        # 趋势方向损失（V3 模型使用）
        trend_loss = torch.tensor(0.0, device=device)
        if aux_trend_weight > 0 and aux_trend_pred is not None and trend_loss_fn is not None and has_cum_seq:
            future_cum_seq = batch[6].to(device)
            trend_loss = trend_loss_fn(aux_trend_pred, future_cum_seq)

        loss = (ce_loss + cl_weight * cl_loss
                + aux_seq_weight * seq_loss
                + aux_stat_weight * stat_loss
                + aux_rank_weight * rank_loss
                + aux_trend_weight * trend_loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += ce_loss.item()
        running_cl += cl_loss.item()
        running_seq += seq_loss.item()
        running_stat += stat_loss.item()
        running_rank += rank_loss.item()
        running_trend += trend_loss.item()

    n = len(train_loader)
    return (running_loss / n, running_ce / n, running_cl / n,
            running_seq / n, running_stat / n,
            running_rank / n, running_trend / n)


def validate(model, val_loader, criterion, device, num_classes=5):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            daily, weekly, intraday, labels = batch[:4]
            daily, weekly, intraday = daily.to(device), weekly.to(device), intraday.to(device)
            labels = labels.to(device)
            macro = batch[4].to(device) if len(batch) > 4 else None
            if macro is not None and _model_accepts_macro(model):
                outputs = model(daily, weekly, intraday, macro)
            else:
                outputs = model(daily, weekly, intraday)
            # 多任务模型返回 4/5 元组，对比学习返回 2 元组
            if isinstance(outputs, tuple) and len(outputs) >= 4:
                logits = outputs[0]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _proj = outputs
            else:
                logits = outputs
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None,
                                labels=range(num_classes), zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None,
                          labels=range(num_classes), zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None,
                  labels=range(num_classes), zero_division=0)

    return val_loss, accuracy, precision.tolist(), recall.tolist(), f1.tolist()


def test(model, test_loader, criterion, device, num_classes=5):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            daily, weekly, intraday, labels = batch[:4]
            daily, weekly, intraday = daily.to(device), weekly.to(device), intraday.to(device)
            labels = labels.to(device)
            macro = batch[4].to(device) if len(batch) > 4 else None
            if macro is not None and _model_accepts_macro(model):
                outputs = model(daily, weekly, intraday, macro)
            else:
                outputs = model(daily, weekly, intraday)
            # 多任务模型返回 4/5 元组，对比学习返回 2 元组
            if isinstance(outputs, tuple) and len(outputs) >= 4:
                logits = outputs[0]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _proj = outputs
            else:
                logits = outputs
            loss = criterion(logits, labels)
            test_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None,
                                labels=range(num_classes), zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None,
                          labels=range(num_classes), zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None,
                  labels=range(num_classes), zero_division=0)

    return test_loss, accuracy, precision.tolist(), recall.tolist(), f1.tolist()


def main(config_path):
    # === 读取配置 ===
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['experiment']['seed'])

    # === 设置输出目录 ===
    os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_index = len(os.listdir(cfg['experiment']['output_dir'])) + 1
    exp_dir = os.path.join(cfg['experiment']['output_dir'],
                           f"{cfg['experiment']['name']}_{experiment_index}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)

    log_file = open(os.path.join(exp_dir, 'log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    # === 数据加载（可选时序分解，见 data.decomp；优先 decomp 键，兼容旧 ssa 键） ===
    ssa_cfg = cfg['data'].get('decomp', cfg['data'].get('ssa', None))
    macro_suffix = cfg['data'].get('macro_suffix', '')
    macro4idx_suffix = cfg['data'].get('macro4idx_suffix', '')
    macro4idx_weekly_suffix = cfg['data'].get('macro4idx_weekly_suffix', None)
    # 多任务模型需要加载辅助目标
    _mtl_types = {'scale_centric_future_mtl', 'scale_centric_mtl_relative',
                  'scale_centric_mtl_relative_v2', 'scale_centric_mtl_relative_v2_ranking',
                  'scale_centric_mtl_v2_ranking_ratio2_v2',
                  'scale_centric_ratio_4idx_mtl'}
    use_aux = cfg['model']['type'] in _mtl_types
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg['training']['batch_size'],
        num_classes=cfg['model']['num_classes'],
        processed_dir=cfg['data']['processed_dir'],
        ssa_cfg=ssa_cfg,
        macro_suffix=macro_suffix,
        aux_targets=use_aux,
        seed=cfg['experiment']['seed'],
        macro4idx_suffix=macro4idx_suffix,
        macro4idx_weekly_suffix=macro4idx_weekly_suffix)

    # === 设备选择：CUDA > MPS (macOS) > CPU ===
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids = [int(id) for id in cfg['gpu_ids'].split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        log(f'使用 CUDA GPU: {gpu_ids}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        log('使用 Apple MPS 加速')
    else:
        device = torch.device('cpu')
        log('使用 CPU 训练')

    # === 模型定义 ===
    model = get_model(
        model_type=cfg['model']['type'],
        input_dim=cfg['model'].get('input_dim', 9),
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        num_classes=cfg['model']['num_classes'],
        dropout_rate=cfg['model']['dropout_rate'],
        kernel_size=cfg['model'].get('kernel_size', 3),
        num_levels=cfg['model'].get('num_levels', 4),
        encoder_type=cfg['model'].get('encoder_type', 'gru'),
        d_model=cfg['model'].get('d_model', 64),
        depth=cfg['model'].get('depth', 4),
        intra_hidden=cfg['model'].get('intra_hidden', 32),
        lags=cfg['model'].get('lags', None),
        patch_size=cfg['model'].get('patch_size', 5),
        patch_embed_dim=cfg['model'].get('patch_embed_dim', 32),
        macro_dim=cfg['model'].get('macro_dim', 14),
        num_scales=cfg['model'].get('num_scales', 4),
        d_state=cfg['model'].get('d_state', 16),
        base_dt=cfg['model'].get('base_dt', 0.01),
        cnn_channels=cfg['model'].get('cnn_channels', 64),
        bilstm_hidden=cfg['model'].get('bilstm_hidden', 128),
    )

    # 多 GPU 并行（仅 CUDA）
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)

    # === 损失函数 ===
    criterion = get_loss_function(
        cfg['loss']['type'],
        alpha=cfg['loss']['alpha'],
        gamma=cfg['loss']['gamma'],
        label_smoothing=cfg['loss'].get('label_smoothing', 0.0),
    ).to(device)

    # === 对比损失（可选） ===
    cl_weight = float(cfg['loss'].get('cl_weight', 0.0))
    cl_criterion = None
    if cl_weight > 0:
        from models.loss import SupConLoss
        cl_temperature = float(cfg['loss'].get('cl_temperature', 0.07))
        cl_criterion = SupConLoss(temperature=cl_temperature).to(device)

    # === 多任务辅助损失（可选） ===
    aux_seq_weight = float(cfg['loss'].get('aux_seq_weight', 0.0))
    aux_stat_weight = float(cfg['loss'].get('aux_stat_weight', 0.0))
    seq_loss_fn = nn.MSELoss().to(device) if aux_seq_weight > 0 else None
    stat_loss_fn = nn.MSELoss().to(device) if aux_stat_weight > 0 else None

    # === 排序损失 + 趋势方向损失（V3 模型专用） ===
    aux_rank_weight = float(cfg['loss'].get('aux_rank_weight', 0.0))
    aux_trend_weight = float(cfg['loss'].get('aux_trend_weight', 0.0))
    rank_loss_fn = None
    trend_loss_fn = None
    if aux_rank_weight > 0 or aux_trend_weight > 0:
        from models.loss import RankingLoss, TrendDirectionLoss
        rank_loss_fn = RankingLoss(temperature=float(cfg['loss'].get('rank_temperature', 1.0))).to(device)
        trend_loss_fn = TrendDirectionLoss(
            threshold=float(cfg['loss'].get('trend_threshold', 0.05)),
            label_smoothing=cfg['loss'].get('label_smoothing', 0.0),
        ).to(device)

    # 余弦衰减配置（新配置项，旧配置缺失时默认 False，保证完全向后兼容）
    aux_decay = cfg['loss'].get('aux_decay', False)
    aux_seq_weight_init = aux_seq_weight
    aux_stat_weight_init = aux_stat_weight
    total_epochs = cfg['training']['epochs']

    # === 优化器（从配置选择） ===
    opt_cfg = cfg['optimizer']
    opt_type = opt_cfg.get('type', 'adam').lower()
    lr = float(cfg['training']['learning_rate'])
    wd = float(opt_cfg.get('weight_decay', 0))

    if opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # === 学习率调度器（从配置选择） ===
    sched_type = opt_cfg.get('lr_scheduler', 'step')
    if sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(opt_cfg['step_size']),
            gamma=float(opt_cfg.get('gamma', 0.5)))
    elif sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['training']['epochs'],
            eta_min=float(opt_cfg.get('eta_min', 1e-6)))
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=int(opt_cfg.get('patience', 8)),
            factor=float(opt_cfg.get('factor', 0.5)))
    elif sched_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type}")

    # === 早停 ===
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stopping_patience'],
        monitor=cfg['training'].get('early_stopping_monitor', 'loss'))

    metrics_log = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_accuracy = 0.0
    best_model_path = None
    best_val_loss = float('inf')
    best_epoch = 0

    # === 训练过程 ===
    for epoch in range(total_epochs):
        # 余弦衰减辅助损失权重（从初始值衰减到 0）
        if aux_decay:
            progress = epoch / max(total_epochs - 1, 1)
            decay_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            aux_seq_weight = aux_seq_weight_init * decay_factor
            aux_stat_weight = aux_stat_weight_init * decay_factor

        train_loss, ce_loss, cl_loss, seq_loss, stat_loss, rank_loss, trend_loss = train(
            model, train_loader, criterion, optimizer, device,
            cl_criterion=cl_criterion, cl_weight=cl_weight,
            aux_seq_weight=aux_seq_weight, aux_stat_weight=aux_stat_weight,
            seq_loss_fn=seq_loss_fn, stat_loss_fn=stat_loss_fn,
            use_cum_seq=aux_decay,
            rank_loss_fn=rank_loss_fn, trend_loss_fn=trend_loss_fn,
            aux_rank_weight=aux_rank_weight, aux_trend_weight=aux_trend_weight)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device, num_classes=cfg['model']['num_classes'])

        metrics_log['train_loss'].append(train_loss)
        metrics_log['val_loss'].append(val_loss)
        metrics_log['val_accuracy'].append(val_acc)
        metrics_log['val_precision'].append(val_prec)
        metrics_log['val_recall'].append(val_rec)
        metrics_log['val_f1'].append(val_f1)

        if (cl_weight > 0 or aux_seq_weight > 0 or aux_stat_weight > 0
                or aux_rank_weight > 0 or aux_trend_weight > 0):
            loss_detail = []
            if cl_weight > 0:
                loss_detail.append(f"CE {ce_loss:.4f} + {cl_weight:.2f}*CL {cl_loss:.4f}")
            else:
                loss_detail.append(f"CE {ce_loss:.4f}")
            if aux_seq_weight > 0:
                loss_detail.append(f"{aux_seq_weight:.3f}*SEQ {seq_loss:.4f}")
            if aux_stat_weight > 0:
                loss_detail.append(f"{aux_stat_weight:.3f}*STAT {stat_loss:.4f}")
            if aux_rank_weight > 0:
                loss_detail.append(f"{aux_rank_weight:.3f}*RANK {rank_loss:.4f}")
            if aux_trend_weight > 0:
                loss_detail.append(f"{aux_trend_weight:.3f}*TREND {trend_loss:.4f}")
            decay_info = f" [decay={decay_factor:.3f}]" if aux_decay else ""
            log(f"Epoch [{epoch+1}/{total_epochs}]{decay_info}, "
                f"Train Loss: {train_loss:.4f} ({' + '.join(loss_detail)}), "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            log(f"Epoch [{epoch+1}/{total_epochs}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 更新最佳模型（按 val_acc，独立于早停逻辑）
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

        # 早停判断
        early_stopping(epoch + 1, val_loss, val_acc)
        if early_stopping.early_stop:
            log(f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best {early_stopping.monitor} at epoch {early_stopping.best_epoch}.")
            break

        if scheduler is not None:
            if sched_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # === 保存最后一个模型 ===
    last_model_path = os.path.join(exp_dir, 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    log(f"Last model saved to {last_model_path}")

    plot_metrics(metrics_log, save_dir=os.path.join(exp_dir, 'metrics_plot'))

    # === 保存最佳模型 ===
    best_model_path = os.path.join(
        exp_dir,
        f'best_model_epoch_{best_epoch}_acc_{best_val_accuracy:.4f}_loss_{best_val_loss:.4f}.pth'
    )
    torch.save(best_model_state, best_model_path)
    log(f"Best model {cfg['model']['type']} saved at epoch {best_epoch} "
        f"with accuracy: {best_val_accuracy:.4f} and loss: {best_val_loss:.4f}")

    # === 测试集评估（加载最佳模型） ===
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = test(
        model, test_loader, criterion, device, num_classes=cfg['model']['num_classes'])

    log(f"Test Accuracy: {test_acc:.4f}")
    for cls_idx in range(cfg['model']['num_classes']):
        log(
            f"Class {cls_idx}: Precision: {test_prec[cls_idx]:.4f} | "
            f"Recall: {test_rec[cls_idx]:.4f} | F1: {test_f1[cls_idx]:.4f}"
        )
    log_file.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./utils/config.yaml')
    args = parser.parse_args()
    main(args.config)
