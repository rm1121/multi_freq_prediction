"""
多频率数据集加载：日频 (12,5,8)、周频 (1,12,8)、分时 (60,13,8)
加载时对类别2（震荡）做欠采样，使其与其他类别数量相当。

优化：引入全局 SVD 缓存（_DECOMP_CACHE），对同一组 ssa_cfg（相同 window/trend_k/seasonal_k）
多次调用 load_data 只做一次完整的 SSA 分解，后续复用缓存结果。
"""

import numpy as np
import torch
import hashlib
import json
from torch.utils.data import Dataset, DataLoader

from models.decomp import apply_decomp
import os

# 欠采样随机种子
_BALANCE_SEED = 42

# ================== 跨变体 SVD 缓存 ==================
# 对同一组 (window, trend_k, seasonal_k, scales, method) 只做一次完整分解，
# 一次性算出全部 5 个 group (raw/trend/seasonal/noise/signal) 并缓存。
# ablation 的 5 个变体串行调用 load_data 时，只有第一次做 SVD，后续 4 次从缓存拿。
_DECOMP_CACHE: dict = {}
_DECOMP_ALL_GROUPS = ['raw', 'trend', 'seasonal', 'noise', 'signal']


def _decomp_cache_key(cfg):
    """生成分解配置的缓存键（排除 group，因为不同 group 共享 SVD）。"""
    if cfg is None:
        return None
    key_fields = {k: cfg.get(k) for k in
                  ('method', 'window', 'trend_k', 'seasonal_k', 'scales',
                   'periods', 'wavelet', 'level', 'noise_levels')}
    raw = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


class MultiFreqDataset(Dataset):
    """多频率数据集：日频 + 周频 + 分时 + (可选)宏观指数 + (可选)辅助目标 → 标签

    SSA 分解已在 load_data() 中完成，此处不再重复应用。
    macro / macro4idx / future_ret_seq / future_cum_seq / aux_stats 为可选。
    若为 None 则 __getitem__ 返回对应长度的元组（向后兼容）。
    """
    def __init__(self, daily, weekly, intraday, labels, macro=None,
                 future_ret_seq=None, future_cum_seq=None, aux_stats=None,
                 macro4idx=None, macro_weekly=None):
        self.daily = daily            # (N, 12, 5, 8)
        self.weekly = weekly          # (N, 1, 12, 8)
        self.intraday = intraday      # (N, 60, 13, 8)
        self.labels = labels          # (N,)
        self.macro = macro            # (N, 60, 14) 或 None（旧版，沪深300+中证500）
        self.macro4idx = macro4idx    # (N, 60, 56) 或 None（新版，4指数14维）
        self.macro_weekly = macro_weekly  # (N, 12, 56) 或 None（周频4指数）
        self.future_ret_seq = future_ret_seq  # (N, 50) 或 None
        self.future_cum_seq = future_cum_seq  # (N, 50) 或 None
        self.aux_stats = aux_stats    # (N, 8) 或 None

        # Every modality must represent exactly the same sample identities.  Truncating
        # to the shortest input silently shifts labels and invalidates metrics.
        arrays = {
            'daily': self.daily, 'weekly': self.weekly, 'intraday': self.intraday,
            'labels': self.labels, 'macro': self.macro,
            'macro4idx': self.macro4idx, 'macro_weekly': self.macro_weekly,
            'future_ret_seq': self.future_ret_seq,
            'future_cum_seq': self.future_cum_seq, 'aux_stats': self.aux_stats,
        }
        expected = len(self.labels)
        mismatches = {name: len(array) for name, array in arrays.items()
                      if array is not None and len(array) != expected}
        if mismatches:
            raise ValueError(
                f'MultiFreqDataset sample-count mismatch: labels={expected}, '
                f'other arrays={mismatches}. Regenerate aligned processed data.')
        self._len = expected

        self.batch_fields = ('daily', 'weekly', 'intraday', 'label')
        if self.macro4idx is not None or self.macro is not None:
            self.batch_fields += ('macro',)
        if self.macro_weekly is not None:
            self.batch_fields += ('macro_weekly',)
        if self.future_ret_seq is not None:
            self.batch_fields += ('future_ret_seq',)
        if self.future_cum_seq is not None:
            self.batch_fields += ('future_cum_seq',)
        if self.aux_stats is not None:
            self.batch_fields += ('aux_stats',)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        items = (
            torch.tensor(self.daily[idx], dtype=torch.float32),
            torch.tensor(self.weekly[idx], dtype=torch.float32),
            torch.tensor(self.intraday[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx].item(), dtype=torch.long),
        )
        # 新版 macro4idx（4指数14维×4=56维，日频）
        if self.macro4idx is not None:
            items = items + (torch.tensor(self.macro4idx[idx], dtype=torch.float32),)
        # 旧版 macro（沪深300+中证500，14/16维）
        elif self.macro is not None:
            items = items + (torch.tensor(self.macro[idx], dtype=torch.float32),)
        # 周频 macro4idx（4指数，12周）
        if self.macro_weekly is not None:
            items = items + (torch.tensor(self.macro_weekly[idx], dtype=torch.float32),)
        if self.future_ret_seq is not None:
            items = items + (torch.tensor(self.future_ret_seq[idx], dtype=torch.float32),)
        if self.future_cum_seq is not None:
            items = items + (torch.tensor(self.future_cum_seq[idx], dtype=torch.float32),)
        if self.aux_stats is not None:
            items = items + (torch.tensor(self.aux_stats[idx], dtype=torch.float32),)
        return items

def undersample_class_2(daily, weekly, intraday, labels, future_ret_seq=None, future_cum_seq=None, aux_stats=None,
                        target_per_class=None, seed=_BALANCE_SEED):
    """
    对类别2做欠采样，使其与其他类别数量相当。
    
    默认以其他类别的中位数作为 target 数量上限，
    仅对超过该上限的类别做随机降采样。
    
    返回 (降采样后的 (daily, weekly, intraday, labels, future_ret_seq, future_cum_seq, aux_stats),
           keep_idx)，其中 keep_idx 为保留样本的原始索引，用于对 macro 等
           未参与欠采样的数组做同样的对齐裁剪。
    """
    unique, counts = np.unique(labels, return_counts=True)

    if target_per_class is None:
        # 剔除类别2后取中位数作为目标上限
        non_c2_counts = [c for u, c in zip(unique, counts) if u != 2]
        target_per_class = int(np.median(non_c2_counts))

    rng = np.random.RandomState(seed)
    keep_idx = []
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        n_keep = min(len(cls_idx), target_per_class)
        if n_keep < len(cls_idx):
            cls_idx = rng.choice(cls_idx, n_keep, replace=False)
        keep_idx.extend(cls_idx.tolist())

    keep_idx = np.sort(keep_idx)
    result = (daily[keep_idx], weekly[keep_idx],
              intraday[keep_idx], labels[keep_idx])
    if future_ret_seq is not None:
        result = result + (future_ret_seq[keep_idx],)
    if future_cum_seq is not None:
        result = result + (future_cum_seq[keep_idx],)
    if aux_stats is not None:
        result = result + (aux_stats[keep_idx],)
    return result, keep_idx


def load_data(processed_dir='data/processeddata2', num_classes=5, ssa_cfg=None,
              macro_suffix='', aux_targets=False, macro4idx_suffix='',
              macro4idx_weekly_suffix=None):
    """
    加载训练/验证/测试集的多频率数据，并对所有集合做类别2欠采样。

    ssa_cfg: 见 MultiFreqDataset。若提供，则在欠采样后对子集做一次 SSA 变换。
    macro_suffix: macro 文件后缀。若为 'noamp' 则加载 macro_noamp.npy；
                  若为 '' 则加载 macro.npy。不存在时对应 split 返回 None。
    macro4idx_suffix: 新版4指数macro文件后缀。若提供则加载 macro4idx{macro4idx_suffix}.npy，
                      且优先级低于 macro_suffix（旧版 macro 存在时优先用旧版）。
    macro4idx_weekly_suffix: 周频4指数macro文件后缀。若提供则加载 macro4idx_weekly{macro4idx_weekly_suffix}.npy。
    aux_targets: 是否加载未来收益率序列和辅助统计量（用于多任务学习）。

    Returns:
        ((d_train, w_train, i_train, y_train),
         (d_val,   w_val,   i_val,   y_val),
         (d_test,  w_test,  i_test,  y_test)),
        (m_train, m_val, m_test),       # macro 数据
        (mw_train, mw_val, mw_test),    # 周频 macro 数据
        (fr_train, fr_val, fr_test),    # 未来收益率序列（aux_targets=True 时）
        (fc_train, fc_val, fc_test),    # 未来累计收益曲线（aux_targets=True 时）
        (as_train, as_val, as_test),    # 辅助统计量（aux_targets=True 时）
    """
    data = {}
    for split in ['train', 'val', 'test']:
        base = f'{processed_dir}/{split}'
        data[split] = {
            'daily':    np.load(f'{base}/daily.npy'),
            'weekly':   np.load(f'{base}/weekly.npy'),
            'intraday': np.load(f'{base}/intraday.npy'),
            'labels':   np.load(f'{base}/labels.npy'),
        }
        if aux_targets:
            fr_path = f'{base}/future_ret_seq.npy'
            data[split]['future_ret_seq'] = np.load(fr_path) if os.path.exists(fr_path) else None
            fc_path = f'{base}/future_cum_seq.npy'
            data[split]['future_cum_seq'] = np.load(fc_path) if os.path.exists(fc_path) else None
            as_path = f'{base}/aux_stats.npy'
            data[split]['aux_stats'] = np.load(as_path) if os.path.exists(as_path) else None

    # === 5类 → 3类映射（仅当 num_classes == 3 且数据为5类标签时） ===
    if num_classes == 3:
        all_labels = np.concatenate([data[s]['labels'] for s in ['train', 'val', 'test']])
        if all_labels.max() > 2:  # 原始是5类标签才做映射
            for split in ['train', 'val', 'test']:
                y = data[split]['labels']
                y[y == 1] = 0
                y[y == 2] = 1
                y[y == 3] = 2
                y[y == 4] = 2

    # === 欠采样已按用户要求禁用：所有集合全量保留，不做类别降采样 ===
    # keep_idx_map 全部置为 None，因此 macro/macro4idx/macro_weekly 也会原样保留，
    # 与 labels 天然长度对齐，不会出现越界。
    keep_idx_map = {}
    for split in ['train', 'val', 'test']:
        d = data[split]
        keep_idx_map[split] = None
        data[split] = {
            'daily': d['daily'], 'weekly': d['weekly'],
            'intraday': d['intraday'], 'labels': d['labels'],
        }
        if aux_targets:
            data[split]['future_ret_seq'] = d.get('future_ret_seq')
            data[split]['future_cum_seq'] = d.get('future_cum_seq')
            data[split]['aux_stats'] = d.get('aux_stats')

    # === 可选：时序分解/去噪（带跨变体缓存） ===
    if ssa_cfg is not None and ssa_cfg.get('group', 'raw') != 'raw':
        method = ssa_cfg.get('method', 'ssa')
        group = ssa_cfg.get('group', 'signal')
        scales = ssa_cfg.get('scales', ['daily', 'weekly', 'intraday'])

        cache_key = _decomp_cache_key(ssa_cfg)
        if cache_key and cache_key in _DECOMP_CACHE:
            cached = _DECOMP_CACHE[cache_key]
            group_results = cached.get(group)
            if group_results is not None:
                print(f"[DECOMP CACHE] 命中缓存: method={method}, group={group}")
                for split in ['train', 'val', 'test']:
                    data[split] = group_results[split]
                # Only core decomposed arrays are cached.  Continue below so macro,
                # weekly macro and auxiliary targets are loaded for this invocation.
                # Returning here used to drop all optional modalities on cache hits.

        # 如果是 SSA 方法且支持缓存，一次性算全部 5 个 group
        if cache_key and method == 'ssa':
            print(f"[DECOMP] 开始分解: method={method}（一次性计算全部 {len(_DECOMP_ALL_GROUPS)} 个变体用于缓存）...")
            all_groups_data = {}
            for g in _DECOMP_ALL_GROUPS:
                g_cfg = dict(ssa_cfg, group=g)
                g_data = {}
                for split in ['train', 'val', 'test']:
                    if g == 'raw':
                        g_data[split] = data[split]
                    else:
                        n = data[split]['daily'].shape[0]
                        print(f"  {g} → {split} ({n} 样本)...", end='', flush=True)
                        d_daily, d_weekly, d_intra, d_labels = (
                            data[split]['daily'], data[split]['weekly'],
                            data[split]['intraday'], data[split]['labels'])
                        d_daily, d_weekly, d_intra = apply_decomp(
                            d_daily, d_weekly, d_intra, g_cfg)
                        g_data[split] = {
                            'daily': d_daily, 'weekly': d_weekly,
                            'intraday': d_intra, 'labels': d_labels,
                        }
                        if aux_targets:
                            g_data[split]['future_ret_seq'] = data[split].get('future_ret_seq')
                            g_data[split]['future_cum_seq'] = data[split].get('future_cum_seq')
                            g_data[split]['aux_stats'] = data[split].get('aux_stats')
                        print(f" 完成", flush=True)
                all_groups_data[g] = g_data
            # 缓存全部 5 个 group
            _DECOMP_CACHE[cache_key] = all_groups_data
            # 返回当前请求的 group
            result = all_groups_data[group]
            for split in ['train', 'val', 'test']:
                data[split] = result[split]
            print(f"[DECOMP] 全部 5 个变体已缓存，当前返回 {group}")
        else:
            # 非 SSA 方法（FFT/小波本身很快，不缓存）
            print(f"[DECOMP] 开始分解: method={method}, group={group}, scales={scales}")
            for split in ['train', 'val', 'test']:
                n = data[split]['daily'].shape[0]
                print(f"[DECOMP]   → 处理 {split} 集 ({n} 样本)...", end='', flush=True)
                d_daily, d_weekly, d_intra, d_labels = (
                    data[split]['daily'], data[split]['weekly'],
                    data[split]['intraday'], data[split]['labels'])
                d_daily, d_weekly, d_intra = apply_decomp(
                    d_daily, d_weekly, d_intra, ssa_cfg)
                data[split] = {
                    'daily': d_daily, 'weekly': d_weekly,
                    'intraday': d_intra, 'labels': d_labels,
                }
                if aux_targets:
                    data[split]['future_ret_seq'] = data[split].get('future_ret_seq')
                    data[split]['future_cum_seq'] = data[split].get('future_cum_seq')
                    data[split]['aux_stats'] = data[split].get('aux_stats')
                print(f" 完成", flush=True)
            print(f"[DECOMP] 分解全部完成", flush=True)

    # === 可选：加载宏观指数序列（macro{macro_suffix}.npy，若存在则加载） ===
    def _align_macro(arr, split):
        """用欠采样的 keep_idx 对 macro 数组做同步裁剪；未欠采样的 split 原样返回。"""
        if arr is None:
            return None
        ki = keep_idx_map.get(split)
        return arr[ki] if ki is not None else arr

    macro_name = f'macro{macro_suffix}.npy'
    macro = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(processed_dir, split, macro_name)
        _m = np.load(path) if os.path.exists(path) else None
        macro.append(_align_macro(_m, split))

    # === 可选：加载新版4指数宏观数据（macro4idx{macro4idx_suffix}.npy） ===
    # 当 macro4idx_suffix 被显式提供时，优先使用新版 4 指数数据
    # 即使旧版 macro 也存在，只要 macro4idx_suffix is not None 就覆盖
    if macro4idx_suffix is not None:
        macro4idx_name = f'macro4idx{macro4idx_suffix}.npy'
        macro = []
        for split in ['train', 'val', 'test']:
            path = os.path.join(processed_dir, split, macro4idx_name)
            _m = np.load(path) if os.path.exists(path) else None
            macro.append(_align_macro(_m, split))
        if all(m is None for m in macro):
            print(f"[WARN] macro4idx{macro4idx_suffix}.npy 不存在于任何 split，macro 保持 None")

    # === 可选：加载周频4指数宏观数据（macro4idx_weekly{macro4idx_weekly_suffix}.npy） ===
    macro_weekly = None
    if macro4idx_weekly_suffix is not None:
        weekly_name = f'macro4idx_weekly{macro4idx_weekly_suffix}.npy'
        macro_weekly = []
        for split in ['train', 'val', 'test']:
            path = os.path.join(processed_dir, split, weekly_name)
            _m = np.load(path) if os.path.exists(path) else None
            macro_weekly.append(_align_macro(_m, split))
        if all(m is None for m in macro_weekly):
            print(f"[WARN] {weekly_name} 不存在于任何 split，macro_weekly 保持 None")
            macro_weekly = None

    return _build_return(data, aux_targets, macro, macro_weekly)


def _build_return(data, aux_targets, macro=None, macro_weekly=None):
    """根据是否加载辅助目标构建 load_data 返回值。"""
    train = data['train']
    val = data['val']
    test = data['test']
    
    def to_tuple(d):
        return (d['daily'], d['weekly'], d['intraday'], d['labels'])
    
    result = (
        (to_tuple(train), to_tuple(val), to_tuple(test)),
        (macro[0] if macro else None, macro[1] if macro else None, macro[2] if macro else None),
    )
    
    if macro_weekly is not None:
        mw = (macro_weekly[0], macro_weekly[1], macro_weekly[2])
        result = result + (mw,)
    
    if aux_targets:
        def to_aux(d):
            return (d.get('future_ret_seq'), d.get('future_cum_seq'), d.get('aux_stats'))
        result = result + (
            (to_aux(train), to_aux(val), to_aux(test)),
        )
    
    return result


def get_dataloaders(batch_size=64, num_classes=5,
                    processed_dir='data/processeddata2', ssa_cfg=None,
                    macro_suffix='', aux_targets=False, seed=None,
                    macro4idx_suffix='', macro4idx_weekly_suffix=None):
    """加载多频率数据并返回 DataLoader（支持可选 macro 和辅助目标）."""
    result = load_data(processed_dir, num_classes, ssa_cfg,
                       macro_suffix=macro_suffix, aux_targets=aux_targets,
                       macro4idx_suffix=macro4idx_suffix,
                       macro4idx_weekly_suffix=macro4idx_weekly_suffix)
    
    ((d_train, w_train, i_train, y_train),
     (d_val,   w_val,   i_val,   y_val),
     (d_test,  w_test,  i_test,  y_test)), \
    (m_train, m_val, m_test) = result[:2]

    # 周频 macro（在 result 的第 3 个元素，如果存在）
    mw_train, mw_val, mw_test = None, None, None
    idx = 2
    if len(result) > idx and result[idx] is not None:
        mw_data = result[idx]
        # 判断是 macro_weekly 还是 aux_targets
        if isinstance(mw_data, tuple) and len(mw_data) == 3 and mw_data[0] is not None:
            # 检查形状：如果第一个元素是 3D (N, 12, 56)，则是 macro_weekly
            if hasattr(mw_data[0], 'shape') and len(mw_data[0].shape) == 3:
                mw_train, mw_val, mw_test = mw_data
                idx = 3

    # 辅助目标
    fr_train, fr_val, fr_test = None, None, None
    fc_train, fc_val, fc_test = None, None, None
    as_train, as_val, as_test = None, None, None
    if aux_targets and len(result) > idx:
        aux_result = result[idx]
        (fr_train, fc_train, as_train) = aux_result[0]
        (fr_val,   fc_val,   as_val)   = aux_result[1]
        (fr_test,  fc_test,  as_test)  = aux_result[2]

    train_ds = MultiFreqDataset(d_train, w_train, i_train, y_train, macro=m_train,
                                 future_ret_seq=fr_train, future_cum_seq=fc_train, aux_stats=as_train,
                                 macro_weekly=mw_train)
    val_ds   = MultiFreqDataset(d_val,   w_val,   i_val,   y_val, macro=m_val,
                                 future_ret_seq=fr_val, future_cum_seq=fc_val, aux_stats=as_val,
                                 macro_weekly=mw_val)
    test_ds  = MultiFreqDataset(d_test,  w_test,  i_test,  y_test, macro=m_test,
                                 future_ret_seq=fr_test, future_cum_seq=fc_test, aux_stats=as_test,
                                 macro_weekly=mw_test)

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"训练集：{len(train_ds)} | 验证集：{len(val_ds)} | 测试集：{len(test_ds)}")
    return train_loader, val_loader, test_loader


# ======================== 自检入口 ========================
if __name__ == '__main__':
    ((d_train, w_train, i_train, y_train),
     (d_val,   w_val,   i_val,   y_val),
     (d_test,  w_test,  i_test,  y_test)), _ = load_data(num_classes=5)

    label_map = {0: "大幅下跌", 1: "小幅下跌", 2: "震荡",
                 3: "小幅拉升", 4: "大幅拉升"}

    for name, (d, w, i_arr, y) in [
        ("训练集", (d_train, w_train, i_train, y_train)),
        ("验证集", (d_val,   w_val,   i_val,   y_val)),
        ("测试集", (d_test,  w_test,  i_test,  y_test)),
    ]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n{'=' * 50}")
        print(f"{name}: {len(y)} 样本")
        print(f"  daily:    {d.shape}  ({d.nbytes / 1024**2:.1f} MB)")
        print(f"  weekly:   {w.shape}  ({w.nbytes / 1024**2:.1f} MB)")
        print(f"  intraday: {i_arr.shape}  ({i_arr.nbytes / 1024**2:.1f} MB)")
        print(f"  labels:   {y.shape}")
        print(f"  {'类别':<12} {'数量':>6} {'占比':>7}")
        print(f"  {'-' * 25}")
        for cls, cnt in sorted(zip(unique, counts)):
            print(f"  {label_map[cls]:<12} {cnt:>6} {cnt/len(y)*100:>6.2f}%")
        print(f"  {'-' * 25}")
        print(f"  {'合计':<12} {len(y):>6}")
