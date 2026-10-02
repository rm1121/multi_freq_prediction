#!/usr/bin/env python3
"""
US 版宏观指数生成器：4 指数宏观对齐到 processeddata2_{exchange} 的每个样本。

对应 A 股版 preprocess_macro_new.py 的美股移植，关键差异：
  1. 指数源：Yahoo 日线 CSV（data/rawdata/index_us/{SPX,NDQ,DJI,RUT}.csv），
     4 指数 = 标普500 / 纳指综合 / 道指 / 罗素2000。
  2. 成份宽度指标（up_ratio/down_ratio/up_limit_ratio/down_limit_ratio/rel_advance_decline）
     从本交易所日线 us_stocks_raw_{exchange}.csv 按日统计得到。美股无指数成份股名单，
     用交易所全部股票近似指数宽度；无涨跌停，用 ±5% 涨跌幅作"涨跌停"代理。
  3. turn（换手率）无流通股本数据置 0；amt（成交额）用 volume × 典型价 近似。
  4. --min-intraday-days 过滤的样本集合直接从分时索引文件重建（免重跑 preprocess），
     与 preprocess_data.py 的 keep_mask 完全一致（已验证：样本有 >=1 条 bar>=8 的记录则保留）。

输出（每个 split，默认只保存日频）：
  data/processeddata2_{exchange}/{split}/macro4idx.npy        (N, 60, 56)  日频
  周频 macro4idx_weekly.npy 仅当 --save-weekly-macro 开启时保存（默认不保存）

切分逻辑与 preprocess_data.py 完全一致（seed=42 分层 7:1:2 + 最终排序），
保证与已裁剪数据 100% 对齐（末尾会对比 daily.npy 行数）。

用法：
  D:/anaconda3/envs/DL/python.exe preprocess_macro_us.py --exchange nasdaq
  D:/anaconda3/envs/DL/python.exe preprocess_macro_us.py --exchange nyse
  D:/anaconda3/envs/DL/python.exe preprocess_macro_us.py --exchange all     # 合并两所=美股全市场宽度
  D:/anaconda3/envs/DL/python.exe preprocess_macro_us.py --exchange nasdaq --save-weekly-macro
"""

import os
import random
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(PROJ, 'data', 'rawdata')
INDEX_DIR = os.path.join(RAW, 'index_us')
SEED = 42
N_DAYS = 60
N_WEEKS = 12
DAYS_PER_WEEK = 5

# 4 个指数：文件名, 中文名
INDICES = [
    ('SPX', '标普500'),
    ('NDQ', '纳指综合'),
    ('DJI', '道琼斯'),
    ('RUT', '罗素2000'),
]

N_FEAT_PER_INDEX = 14
N_INDICES = 4
N_FEAT_TOTAL = N_FEAT_PER_INDEX * N_INDICES  # 56

# 涨跌停代理阈值（美股无涨跌停，用 ±5% 涨跌幅近似"极端波动"宽度）
# 注意：股票文件 us_stocks_raw_{exch}.csv 的 pct_chg 是小数（0.05=5%），故此处为 0.05
LIMIT_PCT = 0.05

# 特征列索引（14 维）—— 与 A 股版一致
PRICE_IDX = [0, 1, 2, 3]        # open, high, low, close（÷窗口最后收盘）
PCT_CHG_IDX = 4
VOL_IDX = [5, 6]                # volume, amt（÷窗口均值，与个股成交量归一化一致）
TURN_AMP_IDX = [7, 8]           # turn, amplitude
RATIO_IDX = [9, 10, 11, 12, 13] # up_ratio, down_ratio, up_limit_ratio, down_limit_ratio, rel_advance_decline


def recompute_splits(labels):
    """完全复刻 preprocess_data.py 的分层切分（seed=42, 7:1:2, 最终 np.sort）。"""
    by_label = defaultdict(list)
    for i, l in enumerate(labels):
        by_label[int(l)].append(i)

    random.seed(SEED)
    tr, va, te = [], [], []
    for lbl in sorted(by_label):
        group = by_label[lbl]
        random.shuffle(group)
        n = len(group)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        tr.extend(group[:n_train])
        va.extend(group[n_train:n_train + n_val])
        te.extend(group[n_train + n_val:])

    return np.sort(np.array(tr)), np.sort(np.array(va)), np.sort(np.array(te))


def build_breadth(exchange):
    """按日统计交易所全部股票的宽度指标，返回 {date_str: (up, down, up_limit, down_limit, total)}。

    输入列: code,date,open,high,low,close,pct_chg,volume,turn,amplitude
    涨跌：pct_chg 正负；涨跌停代理：|pct_chg| >= 5%。
    """
    daily_file = os.path.join(RAW, 'day', f'us_stocks_raw_{exchange}.csv')
    print(f"  读取交易所日线统计宽度: {daily_file}")
    df = pd.read_csv(daily_file, usecols=['date', 'pct_chg'])
    g = df.groupby('date')['pct_chg']
    agg = g.agg(total='count',
                up=lambda s: int((s > 0).sum()),
                down=lambda s: int((s < 0).sum()),
                up_limit=lambda s: int((s >= LIMIT_PCT).sum()),
                down_limit=lambda s: int((s <= -LIMIT_PCT).sum()))
    breadth = {}
    for d, r in agg.iterrows():
        breadth[str(d)] = (r['up'], r['down'], r['up_limit'], r['down_limit'], r['total'])
    print(f"  覆盖 {len(breadth):,} 个交易日")
    # 抽几个日期做 sanity check
    sample_dates = list(breadth.keys())[:3] + list(breadth.keys())[-3:]
    for d in sample_dates:
        u, dn, ul, dl, tot = breadth[d]
        print(f"    {d}: total={tot} up={u} down={dn} up≥{LIMIT_PCT}%={ul} down≤-{LIMIT_PCT}%={dl}")
    return breadth


def load_index(fname, breadth):
    """加载单指数 CSV，构建 13 列原始矩阵（列序同 A 股 Excel 版）+ 日期数组。

    13 列: [open, high, low, close, volume, amt(volume×典型价), turn(=0), pct_chg,
            up_count, down_count, up_limit, down_limit, total_count]
    """
    path = os.path.join(INDEX_DIR, f'{fname}.csv')
    df = pd.read_csv(path)
    df = df.sort_values('date').reset_index(drop=True)
    T = len(df)
    raw = np.zeros((T, 13), dtype=np.float64)

    raw[:, 0] = df['open'].values
    raw[:, 1] = df['high'].values
    raw[:, 2] = df['low'].values
    raw[:, 3] = df['close'].values
    raw[:, 4] = df['volume'].values
    typical = (df['high'] + df['low'] + df['close']).values / 3.0
    raw[:, 5] = df['volume'].values * typical          # amt 代理 = 成交量 × 典型价
    raw[:, 6] = 0.0                                    # turn（无流通股本数据）
    raw[:, 7] = df['pct_chg'].fillna(0.0).values       # pct_chg（首行 fillna 0）

    # 宽度指标按日期对齐（缺失则 ffill）
    dates = df['date'].values
    b_up = np.zeros(T); b_down = np.zeros(T); b_ul = np.zeros(T); b_dl = np.zeros(T); b_tot = np.zeros(T)
    last = None
    for i, d in enumerate(dates):
        row = breadth.get(str(d))
        if row is None:
            row = last
        if row is not None:
            b_up[i], b_down[i], b_ul[i], b_dl[i], b_tot[i] = row
            last = row
    raw[:, 8] = b_up
    raw[:, 9] = b_down
    raw[:, 10] = b_ul
    raw[:, 11] = b_dl
    raw[:, 12] = b_tot

    return raw, pd.to_datetime(dates).values


def build_index_features(raw_vals):
    """从 13 列原始矩阵构建 14 维特征 (T, 14)，列序同 A 股版。

    输出: [open, high, low, close, pct_chg, volume, amt, turn, amplitude,
          up_ratio, down_ratio, up_limit_ratio, down_limit_ratio, rel_advance_decline]
    """
    T = raw_vals.shape[0]
    features = np.zeros((T, N_FEAT_PER_INDEX), dtype=np.float32)

    features[:, 0:4] = raw_vals[:, 0:4]      # OHLC
    features[:, 4] = raw_vals[:, 7] / 100.0  # pct_chg（CSV 存百分数，转小数与个股日线 pct_chg 一致）
    features[:, 5] = raw_vals[:, 4]          # volume
    features[:, 6] = raw_vals[:, 5]          # amt
    features[:, 7] = raw_vals[:, 6]          # turn

    with np.errstate(divide='ignore', invalid='ignore'):
        features[:, 8] = (raw_vals[:, 1] - raw_vals[:, 2]) / np.maximum(raw_vals[:, 2], 1e-8)  # amplitude（小数，与个股振幅一致）

    total = np.maximum(raw_vals[:, 12], 1e-8)
    features[:, 9] = raw_vals[:, 8] / total     # up_ratio
    features[:, 10] = raw_vals[:, 9] / total    # down_ratio
    features[:, 11] = raw_vals[:, 10] / total   # up_limit_ratio
    features[:, 12] = raw_vals[:, 11] / total   # down_limit_ratio
    features[:, 13] = (raw_vals[:, 8] - raw_vals[:, 9]) / total  # rel_advance_decline

    return features


def norm_window(w):
    """归一化单窗口 (60, 14)：价格÷最后收盘、volume/amt 除以窗口均值、其余保持原值。"""
    w = w.copy()
    last_close = w[-1, 3]
    if last_close > 0:
        w[:, PRICE_IDX] /= last_close
    for c in VOL_IDX:
        col = w[:, c]
        mu = col.mean()
        if mu > 1e-8:
            w[:, c] = col / mu   # 除以窗口均值，与个股成交量归一化一致（不再用 z-score）
    return w


def daily_to_weekly(daily_window):
    """日频窗口 (60, F) -> 周频 (12, F)，统计方式同 build_indices_new.py。

    向量化：reshape (12,5,F) 后 axis=1 取均值一次完成所有列；
    pct_chg(4) 用周累计涨跌幅覆盖（价格已归一化，同比例缩放下不变）。
    """
    F = daily_window.shape[1]
    grouped = daily_window.reshape(N_WEEKS, DAYS_PER_WEEK, F)
    weekly = grouped.mean(axis=1).astype(np.float32)   # (12, F) 全部列均值
    first = grouped[:, 0, 3]
    last = grouped[:, -1, 3]
    weekly[:, 4] = np.where(first > 0, (last - first) / np.maximum(first, 1e-8), 0.0)
    return weekly


def window_for(dates, vals, target):
    """取结束于 target（含）的最近 60 个交易日窗口；历史不足则前补首行。"""
    pos = np.searchsorted(dates, np.datetime64(target))
    if pos >= len(dates) or dates[pos] > np.datetime64(target):
        pos -= 1
    pos = int(np.clip(pos, N_DAYS - 1, len(dates) - 1))
    seg = vals[pos - N_DAYS + 1: pos + 1]
    if seg.shape[0] < N_DAYS:
        pad = np.repeat(seg[:1], N_DAYS - seg.shape[0], axis=0)
        seg = np.concatenate([pad, seg], axis=0)
    return seg


def reconstruct_keep(n_samples, intra_file):
    """从分时索引重建 preprocess_data.py 的 keep_mask。

    规则（与 preprocess_data.py Step 6.1 一致）：
      样本保留 <=> 存在 >=1 条记录，其 bar 数（line_end - line_start + 1）>= 8。
    已验证：重建结果与 --min-intraday-days 1 过滤后的 split 规模完全一致。
    """
    idx = pd.read_csv(intra_file,
                      usecols=['sample_idx', 'intraday_line_start', 'intraday_line_end'])
    n_bars = idx['intraday_line_end'] - idx['intraday_line_start'] + 1
    keep_idx = idx.loc[n_bars >= 8, 'sample_idx'].drop_duplicates().values
    keep = np.zeros(n_samples, dtype=bool)
    keep[keep_idx] = True
    return keep


def main():
    ap = argparse.ArgumentParser(description='US 4 指数宏观对齐到 processeddata2_{exchange}')
    ap.add_argument('--exchange', choices=['nasdaq', 'nyse', 'all'], required=True)
    ap.add_argument('--min-intraday-days', type=int, default=1,
                    help='重建 preprocess 的 keep_mask 过滤（须与 preprocess_data.py 一致）')
    ap.add_argument('--save-weekly-macro', action='store_true',
                    help='额外保存周频宏观 macro4idx_weekly.npy（默认不保存）')
    args = ap.parse_args()
    ex = args.exchange
    proc_dir = os.path.join(PROJ, 'data', f'processeddata2_{ex}')

    # --exchange all 前置检查：合并日线 / 标签 / 窗口索引必须已生成
    if ex == 'all':
        need = [os.path.join(RAW, 'day', 'us_stocks_raw_all.csv'),
                os.path.join(RAW, 'future_50d_close_all.csv'),
                os.path.join(RAW, 'label_window_index_new_all.csv')]
        missing = [f for f in need if not os.path.exists(f)]
        if missing:
            print("[错误] --exchange all 缺少以下输入文件:")
            for f in missing:
                print(f"  {f}")
            print("  请先依次运行: gen_us_labels.py --exchange all -> build_indices_new.py --exchange all")
            sys.exit(1)

    # ========== 1. 读标签 & 窗口索引（全局样本行序） ==========
    labels_df = pd.read_csv(os.path.join(RAW, f'future_50d_close_{ex}.csv'))
    labels = labels_df['label'].values.astype(np.int32)
    win = pd.read_csv(os.path.join(RAW, f'label_window_index_new_{ex}.csv'))
    assert len(win) == len(labels), f"label_window_index ({len(win)}) 与 labels ({len(labels)}) 不一致"
    sample_dates = pd.to_datetime(win['date'].values)
    n_samples = len(labels)
    print(f"[{ex}] 全局样本总数: {n_samples}")

    # ========== 2. 复现切分 ==========
    tr, va, te = recompute_splits(labels)
    split_map_raw = {'train': tr, 'val': va, 'test': te}
    print("\n复现切分（未过滤）:")
    for name, idx in split_map_raw.items():
        print(f"  {name}: {len(idx)}")

    # ========== 3. 交易所宽度指标 ==========
    print("\n构建交易所宽度指标（涨跌停代理 ±5%）...")
    breadth = build_breadth(ex)

    # ========== 4. 加载 4 指数原始数据 -> 14 维特征 ==========
    print("\n加载 4 指数并构建 14 维特征...")
    index_features = {}
    for fname, cname in INDICES:
        raw_vals, dates = load_index(fname, breadth)
        features = build_index_features(raw_vals)
        index_features[fname] = (features, dates)
        print(f"  {fname} {cname}: {features.shape}, 日期 {pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}")

    # ========== 5. 对每个样本取窗口并归一化 ==========
    print(f"\n对齐到 {n_samples} 个样本...")
    macro_all = np.zeros((n_samples, N_DAYS, N_FEAT_TOTAL), dtype=np.float32)
    macro_weekly_all = (np.zeros((n_samples, N_WEEKS, N_FEAT_TOTAL), dtype=np.float32)
                        if args.save_weekly_macro else None)
    for i in range(n_samples):
        target_date = sample_dates[i]
        offset = 0
        for fname, _ in INDICES:
            vals, dates = index_features[fname]
            w = norm_window(window_for(dates, vals, target_date))
            macro_all[i, :, offset:offset + N_FEAT_PER_INDEX] = w
            if args.save_weekly_macro:
                macro_weekly_all[i, :, offset:offset + N_FEAT_PER_INDEX] = daily_to_weekly(w)
            offset += N_FEAT_PER_INDEX
        if (i + 1) % 50000 == 0:
            print(f"  已处理 {i + 1}/{n_samples}")
    weekly_desc = f", 周频: {macro_weekly_all.shape}" if args.save_weekly_macro else "（周频未保存，--save-weekly-macro 关闭）"
    print(f"\nmacro4idx 日频: {macro_all.shape}{weekly_desc}")
    print(f"NaN: {int(np.isnan(macro_all).sum())}, Inf: {int(np.isinf(macro_all).sum())}")

    # ========== 6. 重建 keep_mask 并过滤 split ==========
    if args.min_intraday_days > 0:
        keep = reconstruct_keep(n_samples, os.path.join(RAW, f'intraday_sample_index_new_{ex}.csv'))
        n_keep = int(keep.sum())
        print(f"\nkeep_mask 重建: 保留 {n_keep:,} ({n_keep/n_samples*100:.1f}%), "
              f"剔除 {n_samples - n_keep:,}")
        tr = tr[keep[tr]]; va = va[keep[va]]; te = te[keep[te]]
    split_map = {'train': tr, 'val': va, 'test': te}
    for name, idx in split_map.items():
        print(f"  过滤后 {name}: {len(idx):,}")

    # ========== 7. 按 split 保存 ==========
    print(f"\n保存到 {proc_dir}/{{split}}/ ...")
    for sp, idx in split_map.items():
        out = os.path.join(proc_dir, sp, 'macro4idx.npy')
        np.save(out, macro_all[idx])
        print(f"  {sp}: 日频 {macro_all[idx].shape} -> {out}")
        if args.save_weekly_macro:
            out_w = os.path.join(proc_dir, sp, 'macro4idx_weekly.npy')
            np.save(out_w, macro_weekly_all[idx])
            print(f"      周频 {macro_weekly_all[idx].shape} -> {out_w}")

    # ========== 8. 校验与 daily.npy 对齐 ==========
    print("\n--- 校验与 daily.npy 对齐 ---")
    all_ok = True
    for sp, idx in split_map.items():
        daily_path = os.path.join(proc_dir, sp, 'daily.npy')
        if not os.path.exists(daily_path):
            print(f"  [WARN] {daily_path} 不存在，跳过")
            continue
        n_daily = np.load(daily_path, mmap_mode='r').shape[0]
        file_list = ['macro4idx.npy']
        if args.save_weekly_macro:
            file_list.append('macro4idx_weekly.npy')
        for fname in file_list:
            n_macro = np.load(os.path.join(proc_dir, sp, fname), mmap_mode='r').shape[0]
            ok = n_daily == n_macro
            all_ok &= ok
            print(f"  {'[OK]' if ok else '[FAIL]'} {sp}: daily {n_daily} = {fname} {n_macro}")
    print("\n全部对齐！" if all_ok else "\n[ERROR] 对齐失败，请检查。")

    # ========== 9. 数值分布检查 ==========
    print("\n--- 日频数值分布 ---")
    for sp in ['train', 'val', 'test']:
        arr = np.load(os.path.join(proc_dir, sp, 'macro4idx.npy'), mmap_mode='r')
        print(f"\n{sp}:")
        for idx_idx, (fname, cname) in enumerate(INDICES):
            s = idx_idx * N_FEAT_PER_INDEX
            chunk = arr[:, :, s:s + N_FEAT_PER_INDEX]
            print(f"  {fname} {cname}: mean={chunk.mean():.4f} std={chunk.std():.4f} "
                  f"min={chunk.min():.4f} max={chunk.max():.4f}")


if __name__ == '__main__':
    main()
