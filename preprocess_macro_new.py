#!/usr/bin/env python3
"""
把4指数宏观数据（中证全指/中证1000/沪深300/中证500）对齐到 processeddata2 的每个样本，
同时产出日频和周频两个版本：
  - data/processeddata2/{split}/macro4idx.npy          (N, 60, 56) 日频
  - data/processeddata2/{split}/macro4idx_weekly.npy   (N, 12, 56) 周频

对齐原理：
  - label_window_index_new.csv 按全局样本行序给出每样本窗口结束日 date。
  - 对每个样本取4个指数中"结束于该 date 的最近60个交易日"窗口。
  - 对每个指数计算14维特征：
      open/high/low/close: ÷窗口最后收盘价
      pct_chg/turn/amplitude: 保持原值
      volume/amt: zscore
      up_ratio/down_ratio/up_limit_ratio/down_limit_ratio/rel_advance_decline: 保持原值（已除以 total_count）
    4指数 concat → 56 维
  - 日频版: 直接输出 (N, 60, 56)
  - 周频版: 按5天一组做周频统计（参考股票周频处理方式）
    每组的 open/high/low/close/volume/amt/turn/amplitude/各比例指标取平均，
    pct_chg 计算周累计涨跌幅
    输出 (N, 12, 56)
  - 切分逻辑与 preprocess_data.py 完全一致（seed=42, 分层 7:1:2, 最终排序），
    保证与已裁剪数据 100% 对齐。

用法:
  python preprocess_macro_new.py
"""

import os
import sys
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

_PROJ = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_PROJ, 'data/rawdata')
OUT = os.path.join(_PROJ, 'data/processeddata2')
SEED = 42
N_DAYS = 60
N_WEEKS = 12
DAYS_PER_WEEK = 5

# Excel 中 4 个指数的名称（按文件中的出现顺序）
INDEX_NAMES = ['中证全指', '中证1000', '沪深300', '中证500']

# 每个指数的特征列索引（14维）
# open, high, low, close, pct_chg, volume, amt, turn, amplitude,
# up_ratio, down_ratio, up_limit_ratio, down_limit_ratio, rel_advance_decline
PRICE_IDX = [0, 1, 2, 3]        # open, high, low, close（÷最后收盘价）
PCT_CHG_IDX = 4                 # pct_chg（保持原值，周频算累计）
VOL_IDX = [5, 6]                # volume, amt（日频zscore，周频均值）
TURN_AMP_IDX = [7, 8]           # turn, amplitude（保持原值）
RATIO_IDX = [9, 10, 11, 12, 13] # up_ratio, down_ratio, up_limit_ratio, down_limit_ratio, rel_advance_decline（保持原值）
N_FEAT_PER_INDEX = 14
N_INDICES = 4
N_FEAT_TOTAL = N_FEAT_PER_INDEX * N_INDICES  # 56


def recompute_splits(labels):
    """
    完全复刻 preprocess_data.py 的分层切分逻辑：
      - random.seed(42)
      - 按 label 分组 -> shuffle -> 7:1:2 -> 最终 np.sort
    输入 labels: (N,) int32
    返回 (train_idx, val_idx, test_idx)，每个都是 np.ndarray
    """
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


def load_excel(path):
    """
    加载 Excel，返回 dict: {index_name: (values, dates)}
    values: (2430, 11) — 包含所有原始数值列
    dates: (2430,) — datetime64
    """
    df = pd.read_excel(path, header=None)
    # 第0行是中文表头，跳过；第0列是指数名称，第1列是日期
    # 第2~14列是数据列（13列）：open, high, low, close, volume, amt, turn, pct_chg,
    #   up_count, down_count, 涨停, 跌停, total_count
    # 取全部 13 列数值列
    VAL_COL_IDX = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    index_data = {}
    for name in INDEX_NAMES:
        # 找到该指数的起始行
        start_row = None
        for i in range(1, len(df)):
            val = df.iloc[i, 0]
            if pd.notna(val) and str(val).startswith(name):
                start_row = i
                break
        if start_row is None:
            raise ValueError(f"未找到指数 {name} 的起始行")

        # 该指数连续 2430 行
        end_row = start_row + 2430
        sub = df.iloc[start_row:end_row]

        dates = pd.to_datetime(sub.iloc[:, 1].values)
        vals = sub.iloc[:, VAL_COL_IDX].values.astype(np.float32)
        index_data[name] = (vals, dates.values)

    return index_data


def compute_amplitude(open_, high, low, close):
    """振幅 = (最高价 - 最低价) / 最低价 * 100%"""
    # 输入是标量或数组
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = (high - low) / np.maximum(low, 1e-8) * 100.0
    return amp


def build_index_features(raw_vals):
    """
    从原始 13 列构建 14 维特征数组 (T, 14)。
    原始列顺序: [open, high, low, close, volume, amt, turn, pct_chg,
                 up_count, down_count, up_limit, down_limit, total_count]
    输出列顺序: [open, high, low, close, pct_chg, volume, amt, turn, amplitude,
                 up_ratio, down_ratio, up_limit_ratio, down_limit_ratio,
                 rel_advance_decline]
    """
    T = raw_vals.shape[0]
    features = np.zeros((T, N_FEAT_PER_INDEX), dtype=np.float32)

    # 基础列直接赋值
    features[:, 0] = raw_vals[:, 0]   # open
    features[:, 1] = raw_vals[:, 1]   # high
    features[:, 2] = raw_vals[:, 2]   # low
    features[:, 3] = raw_vals[:, 3]   # close
    features[:, 4] = raw_vals[:, 7]   # pct_chg
    features[:, 5] = raw_vals[:, 4]   # volume
    features[:, 6] = raw_vals[:, 5]   # amt
    features[:, 7] = raw_vals[:, 6]   # turn

    # 计算振幅
    features[:, 8] = compute_amplitude(
        raw_vals[:, 0], raw_vals[:, 1], raw_vals[:, 2], raw_vals[:, 3])

    # 计算成份相关指标
    total = np.maximum(raw_vals[:, 12], 1e-8)  # total_count，防止除0
    features[:, 9] = raw_vals[:, 8] / total    # up_ratio
    features[:, 10] = raw_vals[:, 9] / total   # down_ratio
    features[:, 11] = raw_vals[:, 10] / total  # up_limit_ratio（涨停家数/成份总数）
    features[:, 12] = raw_vals[:, 11] / total  # down_limit_ratio（跌停家数/成份总数）
    features[:, 13] = (raw_vals[:, 8] - raw_vals[:, 9]) / total  # rel_advance_decline

    return features


def norm_window(w):
    """
    归一化单个窗口 (60, 14)：
      - 价格类(0-3) ÷ 窗口最后一个 close
      - pct_chg(4) 保持原值
      - volume/amt(5,6) zscore
      - turn/amplitude(7,8) 保持原值
      - up_ratio/down_ratio/up_limit_ratio/down_limit_ratio/rel_advance_decline(9-13) 保持原值
    """
    w = w.copy()
    last_close = w[-1, 3]
    if last_close > 0:
        w[:, PRICE_IDX] /= last_close

    # volume, amt zscore
    for c in VOL_IDX:
        col = w[:, c]
        mu = col.mean()
        std = col.std()
        if std > 1e-8:
            w[:, c] = (col - mu) / std
        else:
            w[:, c] = col - mu  # std=0 时仅去均值

    return w


def daily_to_weekly(daily_window):
    """
    将日频窗口 (60, F) 转为周频 (12, F)。
    每5天一组，参考 build_indices_new.py 的 weekly 统计方式：
      - open/high/low/close/volume/amt/turn/amplitude/各比例: 取均值
      - pct_chg: 周累计涨跌幅 = (last_close - first_close) / first_close
      - up_ratio/down_ratio/up_limit_ratio/down_limit_ratio/rel_advance_decline: 取均值
    """
    F = daily_window.shape[1]
    weekly = np.zeros((N_WEEKS, F), dtype=np.float32)
    for w in range(N_WEEKS):
        w_start = w * DAYS_PER_WEEK
        w_end = w_start + DAYS_PER_WEEK
        week_data = daily_window[w_start:w_end]  # (5, F)

        # 均值类特征
        weekly[w, 0] = week_data[:, 0].mean()   # open_avg
        weekly[w, 1] = week_data[:, 1].mean()   # high_avg
        weekly[w, 2] = week_data[:, 2].mean()   # low_avg
        weekly[w, 3] = week_data[:, 3].mean()   # close_avg
        # pct_chg: 周累计涨跌幅
        first_close = week_data[0, 3]
        last_close = week_data[-1, 3]
        if first_close > 0:
            weekly[w, 4] = (last_close - first_close) / first_close
        else:
            weekly[w, 4] = 0.0
        weekly[w, 5] = week_data[:, 5].mean()   # volume_avg
        weekly[w, 6] = week_data[:, 6].mean()   # amt_avg
        weekly[w, 7] = week_data[:, 7].mean()   # turn_avg
        weekly[w, 8] = week_data[:, 8].mean()   # amplitude_avg
        # 成份比例类：取均值
        for c in range(9, F):
            weekly[w, c] = week_data[:, c].mean()
    return weekly


def window_for(dates, vals, target):
    """
    取结束于 target（含）的最近 60 个交易日窗口。
    若历史不足 60 天则前补（重复首行）。
    vals: (T, 13)
    """
    pos = np.searchsorted(dates, np.datetime64(target))
    if pos >= len(dates) or dates[pos] > np.datetime64(target):
        pos -= 1
    pos = int(np.clip(pos, N_DAYS - 1, len(dates) - 1))
    seg = vals[pos - N_DAYS + 1: pos + 1]  # (60, F)
    if seg.shape[0] < N_DAYS:
        pad = np.repeat(seg[:1], N_DAYS - seg.shape[0], axis=0)
        seg = np.concatenate([pad, seg], axis=0)
    return seg


def main():
    ap = argparse.ArgumentParser(
        description="将4指数宏观数据（中证全指/中证1000/沪深300/中证500）对齐到 processeddata2 的每个样本")
    ap.add_argument('--processed-dir', default=OUT,
                    help='输出目录（已有 daily/weekly/intraday/labels.npy 的 split 目录）')
    ap.add_argument('--excel', default=os.path.join(RAW, 'index_300_500_1000_all.xlsx'),
                    help='4指数 Excel 文件路径')
    args = ap.parse_args()

    # ========== 1. 读标签 & 索引 ==========
    labels_df = pd.read_csv(os.path.join(RAW, 'future_50d_close.csv'))
    labels = labels_df['label'].values.astype(np.int32)
    win = pd.read_csv(os.path.join(RAW, 'label_window_index_new.csv'))
    assert len(win) == len(labels), \
        f"label_window_index ({len(win)}) 与 future_50d_close ({len(labels)}) 行数不一致"
    sample_dates = pd.to_datetime(win['date'].values)
    print(f"全局样本总数: {len(labels)}")

    # ========== 2. 复现切分 ==========
    tr, va, te = recompute_splits(labels)
    split_map = {'train': tr, 'val': va, 'test': te}
    for name, idx in split_map.items():
        print(f"  {name}: {len(idx)} 样本")

    # ========== 3. 加载 4 指数原始数据 ==========
    print("\n加载 4 指数 Excel...")
    index_data = load_excel(args.excel)
    for name, (vals, dates) in index_data.items():
        print(f"  {name}: {vals.shape}, 日期范围 {dates[0]} ~ {dates[-1]}")

    # ========== 4. 对每个指数构建 13 维特征 ==========
    print("\n构建 13 维特征（含振幅/成份指标）...")
    index_features = {}
    for name in INDEX_NAMES:
        raw_vals, dates = index_data[name]
        features = build_index_features(raw_vals)
        index_features[name] = (features, dates)
        print(f"  {name}: {features.shape}")

    # ========== 5. 对每个样本取窗口并归一化 ==========
    print(f"\n对齐到 {len(labels)} 个样本...")
    macro_all = np.zeros((len(labels), N_DAYS, N_FEAT_TOTAL), dtype=np.float32)
    macro_weekly_all = np.zeros((len(labels), N_WEEKS, N_FEAT_TOTAL), dtype=np.float32)
    for i in range(len(labels)):
        target_date = sample_dates[i]
        offset = 0
        for name in INDEX_NAMES:
            vals, dates = index_features[name]
            w = norm_window(window_for(dates, vals, target_date))
            macro_all[i, :, offset:offset + N_FEAT_PER_INDEX] = w
            # 周频：从归一化后的日频窗口做周频统计
            weekly_w = daily_to_weekly(w)
            macro_weekly_all[i, :, offset:offset + N_FEAT_PER_INDEX] = weekly_w
            offset += N_FEAT_PER_INDEX
        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i + 1}/{len(labels)}")

    print(f"\nmacro4idx 日频形状: {macro_all.shape}")
    print(f"macro4idx 周频形状: {macro_weekly_all.shape}")
    nan_cnt = int(np.isnan(macro_all).sum())
    inf_cnt = int(np.isinf(macro_all).sum())
    print(f"  NaN: {nan_cnt}, Inf: {inf_cnt}")
    if nan_cnt > 0 or inf_cnt > 0:
        print("  [WARN] 存在 NaN 或 Inf，请检查原始数据")

    # ========== 6. 按 split 保存 ==========
    print(f"\n保存到 {args.processed_dir}/{{split}}/ ...")
    for sp, idx in split_map.items():
        # 日频
        out = os.path.join(args.processed_dir, sp, 'macro4idx.npy')
        np.save(out, macro_all[idx])
        print(f"  {sp} 日频: {macro_all[idx].shape} -> {out}")
        # 周频
        out_week = os.path.join(args.processed_dir, sp, 'macro4idx_weekly.npy')
        np.save(out_week, macro_weekly_all[idx])
        print(f"  {sp} 周频: {macro_weekly_all[idx].shape} -> {out_week}")

    # ========== 7. 校验：样本数与 daily.npy 一致 ==========
    print("\n--- 校验 ---")
    all_ok = True
    for sp, idx in split_map.items():
        daily_path = os.path.join(args.processed_dir, sp, 'daily.npy')
        for fname in ['macro4idx.npy', 'macro4idx_weekly.npy']:
            macro_path = os.path.join(args.processed_dir, sp, fname)
            if not os.path.exists(daily_path):
                print(f"  [WARN] {daily_path} 不存在，跳过校验")
                all_ok = False
                continue
            n_daily = np.load(daily_path, mmap_mode='r').shape[0]
            n_macro = np.load(macro_path, mmap_mode='r').shape[0]
            ok = n_daily == n_macro
            status = "✓" if ok else "✗"
            print(f"  {status} {sp}: daily {n_daily} = {fname} {n_macro} ? {ok}")
            if not ok:
                all_ok = False
    if all_ok:
        print("\n全部校验通过！macro4idx.npy / macro4idx_weekly.npy 与 daily.npy 完全对齐。")
    else:
        print("\n[ERROR] 校验未通过，请检查。")

    # ========== 8. 检查数值分布 ==========
    for suffix, label in [('macro4idx.npy', '日频'), ('macro4idx_weekly.npy', '周频')]:
        print(f"\n--- {label} 数值分布检查 ---")
        for sp in ['train', 'val', 'test']:
            arr = np.load(os.path.join(args.processed_dir, sp, suffix), mmap_mode='r')
            print(f"\n{sp} ({label}):")
            for idx_idx in range(N_INDICES):
                start = idx_idx * N_FEAT_PER_INDEX
                end = start + N_FEAT_PER_INDEX
                chunk = arr[:, :, start:end]
                print(f"  指数{idx_idx} (feat {start}~{end-1}): "
                      f"mean={chunk.mean():.4f}, std={chunk.std():.4f}, "
                      f"min={chunk.min():.4f}, max={chunk.max():.4f}")


if __name__ == '__main__':
    main()
