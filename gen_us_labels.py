#!/usr/bin/env python3
"""
从美股日线数据生成标签文件（复刻 A 股 future_50d_close.csv 的 schema 与规则）。

输入：data/rawdata/day/us_stocks_raw_{exchange}.csv
      列: code,date,open,high,low,close,pct_chg,volume,turn,amplitude（按 code,date 升序）
输出：data/rawdata/future_50d_close_{exchange}.csv
      列: code,date,future_max_close_50,future_min_close_50,future_max_return_50,
          future_min_return_50,future_max_idx_50,future_min_idx_50,label

标签规则（从旧 A 股文件破译，99.99% 复现）：
  未来 50 个交易日内：
    - 涨(2)：future_max_return_50 > +threshold
    - 跌(0)：future_min_return_50 <= -threshold
    - 两者都破阈值时：max 先出现(max_idx < min_idx) → 涨(2)，否则 → 跌(0)
    - 其余 → 震荡(1)

抽样：--step 每 N 个交易日取 1 个样本（默认 20），并跳过前 59 个交易日
      （保证 build_indices_new.py 有完整 60 天窗口）。

运行：
    python gen_us_labels.py --exchange nasdaq     # 单独：纳斯达克
    python gen_us_labels.py --exchange nyse       # 单独：纽交所
    python gen_us_labels.py --exchange all        # 合并两所=美股全市场（默认；先合并日线再出标签）
"""

import argparse
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rawdata')

EXCHANGES = ['nasdaq', 'nyse']
DEFAULT_THRESHOLD = 0.15   # 涨/跌阈值
DEFAULT_HORIZON = 50       # 未来窗口（交易日）
DEFAULT_STEP = 20          # 抽样间隔（交易日）
MIN_PRIOR = 59             # 起始样本索引：保证 60 天窗口（end_line - 59 >= 0）

OUT_COLS = ['code', 'date', 'future_max_close_50', 'future_min_close_50',
            'future_max_return_50', 'future_min_return_50',
            'future_max_idx_50', 'future_min_idx_50', 'label']


def compute_labels(close, threshold, horizon):
    """对单只股票的收盘价序列计算未来 window 标签。

    返回 (label, max_close, min_close, max_ret, min_ret, imax, imin)，
    每个数组长度 = 有完整未来窗口的样本数 = len(close) - horizon。
    """
    n = len(close)
    if n < horizon + 1:
        empty = np.empty(0)
        return (empty.astype(np.int8), empty.copy(), empty.copy(), empty.copy(),
                empty.copy(), empty.astype(np.int64), empty.astype(np.int64))

    win = sliding_window_view(close, horizon + 1)   # win[i] = close[i .. i+horizon]
    base = win[:, 0]                                # 当日收盘 close[i]
    future = win[:, 1:]                             # close[i+1 .. i+horizon]，shape (n-horizon, horizon)

    imax = future.argmax(1)                         # 0-based，在未来窗口内的位置
    imin = future.argmin(1)
    max_close = np.take_along_axis(future, imax[:, None], 1)[:, 0]
    min_close = np.take_along_axis(future, imin[:, None], 1)[:, 0]

    max_ret = (max_close - base) / base
    min_ret = (min_close - base) / base

    thr = threshold
    label = np.where(
        (max_ret > thr) & (min_ret <= -thr),
        np.where(imax < imin, 2, 0),
        np.where(max_ret > thr, 2, np.where(min_ret <= -thr, 0, 1)),
    ).astype(np.int8)

    return label, max_close, min_close, max_ret, min_ret, imax, imin


def build_exchange(exchange, threshold, horizon, step):
    """为一个交易所生成标签 CSV。"""
    in_path = os.path.join(BASE_DIR, 'day', f'us_stocks_raw_{exchange}.csv')
    out_path = os.path.join(BASE_DIR, f'future_50d_close_{exchange}.csv')

    df = pd.read_csv(in_path, usecols=['code', 'date', 'close'])
    codes = df['code'].values
    dates = df['date'].values
    close = df['close'].values
    n = len(codes)

    grp_start = np.where(np.r_[True, codes[1:] != codes[:-1]])[0]
    grp_end = np.append(grp_start[1:], n)

    frames = []
    for s, e in zip(grp_start, grp_end):
        c = close[s:e]
        label, max_c, min_c, max_r, min_r, imax, imin = compute_labels(c, threshold, horizon)
        n_lab = len(label)
        if n_lab == 0:
            continue

        keep = np.arange(MIN_PRIOR, n_lab, step)
        if len(keep) == 0:
            continue

        frames.append(pd.DataFrame({
            'code': codes[s],
            'date': dates[s:e][keep],
            'future_max_close_50': max_c[keep],
            'future_min_close_50': min_c[keep],
            'future_max_return_50': max_r[keep],
            'future_min_return_50': min_r[keep],
            'future_max_idx_50': imax[keep],
            'future_min_idx_50': imin[keep],
            'label': label[keep],
        }))

    result = pd.concat(frames, ignore_index=True)
    result = result[OUT_COLS]
    result.to_csv(out_path, index=False)

    dist = result['label'].value_counts().sort_index()
    print(f"[{exchange}] 输出 {len(result):,} 样本 -> {out_path}")
    print(f"  日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"  类别分布: " + ", ".join(
        f"c{k}:{v}({v/len(result)*100:.1f}%)" for k, v in dist.items()))
    print(f"  股票数: {result['code'].nunique()}")


def merge_daily_all():
    """合并 nasdaq + nyse 日线为 day/us_stocks_raw_all.csv（按 code, date 排序）。

    --exchange all 的第一步：生成"美股全市场"合并日线，
    供后续 gen_us_labels / build_indices_new / preprocess_data / preprocess_macro_us 共用。
    """
    out_path = os.path.join(BASE_DIR, 'day', 'us_stocks_raw_all.csv')
    frames = []
    for ex in EXCHANGES:
        p = os.path.join(BASE_DIR, 'day', f'us_stocks_raw_{ex}.csv')
        print(f"[all] 读取 {p} ...")
        frames.append(pd.read_csv(p))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(['code', 'date']).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"[all] 合并日线: {len(merged):,} 行, {merged['code'].nunique():,} 只股票 -> {out_path}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', choices=EXCHANGES + ['all'], default='all')
    parser.add_argument('--step', type=int, default=DEFAULT_STEP, help='抽样间隔（交易日）')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='涨/跌阈值')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, help='未来窗口（交易日）')
    args = parser.parse_args()

    if args.exchange == 'all':
        merge_daily_all()   # 先生成合并日线 us_stocks_raw_all.csv
    build_exchange(args.exchange, args.threshold, args.horizon, args.step)


if __name__ == '__main__':
    main()
