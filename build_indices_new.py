#!/usr/bin/env python3
"""
为美股生成窗口索引 / 周频特征 / 分时对齐索引（按交易所参数化，美股 8 特征版本）。

产出（文件名带 _{exchange} 后缀）:
  - label_window_index_new_{exchange}.csv   (code, date, start_line, end_line)
  - weekly_features_new_{exchange}.csv      (sample_idx, week_idx, open_avg, ...) 8 列
  - intraday_sample_index_new_{exchange}.csv (sample_idx, day_idx, code, date, intraday_file, ...)

运行：
    python build_indices_new.py --exchange nasdaq     # 单独：纳斯达克
    python build_indices_new.py --exchange nyse       # 单独：纽交所
    python build_indices_new.py --exchange all        # 合并两所（前置：gen_us_labels.py --exchange all）

前置：convert_us_intraday.py 已生成 data/rawdata/intraday_std/{TICKER}_30min.csv。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rawdata')
INTRA_STD_DIR = os.path.join(RAW_DIR, 'intraday_std')

N_DAYS = 60
N_WEEKS = 12
DAYS_PER_WEEK = 5

# 美股日线 8 特征（无 amt）
DAILY_COLS = ['open', 'high', 'low', 'close', 'pct_chg', 'volume', 'turn', 'amplitude']
# 周频 8 特征（无 amt_avg）
WEEKLY_COLS = ['open_avg', 'high_avg', 'low_avg', 'close_avg', 'volume_avg',
               'cum_return', 'turn_avg', 'amplitude_avg']


def get_paths(exchange):
    return {
        'label': f'{RAW_DIR}/future_50d_close_{exchange}.csv',
        'daily': f'{RAW_DIR}/day/us_stocks_raw_{exchange}.csv',
        'out_label_win': f'{RAW_DIR}/label_window_index_new_{exchange}.csv',
        'out_weekly': f'{RAW_DIR}/week/weekly_features_new_{exchange}.csv',
        'out_intra': f'{RAW_DIR}/intraday_sample_index_new_{exchange}.csv',
    }


def build_daily_lookup(daily_file):
    """扫描日线数据，构建 (code, date) -> 0-based row index 映射。"""
    print("[1/5] 构建日线 (code, date) -> row_index 映射...")
    lookup = {}
    chunk_size = 200000
    reader = pd.read_csv(daily_file, usecols=['code', 'date'], chunksize=chunk_size)
    offset = 0
    for chunk in tqdm(reader, desc="  读取日线"):
        for _, row in chunk.iterrows():
            lookup[(row['code'], row['date'])] = offset
            offset += 1
    print(f"  映射条目数: {len(lookup):,}")
    return lookup


def generate_label_window_index(lookup, label_file, out_path):
    """根据标签文件的 (code, date) 生成窗口索引。"""
    print("\n[2/5] 生成 label_window_index_new_*.csv...")
    labels = pd.read_csv(label_file, usecols=['code', 'date'])
    rows = []
    skipped = 0
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="  匹配窗口"):
        key = (row['code'], row['date'])
        end_idx = lookup.get(key)
        if end_idx is None:
            skipped += 1
            continue
        start_idx = end_idx - (N_DAYS - 1)
        if start_idx < 0:
            skipped += 1
            continue
        rows.append({
            'code': row['code'],
            'date': row['date'],
            'start_line': start_idx + 1,   # 1-based 文件行号
            'end_line': end_idx + 1,       # 1-based (inclusive)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  有效窗口: {len(df)}, 跳过: {skipped}")
    return df


def generate_weekly_features(label_win_df, daily_file, out_path):
    """从日线数据计算周频特征（8 列，无 amt_avg）。"""
    print("\n[3/5] 生成 weekly_features_new_*.csv...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily_df = pd.read_csv(daily_file, usecols=DAILY_COLS)
    daily_raw = daily_df.values.astype(np.float32)
    del daily_df
    print(f"  日线数据: {daily_raw.shape}")

    records = []
    for idx, win_row in tqdm(label_win_df.iterrows(), total=len(label_win_df),
                             desc="  计算周特征"):
        sample_idx = idx
        start = int(win_row['start_line']) - 1
        end = int(win_row['end_line'])      # 1-based inclusive，共 60 天

        day_seq = daily_raw[start:end]      # (60, 8)

        for w in range(N_WEEKS):
            w_start = w * DAYS_PER_WEEK
            w_end = w_start + DAYS_PER_WEEK
            week_data = day_seq[w_start:w_end]  # (5, 8)

            open_avg = week_data[:, 0].mean()
            high_avg = week_data[:, 1].mean()
            low_avg = week_data[:, 2].mean()
            close_avg = week_data[:, 3].mean()
            volume_avg = week_data[:, 5].mean()

            first_close = week_data[0, 3]
            last_close = week_data[-1, 3]
            cum_return = ((last_close - first_close) / first_close) if first_close > 0 else 0.0

            turn_avg = week_data[:, 6].mean()
            amplitude_avg = week_data[:, 7].mean()

            records.append({
                'sample_idx': sample_idx,
                'week_idx': w,
                'open_avg': open_avg,
                'high_avg': high_avg,
                'low_avg': low_avg,
                'close_avg': close_avg,
                'volume_avg': volume_avg,
                'cum_return': cum_return,
                'turn_avg': turn_avg,
                'amplitude_avg': amplitude_avg,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  周频数据: {len(df)} 行 (={len(label_win_df)} 样本 x 12 周)")
    return df


def build_intraday_lookup(daily_file):
    """扫描 intraday_std/ 中属于本交易所的股票，构建 (code, date) -> (file, line_start, line_end)。"""
    print("\n[4/5] 构建分时 (code, date) -> (文件, 行范围) 映射...")
    if not os.path.isdir(INTRA_STD_DIR):
        print(f"  [警告] 未找到 {INTRA_STD_DIR}，请先运行 convert_us_intraday.py")
        return {}

    codes = pd.read_csv(daily_file, usecols=['code'])['code'].unique()
    lookup = {}
    n_files = 0
    for code in tqdm(codes, desc="  扫描分时文件"):
        fname = f'{code}_30min.csv'
        fpath = os.path.join(INTRA_STD_DIR, fname)
        if not os.path.exists(fpath):
            continue
        n_files += 1
        try:
            dates = pd.read_csv(fpath, usecols=['日期']).values[:, 0]
        except Exception as e:
            print(f"  [警告] 读取 {fname} 失败: {e}")
            continue
        if len(dates) == 0:
            continue
        starts = np.where(np.r_[True, dates[1:] != dates[:-1]])[0]
        ends = np.append(starts[1:] - 1, len(dates) - 1)
        for k in range(len(starts)):
            d = dates[starts[k]]
            lookup[(str(code), str(d))] = (fname, int(starts[k]), int(ends[k]))

    print(f"  分时文件数: {n_files}/{len(codes)}，唯一 (code, date) 映射数: {len(lookup):,}")
    return lookup


def generate_intraday_index(label_win_df, intra_lookup, daily_file, out_path):
    """为每个样本 x 60 天生成分时索引。"""
    print("\n[5/5] 生成 intraday_sample_index_new_*.csv...")
    daily_pairs = pd.read_csv(daily_file, usecols=['code', 'date']).values

    records = []
    skipped_dates = 0
    for idx, win_row in tqdm(label_win_df.iterrows(), total=len(label_win_df),
                             desc="  生成分时索引"):
        sample_idx = idx
        start = int(win_row['start_line']) - 1

        for day_idx in range(N_DAYS):
            row_pos = start + day_idx
            code = str(daily_pairs[row_pos, 0])
            date = str(daily_pairs[row_pos, 1])
            info = intra_lookup.get((code, date))
            if info is None:
                skipped_dates += 1
                continue
            fname, ls, le = info
            records.append({
                'sample_idx': sample_idx,
                'day_idx': day_idx,
                'code': code,
                'date': date,
                'intraday_file': fname,
                'intraday_line_start': ls,
                'intraday_line_end': le,
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  分时索引: {len(df)} 行, 缺失日期: {skipped_dates}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', choices=['nasdaq', 'nyse', 'all'], required=True)
    args = parser.parse_args()

    p = get_paths(args.exchange)
    if args.exchange == 'all' and not os.path.exists(p['daily']):
        print(f"[错误] 缺少合并日线 {p['daily']}")
        print("  请先运行: python gen_us_labels.py --exchange all（生成 us_stocks_raw_all.csv 与标签）。")
        sys.exit(1)
    if not os.path.exists(p['label']):
        print(f"[错误] 缺少标签文件 {p['label']}")
        print("  请先运行: python gen_us_labels.py --exchange <nasdaq|nyse|all>。")
        sys.exit(1)

    daily_lookup = build_daily_lookup(p['daily'])
    label_win_df = generate_label_window_index(daily_lookup, p['label'], p['out_label_win'])
    del daily_lookup

    generate_weekly_features(label_win_df, p['daily'], p['out_weekly'])
    intra_lookup = build_intraday_lookup(p['daily'])
    generate_intraday_index(label_win_df, intra_lookup, p['daily'], p['out_intra'])

    print("\n" + "=" * 60)
    print("索引文件生成完毕:")
    print(f"  {p['out_label_win']}")
    print(f"  {p['out_weekly']}")
    print(f"  {p['out_intra']}")
    print(f"样本数: {len(label_win_df)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
