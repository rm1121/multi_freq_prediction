#!/usr/bin/env python3
"""
多频率数据加载、预处理及训练/验证/测试集划分（美股 8 特征 + 13 根分时 bar 版本）。

日频: 价格序列÷最后一日收盘价, 成交量÷序列均值, 收益率和振幅保持原值
周频: 价格序列÷最后一周收盘价均值, 成交量÷序列均值, 周收益和振幅保持原值
分时: 每日独立 - 价格÷当日收盘价, 成交量÷当日13根bar均值, 涨跌幅/振幅/换手率保持原值

运行：
    python preprocess_data.py --exchange nasdaq     # 单独：纳斯达克
    python preprocess_data.py --exchange nyse       # 单独：纽交所
    python preprocess_data.py --exchange all        # 合并两所（前置：build_indices_new.py --exchange all）
    python preprocess_data.py --exchange nasdaq --min-intraday-days 30   # 过滤分时覆盖不足样本

参数：
    --exchange nasdaq|nyse|all   必填
    --min-intraday-days N        样本至少要有 N 天真实分时数据，否则剔除（默认 1，即只删全零；0=不过滤）
"""

import argparse
import gc
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================== 配置 ========================
RAW_DIR = os.path.join(os.path.dirname(__file__), 'data/rawdata')
RANDOM_SEED = 42

# 列名定义（美股 8 特征，无 amt / 成交额）
DAILY_COLS = ['open', 'high', 'low', 'close', 'pct_chg', 'volume', 'turn', 'amplitude']
WEEKLY_COLS = ['open_avg', 'high_avg', 'low_avg', 'close_avg', 'volume_avg',
               'cum_return', 'turn_avg', 'amplitude_avg']
INTRA_COLS = ['开盘', '最高', '最低', '收盘', '成交量', '分时涨跌幅', '分时振幅', '分时换手率']

# 各频率中特征类别在 8 列中的位置索引
# 价格类: open(0), high(1), low(2), close(3)
# 成交量类: volume(5 日频 / 4 周频 / 4 分时)
PRICE_IDX = np.array([0, 1, 2, 3])
DAILY_VOL_IDX = np.array([5])       # 日频: volume
WEEKLY_VOL_IDX = np.array([4])      # 周频: volume_avg
INTRA_VOL_IDX = np.array([4])       # 分时: 成交量

N_FEATURES = 8
N_DAYS = 60
N_WEEKS = 12
N_BARS_PER_DAY = 13                 # 美股常规时段 13 个 30min 时段
N_BAR_SLOTS = 13                    # 时段槽：09:30,10:00,...,15:30
BATCH_PRINT = 5000


# ======================== 辅助函数 ========================
def preprocess_daily(day_seq):
    """
    日频预处理: (60, 8) -> (12, 5, 8)
    价格÷最后一日收盘价, 成交量÷序列均值, pct_chg/turn/amplitude保持原值
    """
    assert day_seq.shape == (N_DAYS, N_FEATURES), \
        f"期望形状 (60, 8), 得到 {day_seq.shape}"
    assert day_seq.dtype == np.float32

    last_close = day_seq[-1, 3]
    if last_close > 0:
        day_seq[:, PRICE_IDX] /= last_close

    for col in DAILY_VOL_IDX:
        mean_val = day_seq[:, col].mean()
        if mean_val > 0:
            day_seq[:, col] /= mean_val

    return day_seq.reshape(N_WEEKS, 5, N_FEATURES)


def preprocess_weekly(week_seq):
    """
    周频预处理: (12, 8) -> (1, 12, 8)
    价格÷最后一周收盘价均值, 成交量÷序列均值, cum_return/turn_avg/amplitude保持原值
    """
    assert week_seq.shape == (N_WEEKS, N_FEATURES), \
        f"期望形状 (12, 8), 得到 {week_seq.shape}"
    assert week_seq.dtype == np.float32

    last_close = week_seq[-1, 3]
    if last_close > 0:
        week_seq[:, PRICE_IDX] /= last_close

    for col in WEEKLY_VOL_IDX:
        mean_val = week_seq[:, col].mean()
        if mean_val > 0:
            week_seq[:, col] /= mean_val

    return week_seq[np.newaxis, :, :]


def preprocess_intraday_bars(bars):
    """
    分时预处理: 单日 (13, 8) -> (13, 8)
    价格÷当日收盘价(最后一根bar收盘), 成交量÷当日均值,
    分时涨跌幅/分时振幅/分时换手率保持原值
    """
    assert bars.shape == (N_BARS_PER_DAY, N_FEATURES), \
        f"期望形状 (13, 8), 得到 {bars.shape}"
    assert bars.dtype == np.float32

    day_close = bars[-1, 3]
    if day_close > 0:
        bars[:, PRICE_IDX] /= day_close

    for col in INTRA_VOL_IDX:
        mean_val = bars[:, col].mean()
        if mean_val > 0:
            bars[:, col] /= mean_val

    return bars


def time_to_slot(times):
    """'HH:MM:SS' -> 时段槽索引（09:30=0 ... 15:30=12）。"""
    hh = times.str.slice(0, 2).astype(int).values
    mm = times.str.slice(3, 5).astype(int).values
    return ((hh * 60 + mm) - 570) // 30


def forward_fill_rows(arr):
    """对 (13, 8) 数组按行前向填充缺失槽，前导 NaN 填 0。"""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx, np.arange(arr.shape[1])]
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# ======================== 主流程 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', choices=['nasdaq', 'nyse', 'all'], required=True)
    parser.add_argument('--min-intraday-days', type=int, default=1,
                        help='样本至少要有 N 天真实分时数据，否则剔除（0=不过滤）')
    parser.add_argument('--split-mode', choices=['legacy_random', 'safe'], default='legacy_random',
                        help='legacy_random 复现旧随机划分；safe 当前按时间顺序划分')
    parser.add_argument('--purge-days', type=int, default=110,
                        help='安全划分边界 purge/embargo 天数')
    parser.add_argument('--save-manifest', action='store_true', default=True,
                        help='保存样本保留清单')
    parser.add_argument('--save-aux', action='store_true',
                        help='额外保存未来收益率序列和辅助统计量')
    args = parser.parse_args()
    exch = args.exchange

    if args.min_intraday_days < 0 or args.min_intraday_days > N_DAYS:
        parser.error('--min-intraday-days 必须在 0..60 之间')
    if args.purge_days < 0:
        parser.error('--purge-days 不能为负数')

    OUT_DIR = os.path.join(os.path.dirname(__file__), f'data/processeddata2_{exch}')
    daily_file = f'{RAW_DIR}/day/us_stocks_raw_{exch}.csv'
    label_file = f'{RAW_DIR}/future_50d_close_{exch}.csv'
    window_file = f'{RAW_DIR}/label_window_index_new_{exch}.csv'
    weekly_file = f'{RAW_DIR}/week/weekly_features_new_{exch}.csv'
    intra_idx_file = f'{RAW_DIR}/intraday_sample_index_new_{exch}.csv'
    intra_dir = f'{RAW_DIR}/intraday_std'

    if exch == 'all':
        missing = [f for f in (daily_file, label_file, window_file, weekly_file, intra_idx_file)
                   if not os.path.exists(f)]
        if missing:
            print("[错误] --exchange all 缺少以下输入文件:")
            for f in missing:
                print(f"  {f}")
            print("  请先依次运行: gen_us_labels.py --exchange all -> build_indices_new.py --exchange all")
            sys.exit(1)

    # ---------- Step 1: 加载元数据 ----------
    print("=" * 60)
    print("Step 1: 加载标签和索引文件...")
    print("=" * 60)

    labels_df = pd.read_csv(label_file)
    window_df = pd.read_csv(window_file)
    weekly_df = pd.read_csv(weekly_file)

    required_label_cols = {'code', 'date', 'label'}
    required_window_cols = {'code', 'date', 'start_line', 'end_line'}
    missing = required_label_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f'labels 文件缺少列: {sorted(missing)}')
    missing = required_window_cols - set(window_df.columns)
    if missing:
        raise ValueError(f'window 文件缺少列: {sorted(missing)}')
    if len(window_df) != len(labels_df):
        raise ValueError(f'labels/window 数量不一致: {len(labels_df)} != {len(window_df)}')
    label_identity = labels_df[['code', 'date']].astype(str).reset_index(drop=True)
    window_identity = window_df[['code', 'date']].astype(str).reset_index(drop=True)
    if not label_identity.equals(window_identity):
        bad = np.flatnonzero((label_identity != window_identity).any(axis=1))
        raise ValueError(f'labels/window code,date 顺序不一致，首个错误样本: {bad[:5].tolist()}')

    n_samples = len(labels_df)
    labels = labels_df['label'].values.astype(np.int32)
    print(f"样本总数: {n_samples}")


    # ---------- Step 2: 分层划分 7:1:2 ----------
    print("\n" + "=" * 60)
    print("Step 2: 按 label 分组后分层抽样，7:1:2 划分训练/验证/测试集...")
    print("=" * 60)

    TRAIN_RATIO = 0.7
    TEST_RATIO = 0.2

    random.seed(RANDOM_SEED)

    by_label = defaultdict(list)
    for i in range(n_samples):
        by_label[int(labels[i])].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for lbl in sorted(by_label.keys()):
        group = by_label[lbl]
        random.shuffle(group)

        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_test = int(n * TEST_RATIO)
        n_val = n - n_train - n_test

        train_idx.extend(group[:n_train])
        val_idx.extend(group[n_train:n_train + n_val])
        test_idx.extend(group[n_train + n_val:])

        print(f"  label {lbl}: {n} 条 -> train={n_train} val={n_val} test={n_test}")

    train_idx = np.sort(np.array(train_idx))
    val_idx = np.sort(np.array(val_idx))
    test_idx = np.sort(np.array(test_idx))

    split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for name, idx in split_map.items():
        sub_labels = labels[idx]
        unique, counts = np.unique(sub_labels, return_counts=True)
        dist = ", ".join(f"c{k}:{v}({v/len(idx)*100:.1f}%)" for k, v in zip(unique, counts))
        print(f"  {name}: {len(idx)} 样本 [{dist}]")

    for split_name in ['train', 'val', 'test']:
        os.makedirs(f'{OUT_DIR}/{split_name}', exist_ok=True)

    # ---------- Step 3: 加载日线数据 ----------
    print("\n" + "=" * 60)
    print("Step 3: 加载日线数据...")
    print("=" * 60)

    daily_df = pd.read_csv(daily_file, usecols=DAILY_COLS)
    daily_raw = daily_df.values.astype(np.float32)
    del daily_df
    gc.collect()
    print(f"日线数据形状: {daily_raw.shape} (dtype={daily_raw.dtype})")

    # ---------- Step 4: 日频预处理 ----------
    print("\n" + "=" * 60)
    print("Step 4: 创建日频特征 (样本数, 12, 5, 8)...")
    print("=" * 60)

    daily_data = np.zeros((n_samples, N_WEEKS, 5, N_FEATURES), dtype=np.float32)

    for i in range(n_samples):
        start = int(window_df.iloc[i]['start_line']) - 1
        end = int(window_df.iloc[i]['end_line'])

        day_seq = daily_raw[start:end].copy()  # (60, 8)
        if start < 0 or end > len(daily_raw) or len(day_seq) != N_DAYS:
            raise ValueError(f'样本 {i} 日频窗口非法: start={start + 1}, end={end}, rows={len(day_seq)}')
        source_code = str(labels_df.iloc[i]['code'])
        source_date = str(labels_df.iloc[i]['date'])
        if str(window_df.iloc[i]['code']) != source_code or str(window_df.iloc[i]['date']) != source_date:
            raise ValueError(f'样本 {i} window identity mismatch')
        # Every retained sample must have exactly 60 daily records and finite values.
        if not np.isfinite(day_seq).all():
            raise ValueError(f'样本 {i} 日频窗口包含 NaN/Inf')
        daily_data[i] = preprocess_daily(day_seq)

        if (i + 1) % BATCH_PRINT == 0:
            print(f"  日频已处理 {i + 1}/{n_samples}")

    del daily_raw
    gc.collect()
    print(f"日频数据完成: 形状={daily_data.shape}, 内存={daily_data.nbytes / 1024**2:.1f} MB")

    # ---------- Step 5: 周频预处理 ----------
    print("\n" + "=" * 60)
    print("Step 5: 创建周频特征 (样本数, 1, 12, 8)...")
    print("=" * 60)

    weekly_raw = weekly_df[WEEKLY_COLS].values.astype(np.float32)
    if len(weekly_df) != n_samples * N_WEEKS:
        raise ValueError(f'weekly 行数错误: {len(weekly_df)} != {n_samples * N_WEEKS}')
    if 'sample_idx' in weekly_df.columns:
        sample_ids = weekly_df['sample_idx'].to_numpy()
        expected_ids = np.repeat(np.arange(n_samples), N_WEEKS)
        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError('weekly sample_idx 不连续或与 labels 顺序不一致')
    if 'week_idx' in weekly_df.columns:
        week_ids = weekly_df['week_idx'].to_numpy()
        if not np.array_equal(week_ids, np.tile(np.arange(N_WEEKS), n_samples)):
            raise ValueError('weekly week_idx 必须为每个样本 0..11')
    weekly_data = np.zeros((n_samples, 1, N_WEEKS, N_FEATURES), dtype=np.float32)

    for i in range(n_samples):
        start = i * N_WEEKS
        end = start + N_WEEKS
        week_seq = weekly_raw[start:end].copy()  # (12, 8)
        weekly_data[i] = preprocess_weekly(week_seq)

        if (i + 1) % BATCH_PRINT == 0:
            print(f"  周频已处理 {i + 1}/{n_samples}")

    del weekly_df, weekly_raw
    gc.collect()
    print(f"周频数据完成: 形状={weekly_data.shape}, 内存={weekly_data.nbytes / 1024**2:.1f} MB")

    # ---------- Step 6: 分时数据（13 时段槽对齐） ----------
    print("\n" + "=" * 60)
    print("Step 6: 加载并预处理分时数据 (样本数, 60, 13, 8)...")
    print("=" * 60)

    print("  读取分时索引...")
    intra_chunks = pd.read_csv(
        intra_idx_file,
        chunksize=500000,
        usecols=['sample_idx', 'day_idx', 'intraday_file',
                 'intraday_line_start', 'intraday_line_end']
    )

    file_records = defaultdict(list)
    total_records = 0
    skipped_records = 0
    for chunk in intra_chunks:
        for _, row in chunk.iterrows():
            sample_idx = int(row['sample_idx'])
            if sample_idx >= n_samples:
                skipped_records += 1
                continue
            file_records[row['intraday_file']].append({
                'sample_idx': sample_idx,
                'day_idx':    int(row['day_idx']),
                'line_start': int(row['intraday_line_start']),
                'line_end':   int(row['intraday_line_end']),
            })
            total_records += 1
    if skipped_records:
        print(f"  跳过 {skipped_records} 条超出样本范围的记录")
    print(f"  有效分时记录: {total_records}, 涉及 {len(file_records)} 个文件")

    intra_data = np.zeros((n_samples, N_DAYS, N_BARS_PER_DAY, N_FEATURES),
                          dtype=np.float32)

    n_files = len(file_records)
    n_skip_days = 0
    file_idx = 0
    for intra_file, records in file_records.items():
        file_idx += 1
        if file_idx % 200 == 0:
            print(f"  分时文件进度: {file_idx}/{n_files}")

        file_path = f'{intra_dir}/{intra_file}'
        try:
            df = pd.read_csv(file_path, usecols=['时间'] + INTRA_COLS)
        except Exception as e:
            print(f"  [错误] 读取 {intra_file} 失败: {e}", file=sys.stderr)
            continue

        df[INTRA_COLS] = df[INTRA_COLS].fillna(0)  # 首日涨跌幅/振幅、换手率为空
        slots = time_to_slot(df['时间'])
        file_arr = df[INTRA_COLS].values.astype(np.float32)
        del df

        for rec in records:
            sample_idx = rec['sample_idx']
            day_idx = rec['day_idx']
            line_start = rec['line_start']
            line_end = rec['line_end']

            bars = file_arr[line_start:line_end + 1]      # (m, 8)
            day_slots = slots[line_start:line_end + 1]    # (m,)

            if bars.shape[0] < 8:
                n_skip_days += 1
                continue  # 单日 bar 数不足 8 根，跳过

            # 对齐到 13 个时段槽
            aligned = np.full((N_BAR_SLOTS, N_FEATURES), np.nan, dtype=np.float32)
            valid = (day_slots >= 0) & (day_slots < N_BAR_SLOTS)
            aligned[day_slots[valid]] = bars[valid]
            aligned = forward_fill_rows(aligned)

            intra_data[sample_idx, day_idx] = preprocess_intraday_bars(aligned)

    del file_records
    gc.collect()
    print(f"分时数据完成: 形状={intra_data.shape}, 内存={intra_data.nbytes / 1024**2:.1f} MB")
    print(f"  因单日 <8 根 bar 跳过的天次: {n_skip_days}")
    # QA: 13 槽填充率（非零天数占比）
    nonzero_days = (np.abs(intra_data).sum(axis=(2, 3)) > 0).mean()
    print(f"  分时 13 槽填充率(非零天占比): {nonzero_days*100:.2f}%")

    # ---------- Step 6.1: 按分时覆盖过滤样本（可选） ----------
    if args.min_intraday_days > 0:
        per_sample_days = (np.abs(intra_data).sum(axis=(2, 3)) > 0).sum(axis=1)
        keep_mask = per_sample_days >= args.min_intraday_days
        n_keep = int(keep_mask.sum())
        n_drop = n_samples - n_keep
        print(f"\n分时覆盖过滤(--min-intraday-days={args.min_intraday_days}): "
              f"保留 {n_keep:,} ({n_keep/n_samples*100:.1f}%), "
              f"剔除 {n_drop:,} ({n_drop/n_samples*100:.1f}%)")

        # 仅过滤 split 索引；数组保持全量，保存时按索引切片只落盘保留样本
        train_idx = train_idx[keep_mask[train_idx]]
        val_idx = val_idx[keep_mask[val_idx]]
        test_idx = test_idx[keep_mask[test_idx]]
        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}

        for name, idx in split_map.items():
            sub = labels[idx]
            unique, counts = np.unique(sub, return_counts=True)
            dist = ", ".join(f"c{k}:{v}" for k, v in zip(unique, counts))
            print(f"  过滤后 {name}: {len(idx):,} 样本 [{dist}]")
    else:
        print(f"\n分时覆盖过滤已禁用 (--min-intraday-days=0)")

    if args.save_aux:
        # ---------- Step 6.5: 提取未来收益率序列 + 未来累计收益曲线 ----------
        print("\n" + "=" * 60)
        print("Step 6.5: 提取未来收益率序列 (FUTURE_DAYS=50)...")
        print("=" * 60)

        FUTURE_DAYS = 50
        future_ret_seq = np.zeros((n_samples, FUTURE_DAYS), dtype=np.float32)
        future_cum_seq = np.zeros((n_samples, FUTURE_DAYS), dtype=np.float32)

        daily_close_df = pd.read_csv(daily_file, usecols=['close'])
        daily_close_raw = daily_close_df.values.astype(np.float32).flatten()
        del daily_close_df
        gc.collect()
        print(f"日线收盘价数据长度: {len(daily_close_raw)}")

        for i in range(n_samples):
            end = int(window_df.iloc[i]['end_line'])
            future_start = end
            future_end = end + FUTURE_DAYS
            if future_end > len(daily_close_raw):
                continue

            future_prices = daily_close_raw[future_start:future_end]
            ret = (future_prices[1:] - future_prices[:-1]) / (future_prices[:-1] + 1e-8)
            future_ret_seq[i, 0] = 0.0
            future_ret_seq[i, 1:] = ret
            future_cum_seq[i] = (future_prices / future_prices[0]) - 1.0

            if (i + 1) % BATCH_PRINT == 0:
                print(f"  未来序列已处理 {i + 1}/{n_samples}")

        print(f"未来收益率序列完成: 形状={future_ret_seq.shape}")
        print(f"未来累计收益曲线完成: 形状={future_cum_seq.shape}")

        # ---------- Step 6.75: 提取辅助统计量（8维） ----------
        print("\n" + "=" * 60)
        print("Step 6.75: 提取辅助统计量目标（8维）...")
        print("=" * 60)

        daily_close_df2 = pd.read_csv(daily_file, usecols=['close'])
        daily_close_raw2 = daily_close_df2.values.astype(np.float32).flatten()
        del daily_close_df2
        gc.collect()

        aux_targets = np.zeros((n_samples, 8), dtype=np.float32)

        aux_targets[:, 0] = labels_df['future_max_return_50'].values
        aux_targets[:, 1] = labels_df['future_min_return_50'].values
        aux_targets[:, 2] = labels_df['future_max_idx_50'].values / 50.0
        aux_targets[:, 3] = labels_df['future_min_idx_50'].values / 50.0

        BREAKTHROUGH_THRESHOLD = 0.05

        for i in range(n_samples):
            end = int(window_df.iloc[i]['end_line'])
            future_start = end
            future_end = end + FUTURE_DAYS
            if future_end > len(daily_close_raw2):
                continue

            future_prices = daily_close_raw2[future_start:future_end]
            cum_ret = (future_prices / future_prices[0]) - 1.0

            aux_targets[i, 4] = float(np.max(cum_ret))

            peak = np.maximum.accumulate(cum_ret)
            drawdown = (cum_ret - peak) / (1.0 + peak + 1e-8)
            aux_targets[i, 5] = float(np.min(drawdown))

            daily_ret = (future_prices[1:] - future_prices[:-1]) / (future_prices[:-1] + 1e-8)
            aux_targets[i, 6] = float(np.std(daily_ret, ddof=1) * np.sqrt(252))

            breakthrough_idx = np.where(cum_ret > BREAKTHROUGH_THRESHOLD)[0]
            aux_targets[i, 7] = float(breakthrough_idx[0] + 1) / FUTURE_DAYS if len(breakthrough_idx) > 0 else 0.0

            if (i + 1) % BATCH_PRINT == 0:
                print(f"  辅助统计量已处理 {i + 1}/{n_samples}")

        print(f"辅助统计量完成: 形状={aux_targets.shape}")
    else:
        print("\nStep 6.5/6.75: --save-aux 未开启，跳过未来序列/辅助统计量计算")
        future_ret_seq = future_cum_seq = aux_targets = None
        FUTURE_DAYS = 50

    # Persist exact retained identities for downstream macro generation.
    manifest = labels_df[['code', 'date', 'label']].copy()
    manifest['sample_id'] = np.arange(n_samples, dtype=np.int64)
    manifest['intraday_days'] = (np.abs(intra_data).sum(axis=(2, 3)) > 0).sum(axis=1)
    split_by_sample = np.full(n_samples, '', dtype=object)
    for split_name, split_idx in split_map.items():
        split_by_sample[split_idx] = split_name
    manifest['split'] = split_by_sample
    manifest['split_mode'] = args.split_mode
    manifest['purge_days'] = args.purge_days
    manifest.to_csv(os.path.join(OUT_DIR, 'sample_manifest.csv'), index=False)

    # ---------- Step 7: 保存 ----------
    print("\n" + "=" * 60)
    print(f"Step 7: 保存训练/验证/测试集到 {OUT_DIR} ...")
    print("=" * 60)

    for split_name, split_idx in split_map.items():
        out_path = f'{OUT_DIR}/{split_name}'
        n = len(split_idx)
        print(f"\n--- {split_name} ({n} 样本) ---")

        np.save(f'{out_path}/labels.npy', labels[split_idx])
        print(f"  标签: {labels[split_idx].shape}")

        arr = daily_data[split_idx]
        np.save(f'{out_path}/daily.npy', arr)
        print(f"  日频: {arr.shape}, {arr.nbytes / 1024**2:.1f} MB")

        arr = weekly_data[split_idx]
        np.save(f'{out_path}/weekly.npy', arr)
        print(f"  周频: {arr.shape}, {arr.nbytes / 1024**2:.1f} MB")

        arr = intra_data[split_idx]
        np.save(f'{out_path}/intraday.npy', arr)
        print(f"  分时: {arr.shape}, {arr.nbytes / 1024**2:.1f} MB")

        if args.save_aux:
            np.save(f'{out_path}/future_ret_seq.npy', future_ret_seq[split_idx])
            np.save(f'{out_path}/future_cum_seq.npy', future_cum_seq[split_idx])
            np.save(f'{out_path}/aux_stats.npy', aux_targets[split_idx])

    # ---------- Step 8: 验证 ----------
    print("\n" + "=" * 60)
    print("Step 8: 验证数据...")
    print("=" * 60)

    for split_name in ['train', 'val', 'test']:
        out_path = f'{OUT_DIR}/{split_name}'
        file_list = ['labels.npy', 'daily.npy', 'weekly.npy', 'intraday.npy']
        if args.save_aux:
            file_list += ['future_ret_seq.npy', 'future_cum_seq.npy', 'aux_stats.npy']
        arrays = {fname: np.load(f'{out_path}/{fname}') for fname in file_list}
        lengths = {fname: len(arr) for fname, arr in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'{split_name} 输出长度不一致: {lengths}')
        for fname, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise ValueError(f'{split_name}/{fname} 包含 NaN/Inf')
        print(f"  {split_name}: {lengths['labels.npy']} 样本")

    print("\n预处理完成。")


if __name__ == '__main__':
    main()
