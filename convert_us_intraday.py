#!/usr/bin/env python3
"""
把美股分时数据（30分钟 bar，含盘前盘后）改造成标准格式，只保留常规时段。

输入：data/rawdata/intraday/*_30min.csv   列: DateTime,Open,High,Low,Close,Volume
处理：
  1. 只保留常规时段 bar（起始时间 09:30 ~ 15:30，每天最多 13 根），去掉盘前(04:00~09:30)、盘后(16:00~)
  2. 拆 DateTime → 日期 / 时间
  3. 按 process_intraday_features.py Step 2 重算三个衍生列：
     - 分时涨跌幅 = (当前收盘 − 昨日收盘) / 昨日收盘 × 100
     - 分时振幅   = (当日截至目前最高 − 最低) / 昨日收盘 × 100
     - 分时换手率 = 当日累计成交量 × 100 / 总股本 × 100   ← 美股无总股本数据，留空
输出：data/rawdata/intraday_std/<同名>.csv
      列: 日期,时间,开盘,最高,最低,收盘,成交量,分时涨跌幅,分时振幅,分时换手率

运行：python convert_us_intraday.py
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rawdata')
INTRA_DIR = os.path.join(BASE_DIR, 'intraday')
OUT_DIR = os.path.join(BASE_DIR, 'intraday_std')

MIN_DATE = '2016-01-01'
MAX_DATE = '2026-01-01'  # exclusive

OUT_COLS = ['日期', '时间', '开盘', '最高', '最低', '收盘', '成交量',
            '分时涨跌幅', '分时振幅', '分时换手率']


def process_file(fname):
    """读单个分时文件 → 常规时段标准格式 DataFrame，无数据返回 None"""
    fpath = os.path.join(INTRA_DIR, fname)
    try:
        df = pd.read_csv(fpath, usecols=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                         parse_dates=['DateTime'])
    except (ValueError, pd.errors.EmptyDataError):
        print(f"  跳过 {fname}: 列不匹配或空文件")
        return None
    if df.empty:
        return None

    df['日期'] = df['DateTime'].dt.strftime('%Y-%m-%d')
    df['时间'] = df['DateTime'].dt.strftime('%H:%M:%S')

    # 只留常规时段 09:30 ~ <16:00（起始时间），再去掉范围外日期
    df = df[(df['时间'] >= '09:30') & (df['时间'] < '16:00')]
    df = df[(df['日期'] >= MIN_DATE) & (df['日期'] < MAX_DATE)]
    if df.empty:
        return None

    df = df.sort_values('DateTime').reset_index(drop=True)

    # 昨收 = 前一交易日最后一根常规 bar 的收盘
    day_last_close = df.groupby('日期')['Close'].last()
    prev_close = day_last_close.shift(1)          # 按日期后移一天
    pc = df['日期'].map(prev_close).fillna(0.0)

    # 分时涨跌幅
    ret = np.where(pc > 0, (df['Close'] - pc) / pc * 100, np.nan)
    # 分时振幅：当日截至目前最高 / 最低（分组累计）
    day_high = df.groupby('日期')['High'].cummax()
    day_low = df.groupby('日期')['Low'].cummin()
    amp = np.where(pc > 0, (day_high - day_low) / pc * 100, np.nan)

    # 分时换手率：无总股本，全部留空
    out = pd.DataFrame({
        '日期': df['日期'],
        '时间': df['时间'],
        '开盘': df['Open'],
        '最高': df['High'],
        '最低': df['Low'],
        '收盘': df['Close'],
        '成交量': df['Volume'],
        '分时涨跌幅': pd.Series(ret, index=df.index).round(4),
        '分时振幅': pd.Series(amp, index=df.index).round(4),
        '分时换手率': '',
    }, index=df.index)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INTRA_DIR) if f.endswith('.csv'))
    print(f"共 {len(files)} 个分时文件")

    n_days_low_bars = 0     # 低于 8 根 bar 的天数
    n_warn_files = 0
    total_days = 0
    total_bars = 0

    for i, fname in enumerate(files, 1):
        out = process_file(fname)
        if out is None:
            continue

        # 统计低于 8 根的天
        per_day = out.groupby('日期').size()
        low_days = int((per_day < 8).sum())
        n_days_low_bars += low_days
        n_warn_files += (low_days > 0)
        total_days += len(per_day)
        total_bars += len(out)

        out[OUT_COLS].to_csv(os.path.join(OUT_DIR, fname), index=False)
        if i % 50 == 0:
            print(f"  进度 {i}/{len(files)}")

    print(f"\n完成：写出 {len(files)} 个标准格式文件到 {OUT_DIR}")
    print(f"总 bar 数: {total_bars:,}，总交易日: {total_days:,}")
    print(f"低于 8 根 bar/天 的天数: {n_days_low_bars:,}（涉及 {n_warn_files} 个文件）")
    print("说明：分时换手率因无总股本数据全部留空；涨跌幅/振幅首日无昨收也留空。")


if __name__ == '__main__':
    main()
