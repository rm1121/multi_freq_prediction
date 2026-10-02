#!/usr/bin/env python3
"""
下载 4 个美股指数日线数据（Yahoo Finance，经 yfinance）。

指数选择（对应 A 股版 preprocess_macro_new.py 的 中证全指/中证1000/沪深300/中证500）：
  ^GSPC  S&P 500            -> SPX.csv
  ^IXIC  NASDAQ Composite   -> NDQ.csv
  ^DJI   Dow Jones          -> DJI.csv
  ^RUT   Russell 2000       -> RUT.csv

输出：data/rawdata/index_us/{SPX,NDQ,DJI,RUT}.csv
  列: date,open,high,low,close,volume,pct_chg
  pct_chg = 当日涨跌幅（%，相对昨收）

样本日期范围 2016-03-31 ~ 2025-10-17，下载从 2015-01-01 起，留足 60 日 lookback 余量。

运行（需要网络 + yfinance，用 DL 环境）：
  D:/anaconda3/envs/DL/python.exe download_us_indices.py
"""

import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJ, 'data', 'rawdata', 'index_us')
START = '2015-01-01'
END = '2026-08-15'

# 指数映射: Yahoo 符号 -> (文件名, 中文名)
INDICES = [
    ('^GSPC', 'SPX', '标普500'),
    ('^IXIC', 'NDQ', '纳斯达克综合'),
    ('^DJI', 'DJI', '道琼斯'),
    ('^RUT', 'RUT', '罗素2000'),
]

N_RETRY = 3


def fetch(symbol, name):
    """下载单个指数，返回 date 升序的 DataFrame[date,open,high,low,close,volume,pct_chg]。"""
    for attempt in range(1, N_RETRY + 1):
        try:
            df = yf.Ticker(symbol).history(
                start=START, end=END, auto_adjust=False)
            if df is None or len(df) == 0:
                raise RuntimeError('empty data')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.sort_index()
            # 涨跌幅 %
            close = df['close'].values
            pct = np.full(len(close), np.nan, dtype=np.float64)
            pct[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
            df['pct_chg'] = pct
            df = df[df['close'].notna()]
            df = df[df['open'].notna()]
            df['date'] = df.index
            df = df.reset_index(drop=True)
            print(f"  {symbol:7s} {name}  OK  {len(df):,} 行  "
                  f"{df['date'].min().date()} ~ {df['date'].max().date()}")
            return df
        except Exception as e:
            print(f"  {symbol:7s} {name}  第{attempt}次失败: {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"下载 4 个美股指数 -> {OUT_DIR}")
    ok = 0
    for symbol, fname, name in INDICES:
        df = fetch(symbol, name)
        if df is None:
            print(f"  [ERROR] {symbol} 下载失败，跳过")
            continue
        out = os.path.join(OUT_DIR, f'{fname}.csv')
        df.to_csv(out, index=False)
        print(f"  -> {out}")
        ok += 1
        time.sleep(1)  # 避免触发 Yahoo 限流

    print(f"\n完成: {ok}/{len(INDICES)} 个指数下载成功")


if __name__ == '__main__':
    main()
