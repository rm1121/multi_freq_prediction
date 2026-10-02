#!/usr/bin/env python3
"""
处理分时数据：
1. 提取各股票总股本（从日线数据 volume/turn 计算）
2. 将 30分钟/ 补充文件过滤2016-2025后拷入 intraday/
3. 对所有 intraday 文件过滤2016-2025日期，重新计算 分时涨跌幅、分时振幅、分时换手率
4. 重建样本-分时对齐索引 (intraday_sample_index.csv, intraday_date_lookup.csv)
"""

import csv
import os
from collections import defaultdict, OrderedDict
from statistics import median

# 配置路径（基于脚本所在目录，兼容任何机器）
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rawdata')
DAY_FILE = os.path.join(BASE_DIR, 'day/all_2016_2025_with_amplitude.csv')
INTRA_DIR = os.path.join(BASE_DIR, 'intraday')
SUPP_DIR = os.path.join(BASE_DIR, '30分钟')
LABEL_FILE = os.path.join(BASE_DIR, 'classification_labeled.csv')
WINDOW_INDEX_FILE = os.path.join(BASE_DIR, 'label_window_index.csv')

MIN_DATE = '2016-01-01'
MAX_DATE = '2026-01-01'  # exclusive upper bound

# ============================================================
# Step 0: 从日线数据推算每只股票的总股本
# ============================================================
def compute_total_shares():
    """从 volume/turn 推算总股本，取中位数"""
    print("=" * 60)
    print("Step 0: 从日线数据推算各股票总股本")
    
    code_shares = defaultdict(list)

    with open(DAY_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['code']
            try:
                vol = float(row['volume'])
                turn = float(row['turn']) if row['turn'] else 0.0
            except (ValueError, KeyError):
                continue
            if turn > 0 and vol > 0:
                shares = vol * 100.0 / turn
                code_shares[code].append(shares)

    shares_map = {}
    for code, lst in code_shares.items():
        shares_map[code] = median(lst) if lst else None

    # 保存
    shares_file = os.path.join(BASE_DIR, 'total_shares.csv')
    with open(shares_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'total_shares'])
        for code in sorted(shares_map.keys()):
            writer.writerow([code, f'{shares_map[code]:.0f}' if shares_map[code] else ''])

    print(f"  计算了 {len(shares_map)} 只股票的总股本，保存到 {shares_file}")
    return shares_map


# ============================================================
# Step 1: 拷贝30分钟补充文件并过滤日期范围
# ============================================================
def copy_supplementary_files():
    """把30分钟/ 的3个文件过滤后拷入 intraday/"""
    print("\n" + "=" * 60)
    print("Step 1: 处理补充文件 (30分钟/)")

    if not os.path.exists(SUPP_DIR):
        print("  目录不存在，跳过")
        return []

    copied = []
    for fname in sorted(os.listdir(SUPP_DIR)):
        if not fname.endswith('.csv'):
            continue
        src = os.path.join(SUPP_DIR, fname)
        dst = os.path.join(INTRA_DIR, fname)

        rows_in = 0
        rows_out = 0
        with open(src, 'r', encoding='utf-8-sig') as fin:
            reader = csv.reader(fin)
            header = next(reader)  # 日期,时间,开盘,最高,最低,收盘,成交量,成交额
            all_rows = []
            for row in reader:
                rows_in += 1
                date_str = row[0].strip()
                if MIN_DATE <= date_str < MAX_DATE:
                    all_rows.append(row)
                    rows_out += 1

        if all_rows:
            new_header = ['日期', '时间', '开盘', '最高', '最低', '收盘', '成交量', '成交额',
                          '分时涨跌幅', '分时振幅', '分时换手率']
            with open(dst, 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerow(new_header)
                for row in all_rows:
                    row = row[:8]
                    row.extend(['', '', ''])
                    writer.writerow(row)
            print(f"  {fname}: {rows_in} -> {rows_out} 行 (已过滤 {MIN_DATE}~2025-12-31)")
            copied.append(fname)
        else:
            print(f"  {fname}: {rows_in} 行全部在范围外，跳过")

    print(f"  共拷贝 {len(copied)} 个补充文件")
    return copied


# ============================================================
# Step 2: 对所有intraday文件重新计算分时特征
# ============================================================
def recompute_intraday_features(shares_map):
    """
    重算规则（与现有数据一致）：
    - 分时涨跌幅 = (当前收盘 - 昨日收盘) / 昨日收盘 × 100
    - 分时振幅   = (当日截至目前最高 - 当日截至目前最低) / 昨日收盘 × 100
    - 分时换手率 = 当日累计成交量 / 总股本 × 100
    每个交易日首日无前收盘，涨跌幅和振幅留空
    """
    print("\n" + "=" * 60)
    print("Step 2: 重新计算所有 intraday 文件的分时特征")

    files = sorted([f for f in os.listdir(INTRA_DIR) if f.endswith('.csv')])
    stats_no_shares = 0
    stats_filtered_out = 0

    for fname in files:
        fpath = os.path.join(INTRA_DIR, fname)
        code = fname.replace('.csv', '')  # sh600000
        stock_code = code[2:] + '.' + code[:2].upper()  # 600000.SH
        total_shares = shares_map.get(stock_code)

        # 读取所有行
        with open(fpath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            all_rows = list(reader)

        if not all_rows:
            continue

        # 过滤日期范围
        filtered = [r for r in all_rows if MIN_DATE <= r[0].strip() < MAX_DATE]
        stats_filtered_out += len(all_rows) - len(filtered)

        if not filtered:
            print(f"  WARNING: {fname} 过滤后无数据")
            continue

        # 按日期分组
        day_groups = OrderedDict()
        for row in filtered:
            d = row[0].strip()
            day_groups.setdefault(d, []).append(row)

        # 逐天计算
        prev_close = None
        output_rows = []

        for d in day_groups:
            bars = day_groups[d]
            day_high = -float('inf')
            day_low = float('inf')
            cum_vol = 0.0

            computed_bars = []
            for bar in bars:
                # 确保有8个原始字段
                raw = list(bar[:8])
                while len(raw) < 8:
                    raw.append('')

                try:
                    o = float(raw[2]) if raw[2] else 0.0
                    h = float(raw[3]) if raw[3] else 0.0
                    l_bar = float(raw[4]) if raw[4] else 0.0
                    c = float(raw[5]) if raw[5] else 0.0
                    vol = float(raw[6]) if raw[6] else 0.0
                except (ValueError, IndexError):
                    o = h = l_bar = c = vol = 0.0

                cum_vol += vol
                if h > 0:
                    day_high = max(day_high, h)
                if l_bar > 0:
                    day_low = min(day_low, l_bar)

                # 分时涨跌幅 = (当前收盘 - 昨日收盘) / 昨日收盘 × 100
                ret = ''
                if prev_close is not None and prev_close > 0:
                    ret = f'{(c - prev_close) / prev_close * 100:.4f}'

                # 分时振幅 = (当日截至目前最高 - 当日截至目前最低) / 昨日收盘 × 100
                amp = ''
                if prev_close is not None and prev_close > 0:
                    if day_high > 0 and day_low < float('inf') and day_low > 0:
                        amp = f'{(day_high - day_low) / prev_close * 100:.4f}'

                # 分时换手率 = 当日累计成交量(手) × 100 / 总股本(股) × 100
                turnover = ''
                if total_shares and total_shares > 0:
                    turnover = f'{cum_vol * 100 / total_shares * 100:.6f}'

                # 保留原始数据 + 计算值
                computed_bars.append(raw + [ret, amp, turnover])

            prev_close = float(bars[-1][5]) if bars[-1][5] else prev_close
            output_rows.extend(computed_bars)

        if not total_shares:
            stats_no_shares += 1

        # 写回文件
        out_header = ['日期', '时间', '开盘', '最高', '最低', '收盘', '成交量', '成交额',
                       '分时涨跌幅', '分时振幅', '分时换手率']
        with open(fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(out_header)
            writer.writerows(output_rows)

    print(f"  处理完成: {len(files)} 个文件")
    if stats_filtered_out:
        print(f"  共过滤掉 {stats_filtered_out} 行范围外数据")
    if stats_no_shares:
        print(f"  {stats_no_shares} 个股票无总股本数据(换手率将为空)")


# ============================================================
# Step 3: 重建样本-分时对齐索引
# ============================================================
def rebuild_sample_alignment():
    """
    生成:
    - intraday_sample_index.csv: 每个样本每天对应的分时文件行号
    - intraday_date_lookup.csv: 每个分时文件日期的行号范围
    - missing_intraday_samples.csv: 缺失分时数据的样本
    """
    print("\n" + "=" * 60)
    print("Step 3: 重建样本-分时对齐索引")

    # --- 3a: 构建分时日期查找表 ---
    print("  3a: 扫描分时文件，构建日期->行号映射...")
    code_date_map = defaultdict(dict)  # code -> {date: (line_start, line_end)}

    files = sorted([f for f in os.listdir(INTRA_DIR) if f.endswith('.csv')])
    for fname in files:
        fpath = os.path.join(INTRA_DIR, fname)
        code = fname.replace('.csv', '')

        with open(fpath, 'r') as f:
            header = f.readline()  # skip header
            prev_date = None
            line_start = 0
            for i, line in enumerate(f):
                date_str = line.split(',', 1)[0].strip()
                if date_str != prev_date:
                    if prev_date is not None:
                        code_date_map[code][prev_date] = (line_start, i - 1)
                    prev_date = date_str
                    line_start = i
            if prev_date is not None:
                code_date_map[code][prev_date] = (line_start, i)

    total_dates = sum(len(v) for v in code_date_map.values())
    print(f"    共 {len(files)} 个分时文件, {total_dates} 个交易日日期")

    # --- 3b: 读取样本标签 ---
    print("  3b: 读取样本标签...")
    samples = []
    with open(LABEL_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            samples.append((i, row['code'], row['date']))
    print(f"    共 {len(samples)} 个样本")

    # --- 3c: 读取日线数据和窗口索引 ---
    print("  3c: 读取日线数据和窗口索引...")
    # 读取日线全量数据
    day_data = []
    with open(DAY_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            day_data.append((row['code'], row['date']))

    # 读取窗口索引
    sample_windows = {}  # sample_idx -> [(code, date), ...]
    with open(WINDOW_INDEX_FILE, 'r') as f:
        reader = csv.DictReader(f)
        # columns: code,date,start_line,end_line - enumerated by row order
        for idx, row in enumerate(reader):
            sl = int(row['start_line'])
            el = int(row['end_line'])
            window = []
            for li in range(sl, el + 1):
                if li < len(day_data):
                    window.append(day_data[li])
            sample_windows[idx] = window

    # --- 3d: 生成对齐索引 ---
    print("  3d: 匹配样本窗口与分时数据...")
    index_rows = []
    missing_list = []

    for idx, code, t_date in samples:
        intra_code = code_to_intra_name(code)  # 600000.SH -> sh600000

        if idx not in sample_windows:
            missing_list.append([idx, code, t_date, 'N/A', 0, 60, 'NO_WINDOW_DATA'])
            continue

        window = sample_windows[idx]
        matched = 0
        missing = 0
        first_match = 'N/A'
        last_match = 'N/A'

        for day_idx, (d_code, d_date) in enumerate(window):
            if d_code != code:
                continue  # 窗口可能包含不同股票的行，只取当前code的
            if intra_code in code_date_map and d_date in code_date_map[intra_code]:
                ls, le = code_date_map[intra_code][d_date]
                index_rows.append([idx, day_idx, code, d_date, intra_code + '.csv', ls, le])
                matched += 1
                if first_match == 'N/A':
                    first_match = d_date
                last_match = d_date
            else:
                missing += 1

        if matched == 0:
            if intra_code not in code_date_map:
                reason = 'FILE_NOT_FOUND'
            else:
                # check if any dates overlap
                file_dates = set(code_date_map[intra_code].keys())
                win_dates = {d for c, d in window if c == code}
                if not file_dates & win_dates:
                    reason = f'NO_OVERLAP(file:{min(file_dates) if file_dates else "none"}~{max(file_dates) if file_dates else "none"})'
                else:
                    reason = 'PARTIAL_MISMATCH'
            missing_list.append([idx, code, t_date, first_match, matched, missing, reason])

    # 写入索引文件
    idx_file = os.path.join(BASE_DIR, 'intraday_sample_index.csv')
    with open(idx_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'day_idx', 'code', 'date',
                         'intraday_file', 'intraday_line_start', 'intraday_line_end'])
        writer.writerows(index_rows)
    print(f"    样本分时索引: {len(index_rows)} 行 -> {idx_file}")

    # 写入查找表
    lkp_file = os.path.join(BASE_DIR, 'intraday_date_lookup.csv')
    with open(lkp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['intraday_file', 'date', 'line_start', 'line_end'])
        for code in sorted(code_date_map.keys()):
            for date in sorted(code_date_map[code].keys()):
                ls, le = code_date_map[code][date]
                writer.writerow([code + '.csv', date, ls, le])
    print(f"    日期查找表: {total_dates} 行 -> {lkp_file}")

    # 写入缺失样本
    mis_file = os.path.join(BASE_DIR, 'missing_intraday_samples.csv')
    with open(mis_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'code', 't_date', 'window_start',
                         'matched_days', 'missing_days', 'intraday_range'])
        writer.writerows(missing_list)
    print(f"    缺失样本: {len(missing_list)} 个 -> {mis_file}")

    return len(missing_list)


def code_to_intra_name(code):
    """600000.SH -> sh600000, 000001.SZ -> sz000001"""
    parts = code.split('.')
    if len(parts) == 2:
        return parts[1].lower() + parts[0]
    return code.lower()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # Step 0
    shares_map = compute_total_shares()

    # Step 1
    copy_supplementary_files()

    # Step 2
    recompute_intraday_features(shares_map)

    # Step 3
    n_missing = rebuild_sample_alignment()

    print("\n" + "=" * 60)
    print(f"全部完成！缺失分时数据的样本数: {n_missing}")
