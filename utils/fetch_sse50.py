"""
上证50成分股数据获取与预处理脚本。
用法：
    python utils/fetch_sse50.py
输出：
    ./dataset/SSE50/panel.csv  （拼接后的多股票面板数据）
    ./dataset/SSE50/constituents.json （每个调整周期的成分股列表，便于追溯）
    ./dataset/SSE50/fetch_log.txt （获取日志）
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import tushare as ts


# ============================================================
# 0. 全局配置
# ============================================================
DEFAULT_TOKEN = os.environ.get('TUSHARE_TOKEN', '')  # 建议用环境变量传入
INDEX_CODE = '000016.SH'          # 上证50
START_YEAR = 2020
END_YEAR = 2025
OUTPUT_DIR = './dataset/SSE50'
PANEL_PATH = os.path.join(OUTPUT_DIR, 'panel.csv')
CONSTITUENTS_PATH = os.path.join(OUTPUT_DIR, 'constituents.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'fetch_log.txt')

# 11个特征列 + stock_code + target
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume',
                'amount', 'pct_chg', 'turn', 'ma5', 'ma20']
EXTRA_COLS = ['stock_code']
TARGET_COL = 'target'

BLACKLIST = {'600340.SH'}

# ============================================================
# 1. 日志工具
# ============================================================
class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fp = open(path, 'w', encoding='utf-8')

    def log(self, msg):
        print(msg)
        self.fp.write(msg + '\n')
        self.fp.flush()

    def close(self):
        self.fp.close()


# ============================================================
# 2. 获取上证50成分股
# ============================================================
def get_sse50_constituents(pro, start_year, end_year, logger):
    """
    获取上证50每个半年度调整周期的成分股列表。
    返回 dict: {period_key: {'date': 'YYYYMMDD', 'codes': [...]}}
    """
    logger.log('\n' + '=' * 60)
    logger.log('[Step 1] 获取上证50成分股（按半年度调整周期）')
    logger.log('=' * 60)

    periods = {}
    for year in range(start_year, end_year + 1):
        for half, months in [('H1', ('01', '06')), ('H2', ('07', '12'))]:
            period_key = f'{year}{half}'
            start = f'{year}{months[0]}01'
            end = f'{year}{months[1]}28'
            try:
                df = pro.index_weight(
                    index_code=INDEX_CODE,
                    start_date=start,
                    end_date=end
                )
            except Exception as e:
                logger.log(f'  [WARN] {period_key} 查询失败: {e}')
                continue

            if df is None or df.empty:
                logger.log(f'  [WARN] {period_key} 无数据，跳过')
                continue

            latest_date = df['trade_date'].max()
            df_latest = df[df['trade_date'] == latest_date]
            codes = sorted(df_latest['con_code'].unique().tolist())
            periods[period_key] = {'date': latest_date, 'codes': codes}
            logger.log(f'  {period_key}  ({latest_date})  成分股 {len(codes)} 只')

    logger.log(f'\n共获取 {len(periods)} 个调整周期的成分股列表。')
    return periods


# ============================================================
# 3. 获取单只股票行情
# ============================================================
def fetch_stock_data(pro, stock_code, start_date, end_date):
    """
    获取单只股票前复权日线，计算技术指标和目标变量。
    缺失值处理：所有特征列统一用前向填充（ffill），首行若仍缺则用后向填充（bfill）。
    """
    # ---- 1. 前复权行情 ----
    try:
        df = ts.pro_bar(
            ts_code=stock_code,
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f'  [ERROR] {stock_code} pro_bar 异常: {repr(e)}')
        return None

    if df is None or df.empty:
        print(f'  [WARN] {stock_code} pro_bar 返回空')
        return None

    df = df.sort_values('trade_date').reset_index(drop=True)

    # ---- 2. 换手率 ----
    try:
        basic = pro.daily_basic(
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,turnover_rate'
        )
    except Exception as e:
        print(f'  [WARN] {stock_code} daily_basic 异常: {repr(e)}')
        basic = None

    # 换手率接口整体失败，直接跳过该股票（避免用假数据）
    if basic is None or basic.empty:
        print(f'  [SKIP] {stock_code} 无法获取换手率，跳过')
        return None

    df = df.merge(basic, on='trade_date', how='left')
    df = df.rename(columns={'turnover_rate': 'turn'})
    df = df.sort_values('trade_date').reset_index(drop=True)

    # ---- 3. 日期解析 ----
    df['date'] = pd.to_datetime(
        df['trade_date'].astype(str), format='%Y%m%d'
    ).dt.strftime('%Y-%m-%d')

    # ---- 4. 列重命名 ----
    df = df.rename(columns={'vol': 'volume'})

    # ---- 5. pct_chg 兜底 ----
    if 'pct_chg' not in df.columns:
        df['pct_chg'] = df['close'].pct_change() * 100

    # ---- 6. 前向填充所有可能缺失的特征列 ----
    # 价格/成交量/成交额/涨跌幅/换手率：按时间顺序 ffill，首行若仍缺用 bfill
    fill_cols = ['open', 'high', 'low', 'close',
                 'volume', 'amount', 'pct_chg', 'turn']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # ---- 7. 技术指标 ----
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    # ---- 目标：第 t+n 日的 n 日收益率 ----
    df['target'] = df['close'].shift(-10) / df['close'] - 1

    df['stock_code'] = stock_code

    # ---- 9. 保留列 ----
    keep_cols = ['date', 'stock_code', 'open', 'high', 'low', 'close',
                 'volume', 'amount', 'pct_chg', 'turn', 'ma5', 'ma20', 'target']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # ---- 10. 只丢 ma5/ma20/target 的边界 NaN 行 ----
    df = df.dropna(subset=['ma5', 'ma20', 'target']).reset_index(drop=True)

    return df


# ============================================================
# 4. 构建面板 CSV
# ============================================================
def build_panel(pro, periods, start_date, end_date, logger):
    logger.log('\n' + '=' * 60)
    logger.log('[Step 2] 获取个股行情并拼接面板数据')
    logger.log('=' * 60)

    # ---- 一次性获取所有A股基本信息，用于识别ST股 ----
    logger.log('获取股票基本信息（用于识别ST股）...')
    try:
        basic_all = pro.stock_basic(exchange='', list_status='L',
                                    fields='ts_code,name')
        # 同时获取已退市的
        basic_delisted = pro.stock_basic(exchange='', list_status='D',
                                         fields='ts_code,name')
        basic_all = pd.concat([basic_all, basic_delisted], ignore_index=True)
        # 标记ST股
        basic_all['is_st'] = basic_all['name'].str.contains('ST', na=False)
        st_codes = set(basic_all[basic_all['is_st']]['ts_code'].tolist())
        logger.log(f'  识别到 {len(st_codes)} 只ST/曾经ST股')
    except Exception as e:
        logger.log(f'  [WARN] 获取stock_basic失败: {e}，跳过ST过滤')
        st_codes = set()

    # 去重
    seen = set()
    ordered_codes = []
    for period_key in sorted(periods.keys()):
        for code in periods[period_key]['codes']:
            if code not in seen:
                seen.add(code)
                ordered_codes.append(code)

    logger.log(f'去重后待获取股票数: {len(ordered_codes)}')

    all_data = []
    failed = []
    skipped_st = []
    t0 = time.time()

    for i, code in enumerate(ordered_codes, 1):
        if code in st_codes:
            skipped_st.append(code)
            logger.log(f'  [{i}/{len(ordered_codes)}] {code}  跳过（ST股）')
            continue

        df = fetch_stock_data(pro, code, start_date, end_date)
        if code in BLACKLIST:
            logger.log(f'  [{i}/{len(ordered_codes)}] {code}  跳过（黑名单）')
            continue
        if df is None:
            failed.append((code, 'fetch 返回 None'))
            logger.log(f'  [{i}/{len(ordered_codes)}] {code}  fetch 返回 None')
            continue
        if len(df) < 100:
            failed.append((code, f'长度不足 ({len(df)})'))
            logger.log(f'  [{i}/{len(ordered_codes)}] {code}  长度不足 ({len(df)})')
            continue

        all_data.append(df)
        logger.log(f'  [{i}/{len(ordered_codes)}] {code}  {len(df)} 行  '
                   f'({df["date"].min()} ~ {df["date"].max()})')

    elapsed = time.time() - t0
    logger.log(f'\n行情获取完成，耗时 {elapsed:.1f}s，'
               f'成功 {len(all_data)} 只，ST跳过 {len(skipped_st)} 只，失败 {len(failed)} 只')
    if skipped_st:
        logger.log(f'ST股清单: {skipped_st}')
    if failed:
        logger.log('失败清单:')
        for code, reason in failed:
            logger.log(f'  {code}: {reason}')

    if not all_data:
        raise RuntimeError('没有任何股票数据获取成功。')

    panel = pd.concat(all_data, ignore_index=True)
    panel = panel[['date'] + EXTRA_COLS + FEATURE_COLS + [TARGET_COL]]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    logger.log(f'\n面板 CSV 已保存: {PANEL_PATH}')
    return panel


# ============================================================
# 5. 数据校验与终端报告
# ============================================================
def validate_and_report(panel, logger):
    logger.log('\n' + '=' * 60)
    logger.log('[Step 3] 数据校验与统计报告')
    logger.log('=' * 60)

    n_rows = len(panel)
    n_stocks = panel['stock_code'].nunique()
    date_min = panel['date'].min()
    date_max = panel['date'].max()

    logger.log(f'总行数            : {n_rows}')
    logger.log(f'股票数量          : {n_stocks}')
    logger.log(f'日期范围          : {date_min} ~ {date_max}')
    logger.log(f'列数              : {panel.shape[1]}')
    logger.log(f'列名              : {list(panel.columns)}')

    # 缺失值
    logger.log('\n--- 缺失值统计 ---')
    na_counts = panel.isna().sum()
    for col, cnt in na_counts.items():
        if cnt > 0:
            logger.log(f'  {col}: {cnt} 个缺失')
    if na_counts.sum() == 0:
        logger.log('  无缺失值')

    # 每只股票的样本数分布
    logger.log('\n--- 每只股票样本数 ---')
    counts = panel.groupby('stock_code').size()
    logger.log(f'  最小: {counts.min()}  最大: {counts.max()}  '
               f'中位数: {int(counts.median())}  平均: {counts.mean():.1f}')

    # pct_chg 分布检查
    logger.log('\n--- pct_chg 分布（收益率类，不做标准化） ---')
    pc = panel['pct_chg']
    logger.log(f'  min={pc.min():.3f}  max={pc.max():.3f}  '
               f'mean={pc.mean():.3f}  std={pc.std():.3f}')

    # 价格类特征分布
    logger.log('\n--- 价格类特征分布（做标准化） ---')
    for col in ['open', 'high', 'low', 'close']:
        v = panel[col]
        logger.log(f'  {col}: min={v.min():.3f}  max={v.max():.3f}  '
                   f'mean={v.mean():.3f}  std={v.std():.3f}')

    # 目标变量
    logger.log('\n--- target 分布 ---')
    tg = panel[TARGET_COL]
    logger.log(f'  min={tg.min():.3f}  max={tg.max():.3f}  '
               f'mean={tg.mean():.3f}  std={tg.std():.3f}')

    # 采样展示
    logger.log('\n--- 前 5 行样例 ---')
    logger.log(panel.head().to_string())
    logger.log('\n--- 随机 5 行样例 ---')
    logger.log(panel.sample(5, random_state=42).to_string())

    # 简单健康检查
    logger.log('\n--- 健康检查 ---')
    checks = {
        '行数 > 10000': n_rows > 10000,
        '股票数 >= 20': n_stocks >= 20,
        '无缺失值': na_counts.sum() == 0,
        'pct_chg 有方差': pc.std() > 0.1,
        'target 无 NaN': tg.isna().sum() == 0,
    }
    all_ok = True
    for k, v in checks.items():
        flag = 'PASS' if v else 'FAIL'
        logger.log(f'  [{flag}] {k}')
        all_ok = all_ok and v

    if all_ok:
        logger.log('\n数据校验全部通过，可以进行下一步训练。')
    else:
        logger.log('\n部分校验未通过，请检查数据。')
    return all_ok


# ============================================================
# 6. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=DEFAULT_TOKEN,                        help='Tushare Token，默认读环境变量 TUSHARE_TOKEN')
    parser.add_argument('--start_year', type=int, default=START_YEAR)
    parser.add_argument('--end_year', type=int, default=END_YEAR)
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.token:
        print('ERROR: 未提供 Tushare Token。'
              '请设置环境变量 TUSHARE_TOKEN 或使用 --token 传入。')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    logger = Logger(LOG_PATH)
    try:
        ts.set_token(args.token)
        pro = ts.pro_api()

        # Step 1: 成分股
        periods = get_sse50_constituents(pro, args.start_year, args.end_year, logger)
        with open(CONSTITUENTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(periods, f, ensure_ascii=False, indent=2)
        logger.log(f'成分股列表已保存: {CONSTITUENTS_PATH}')

        # Step 2: 行情 + 面板
        start_date = f'{args.start_year}0101'
        end_date = f'{args.end_year}1231'
        panel = build_panel(pro, periods, start_date, end_date, logger)

        # Step 3: 校验
        validate_and_report(panel, logger)

    finally:
        logger.close()


if __name__ == '__main__':
    main()