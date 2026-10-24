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

# 旧版静态黑名单已暂停自动应用：应迁移为带有效期、原因和来源的可审计规则。
BLACKLIST = set()

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
    """使用 Tushare index_weight 月度快照近似重建历史成分区间。

    注意：index_weight 是月度成分/权重数据，不提供精确纳入/剔除生效日。
    本函数将每个快照中的成分状态向前延续到下一个快照日期之前，属于月度快照近似，
    不应在论文中描述为交易所公告级别的精确有效区间。
    """
    logger.log('\n' + '=' * 60)
    logger.log('[Step 1] 获取上证50历史成分股（月度 index_weight 快照近似）')
    logger.log('=' * 60)
    start_date = f'{start_year}0101'
    end_date = f'{end_year}1231'

    # 按自然月查询，符合 Tushare 文档建议，并避免单次返回量过大。
    snapshots = {}
    month_starts = pd.date_range(
        pd.to_datetime(start_date, format='%Y%m%d').replace(day=1),
        pd.to_datetime(end_date, format='%Y%m%d').replace(day=1), freq='MS'
    )
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        lo = max(month_start.strftime('%Y%m%d'), start_date)
        hi = min(month_end.strftime('%Y%m%d'), end_date)
        try:
            weights = pro.index_weight(
                index_code=INDEX_CODE,
                start_date=lo,
                end_date=hi,
                fields='index_code,con_code,trade_date,weight'
            )
        except Exception as e:
            raise RuntimeError(
                f'Tushare index_weight 查询失败（{lo}—{hi}）。请检查 Token、接口权限和指数代码 {INDEX_CODE}。'
            ) from e
        if weights is None or weights.empty:
            logger.log(f'  [WARN] {lo}—{hi} 没有返回月度成分快照')
            continue
        required = {'con_code', 'trade_date', 'weight'}
        if not required.issubset(weights.columns):
            raise RuntimeError(f'index_weight 返回字段不完整: {sorted(required - set(weights.columns))}')
        weights = weights.copy()
        weights['trade_date'] = weights['trade_date'].astype(str).str.replace('-', '', regex=False)
        weights['con_code'] = weights['con_code'].astype(str)
        for trade_date, group in weights.groupby('trade_date'):
            codes = sorted(set(group['con_code'].dropna()))
            if codes:
                snapshots[trade_date] = codes

    if not snapshots:
        raise RuntimeError(
            f'Tushare index_weight 在 {start_date}—{end_date} 没有返回任何快照。'
            '请确认上证50指数代码、Token 和接口积分权限。'
        )

    snapshot_dates = sorted(snapshots)
    logger.log(f'获取到月度快照数: {len(snapshot_dates)}；首末快照: {snapshot_dates[0]}—{snapshot_dates[-1]}')
    logger.log('重要限制：成员状态按快照日期阶梯延续，非精确的指数调样生效日期。')

    # 每个快照状态延续到下一个快照日期前一天；最后一个快照延续到查询结束日。
    memberships = []
    for i, snap_date in enumerate(snapshot_dates):
        interval_start = max(start_date, snap_date)
        if i + 1 < len(snapshot_dates):
            next_date = pd.to_datetime(snapshot_dates[i + 1], format='%Y%m%d')
            interval_end = (next_date - pd.Timedelta(days=1)).strftime('%Y%m%d')
        else:
            interval_end = end_date
        interval_end = min(interval_end, end_date)
        if interval_start > interval_end:
            continue
        for code in snapshots[snap_date]:
            memberships.append({
                'stock_code': code,
                'in_date': None,
                'out_date': None,
                'effective_start': interval_start,
                'effective_end': interval_end,
                'snapshot_date': snap_date,
                'source': 'tushare.index_weight',
                'interval_method': 'monthly_snapshot_step_forward_approximation',
            })

    # 兼容旧的半年摘要结构；仅用于日志，实际资格仍由区间标记决定。
    periods = {}
    for year in range(start_year, end_year + 1):
        for half, month_start, month_end in [('H1', '0101', '0630'), ('H2', '0701', '1231')]:
            key = f'{year}{half}'
            lo, hi = f'{year}{month_start}', f'{year}{month_end}'
            codes = sorted({m['stock_code'] for m in memberships
                            if m['effective_start'] <= hi and m['effective_end'] >= lo})
            periods[key] = {'date': None, 'codes': codes}
            logger.log(f'  {key}: 区间内出现 {len(codes)} 只股票（月度快照近似）')

    logger.log(f'近似有效区间记录: {len(memberships)}；股票数: {len(set(m["stock_code"] for m in memberships))}')
    return {'memberships': memberships, 'periods': periods}


# ============================================================
# 3. 获取单只股票行情
# ============================================================
def fetch_stock_data(pro, stock_code, start_date, end_date):
    """
    获取单只股票前复权日线，计算技术指标、目标收益及目标结束日期。
    缺失值仅允许按时间前向填充；首端缺失保留并在后续丢弃，不使用未来观测回填。
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

    # ---- 6. 仅允许因果性前向填充；禁止 bfill 将未来值带回过去 ----
    fill_cols = ['open', 'high', 'low', 'close',
                 'volume', 'amount', 'pct_chg', 'turn']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ---- 7. 技术指标 ----
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    # ---- 目标：收盘后产生信号，下一有效交易观测开盘进场，第 10 个后续观测收盘退出 ----
    # date=t 是信号日；entry date=t+1 的 open 是可执行的下一交易日开盘价；
    # exit date=t+10 的 close 是持有期终点。禁止用信号日 close 假定同价成交。
    df['target_start_date'] = pd.to_datetime(df['date']).shift(-1).dt.strftime('%Y-%m-%d')
    df['target_end_date'] = pd.to_datetime(df['date']).shift(-10).dt.strftime('%Y-%m-%d')
    entry_price = df['open'].shift(-1)
    exit_price = df['close'].shift(-10)
    valid_prices = entry_price.notna() & exit_price.notna() & (entry_price > 0)
    df['target'] = np.where(valid_prices, exit_price / entry_price - 1.0, np.nan)

    df['stock_code'] = stock_code

    # ---- 9. 保留列 ----
    keep_cols = ['date', 'target_start_date', 'target_end_date', 'stock_code', 'open', 'high', 'low', 'close',
                 'volume', 'amount', 'pct_chg', 'turn', 'ma5', 'ma20', 'target']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # ---- 10. 删除无有效目标、技术指标或必要特征的行，不用未来值填补 ----
    required = ['ma5', 'ma20', 'target', 'target_start_date', 'target_end_date'] + [c for c in fill_cols if c in df.columns]
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


# ============================================================
# 4. 构建面板 CSV
# ============================================================
def get_historical_st_intervals(pro, stock_codes, start_date, end_date, logger):
    """通过历史证券名称变更记录识别 ST 区间；数据缺失不静默视为非 ST。"""
    records = []
    failed_codes = []
    for code in sorted(set(stock_codes)):
        try:
            hist = pro.namechange(
                ts_code=code,
                fields='ts_code,name,start_date,end_date,change_reason'
            )
        except Exception as e:
            logger.log(f'  [ERROR] {code} 历史名称查询失败: {e}')
            failed_codes.append(code)
            continue
        if hist is None:
            failed_codes.append(code)
            logger.log(f'  [WARN] {code} 历史名称接口返回 None，状态标记为未知')
            continue
        if hist.empty:
            # 接口成功返回空表表示没有可用的历史更名记录；与请求失败分开记录。
            logger.log(f'  [INFO] {code} 没有历史名称变更记录')
            continue
        hist = hist.copy()
        hist['name'] = hist['name'].fillna('').astype(str)
        hist['start_date'] = hist['start_date'].fillna('').astype(str).str.replace('-', '', regex=False)
        hist['end_date'] = hist['end_date'].fillna('').astype(str).str.replace('-', '', regex=False)
        for r in hist.itertuples(index=False):
            # 名称中含 ST 的历史区间；终止日期为空时按查询区间末日截断。
            if 'ST' not in str(r.name).upper():
                continue
            lo = str(r.start_date or start_date)
            hi = str(r.end_date or end_date)
            lo = max(lo, start_date) if lo else start_date
            hi = min(hi, end_date) if hi else end_date
            if lo <= hi:
                records.append({'stock_code': code, 'start_date': lo, 'end_date': hi,
                                'name': str(r.name), 'source': 'tushare.namechange'})
    if failed_codes:
        logger.log(f'历史名称状态查询不完整的股票数: {len(set(failed_codes))}；详见日志。')
    return records, sorted(set(failed_codes))


def build_panel(pro, universe, start_date, end_date, logger):
    logger.log('\n' + '=' * 60)
    logger.log('[Step 2] 获取行情并附加逐日历史交易资格（月度成分快照近似）')
    logger.log('=' * 60)

    memberships = universe['memberships']
    stock_codes = sorted({m['stock_code'] for m in memberships})
    st_intervals, unknown_status_codes = get_historical_st_intervals(
        pro, stock_codes, start_date, end_date, logger
    )
    # 退市状态只使用可追溯的实际退市日期，不从当前挂牌状态反推整个历史区间。
    try:
        delisted_basic = pro.stock_basic(
            exchange='', list_status='D', fields='ts_code,delist_date'
        )
    except Exception as e:
        raise RuntimeError('无法读取带 delist_date 的退市记录；为保证历史资格审计，停止生成面板。') from e
    if delisted_basic is None or 'ts_code' not in delisted_basic.columns or 'delist_date' not in delisted_basic.columns:
        raise RuntimeError('退市记录缺少 ts_code/delist_date 字段，停止生成面板。')
    delisted_basic = delisted_basic.copy()
    delisted_basic['delist_date'] = delisted_basic['delist_date'].fillna('').astype(str).str.replace('-', '', regex=False)
    delist_map = {r.ts_code: r.delist_date for r in delisted_basic.itertuples(index=False)
                  if r.ts_code in stock_codes and str(r.delist_date).strip()}
    logger.log(f'可追溯退市日期记录覆盖股票数: {len(delist_map)}')
    # 黑名单不再接受无元数据的裸代码列表；旧规则需提供有效期、原因与来源后再迁移。
    logger.log('研究者黑名单未自动应用：旧版 600340.SH 静态排除缺少有效期/依据，需补齐审计记录后再纳入。')

    all_data, failed = [], []
    t0 = time.time()
    for i, code in enumerate(stock_codes, 1):
        df = fetch_stock_data(pro, code, start_date, end_date)
        if df is None:
            failed.append((code, '行情或换手率数据获取失败'))
            logger.log(f'  [{i}/{len(stock_codes)}] {code} 获取失败')
            continue
        if len(df) < 100:
            failed.append((code, f'长度不足 ({len(df)})'))
            logger.log(f'  [{i}/{len(stock_codes)}] {code} 长度不足 ({len(df)})')
            continue

        df['date_key'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        df['is_constituent'] = False
        for m in memberships:
            if m['stock_code'] == code:
                df.loc[df['date_key'].between(m['effective_start'], m['effective_end']), 'is_constituent'] = True

        df['is_st'] = False
        for st in st_intervals:
            if st['stock_code'] == code:
                df.loc[df['date_key'].between(st['start_date'], st['end_date']), 'is_st'] = True
        df['historical_status_known'] = code not in unknown_status_codes
        df['is_blacklisted'] = False
        delist_date = delist_map.get(code, '')
        df['is_delisted'] = bool(delist_date) & (df['date_key'] >= delist_date) if delist_date else False
        df['is_eligible'] = (df['is_constituent'] & ~df['is_st'] & ~df['is_delisted'] &
                             ~df['is_blacklisted'] & df['historical_status_known'])
        # target_start_date 为下一有效交易观测，target_end_date 为第 10 个后续观测；均按股票交易序列生成，不用日历日近似。
        df = df.drop(columns=['date_key'])
        all_data.append(df)
        logger.log(f'  [{i}/{len(stock_codes)}] {code}: {len(df)} 行')

    logger.log(f'行情获取完成，耗时 {time.time() - t0:.1f}s，成功 {len(all_data)} 只，失败 {len(failed)} 只')
    if failed:
        for code, reason in failed:
            logger.log(f'  失败: {code}: {reason}')
    if not all_data:
        raise RuntimeError('没有任何股票数据获取成功。')

    panel = pd.concat(all_data, ignore_index=True)
    panel = panel.sort_values(['date', 'stock_code']).reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    membership_path = os.path.join(OUTPUT_DIR, 'membership_intervals.json')
    with open(membership_path, 'w', encoding='utf-8') as f:
        json.dump(memberships, f, ensure_ascii=False, indent=2)
    st_path = os.path.join(OUTPUT_DIR, 'historical_st_intervals.json')
    with open(st_path, 'w', encoding='utf-8') as f:
        json.dump({'intervals': st_intervals, 'unknown_status_codes': unknown_status_codes},
                  f, ensure_ascii=False, indent=2)
    logger.log(f'面板 CSV 已保存: {PANEL_PATH}')
    logger.log(f'历史成分股区间已保存: {membership_path}')
    logger.log(f'历史 ST 区间已保存: {st_path}')
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
    global OUTPUT_DIR, PANEL_PATH, CONSTITUENTS_PATH, LOG_PATH
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

    OUTPUT_DIR = args.out_dir
    PANEL_PATH = os.path.join(OUTPUT_DIR, 'panel.csv')
    CONSTITUENTS_PATH = os.path.join(OUTPUT_DIR, 'constituents.json')
    LOG_PATH = os.path.join(OUTPUT_DIR, 'fetch_log.txt')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = Logger(os.path.join(args.out_dir, 'fetch_log.txt'))
    try:
        ts.set_token(args.token)
        pro = ts.pro_api()

        # Step 1: 成分股
        universe = get_sse50_constituents(pro, args.start_year, args.end_year, logger)
        periods = universe['periods']
        with open(CONSTITUENTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(universe, f, ensure_ascii=False, indent=2)
        logger.log(f'历史成分股区间及摘要已保存: {CONSTITUENTS_PATH}')

        # Step 2: 行情 + 面板
        start_date = f'{args.start_year}0101'
        end_date = f'{args.end_year}1231'
        panel = build_panel(pro, universe, start_date, end_date, logger)

        # Step 3: 校验
        validate_and_report(panel, logger)

    finally:
        logger.close()


if __name__ == '__main__':
    main()