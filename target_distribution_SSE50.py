#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 SSE50 实际 Dataset_MultiStock 样本中归一化后 target 的分布。

把脚本放在 TimesNet 项目根目录（与 data_loader.py 同级），运行：
    python target_distribution_SSE50.py --root_path ./dataset/SSE50
或按项目实际 panel.csv 所在目录修改 --root_path。
统计对象是 Dataset_MultiStock 最终产出的 data_y，不是原始 CSV 的粗略日期切片。
"""
import argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_MultiStock


def summarize(split, ds):
    y = np.asarray(ds.data_y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise RuntimeError(f'{split} 集没有 target 样本')
    meta = pd.DataFrame(ds.data_meta)
    dates = pd.to_datetime(meta['date'])
    print(f'[{split}]')
    print(f'  n_samples       = {y.size}')
    print(f'  n_stocks        = {meta["stock_code"].nunique()}')
    print(f'  signal_date_min = {dates.min().date()}')
    print(f'  signal_date_max = {dates.max().date()}')
    print(f'  target_mean     = {y.mean():.10f}')
    print(f'  target_std_pop  = {y.std(ddof=0):.10f}')
    print(f'  target_std_samp = {y.std(ddof=1):.10f}' if y.size > 1 else '  target_std_samp = NA')
    print(f'  target_min      = {y.min():.10f}')
    print(f'  target_p25      = {np.quantile(y, .25):.10f}')
    print(f'  target_median   = {np.median(y):.10f}')
    print(f'  target_p75      = {np.quantile(y, .75):.10f}')
    print(f'  target_max      = {y.max():.10f}')
    return {
        'split': split, 'n_samples': y.size,
        'n_stocks': meta['stock_code'].nunique(),
        'signal_date_min': dates.min().date().isoformat(),
        'signal_date_max': dates.max().date().isoformat(),
        'mean': y.mean(), 'std_population': y.std(ddof=0),
        'std_sample': y.std(ddof=1) if y.size > 1 else np.nan,
        'min': y.min(), 'p25': np.quantile(y, .25),
        'median': np.median(y), 'p75': np.quantile(y, .75), 'max': y.max(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='./dataset/SSE50', help='panel.csv 所在目录')
    parser.add_argument('--data_path', default='panel.csv')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--window_stride', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=.7)
    parser.add_argument('--vali_ratio', type=float, default=.1)
    args = parser.parse_args()
    cfg = SimpleNamespace()
    rows = []
    for split in ('train', 'val', 'test'):
        ds = Dataset_MultiStock(
            args=cfg, root_path=args.root_path, flag=split,
            size=(args.seq_len, 0, 1), features='MS', data_path=args.data_path,
            target='target', scale=True, timeenc=0, freq='d',
            window_stride=args.window_stride,
            train_ratio=args.train_ratio, vali_ratio=args.vali_ratio,
        )
        rows.append(summarize(split, ds))
    out = pd.DataFrame(rows)
    out.to_csv('target_distribution_SSE50.csv', index=False, encoding='utf-8-sig')
    print('\n=== 汇总表（归一化后的实际 data_y）===')
    print(out.to_string(index=False, float_format=lambda x: f'{x:.8f}'))
    print('\n已写出 target_distribution_SSE50.csv')
    print('注：std_population 使用 ddof=0；比较分布漂移时建议以它作为主报告口径。')

if __name__ == '__main__':
    main()
