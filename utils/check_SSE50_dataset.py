import sys, os
sys.path.append(os.getcwd())
from data_provider.data_loader import Dataset_MultiStock

class Args:
    task_name = 'long_term_forecast'
    embed = 'timeF'
    freq = 'd'
    seasonal_patterns = 'Quarterly'

args = Args()

# 训练集
train_set = Dataset_MultiStock(
    args=args,
    root_path='./dataset/SSE50/',
    data_path='panel.csv',
    flag='train',
    size=[60, 0, 1],
    features='MS',
    target='target',
    scale=True,
    timeenc=0,
    freq='d',
    window_stride=1
)
print(f'训练集样本数: {len(train_set)}')
x, y, xm, ym = train_set[0]
print(f'输入 x 形状: {x.shape}    (期望 (60, 10))')
print(f'输出 y 形状: {y.shape}    (期望 (1, 1))')
print(f'x_mark 形状: {xm.shape}   (期望 (60, 4))')
print(f'y_mark 形状: {ym.shape}   (期望 (1, 4))')
print(f'第一个样本 y 值 (标准化后): {y.flatten()}')
print(f'反标准化后: {train_set.inverse_transform(y).flatten()}')

# 验证集
vali_set = Dataset_MultiStock(args=args, root_path='./dataset/SSE50',
                               data_path='panel.csv', flag='val',
                               size=[60, 0, 1], features='MS',
                               target='target', scale=True, timeenc=0,
                               freq='d', window_stride=1)
print(f'\n验证集样本数: {len(vali_set)}')

# 测试集
test_set = Dataset_MultiStock(args=args, root_path='./dataset/SSE50',
                               data_path='panel.csv', flag='test',
                               size=[60, 0, 1], features='MS',
                               target='target', scale=True, timeenc=0,
                               freq='d', window_stride=1)
print(f'测试集样本数: {len(test_set)}')

import numpy as np

# 取全部训练集的 x
X_all = train_set.data_x          # 形状 (N, 60, 10)
print(f'\n训练集 x 总形状: {X_all.shape}')

# 对每个特征，统计它在所有样本、所有时间步上的均值和标准差
print('\n--- 每个特征的全局统计（应该：标准化后的接近0/1，pct_chg 接近原始值）---')
for i, col in enumerate(train_set.feature_cols):
    vals = X_all[:, :, i].flatten()
    print(f'  {col:10s}  mean={vals.mean():8.4f}  std={vals.std():8.4f}')

# 反标准化验证：随便取一个样本的 target，还原后应该等于该股票某日真实收盘价
print('\n--- 反标准化验证 ---')
for idx in [0, 1000, 5000]:
    x_, y_, _, _ = train_set[idx]
    y_std = y_.flatten()[0]
    y_raw = train_set.inverse_transform(y_).flatten()[0]
    print(f'  sample {idx}: y_std={y_std:8.4f}  ->  y_raw={y_raw:8.4f}')