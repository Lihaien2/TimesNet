# -*- coding: utf-8 -*-
"""
TimesNet Exchange 长期预测一体化实验脚本（修复版）
- 自动训练
- 自动解析损失曲线
- 自动生成图表和表格
- 所有输出保存到 exchange_result/run_时间戳/ 目录
"""

import os
import sys
import re
import glob
import time
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 实验配置
# ============================================================
CONFIG = {
    # 任务
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'itr': 1,
    'num_workers': 0,

    # 数据（exchange_rate.csv 就在 ./dataset/ 下）
    'root_path': './dataset/',
    'data_path': 'exchange_rate.csv',
    'data': 'custom',
    'features': 'M',
    'target': 'OT',

    # 序列长度
    'seq_len': 96,
    'label_len': 48,
    'pred_len': 96,

    # 模型
    'model': 'TimesNet',
    'model_id': 'Exchange_96_96',
    'top_k': 5,
    'num_kernels': 6,
    'enc_in': 8,
    'dec_in': 8,
    'c_out': 8,
    'd_model': 64,
    'd_ff': 64,
    'e_layers': 2,
    'd_layers': 1,
    'factor': 3,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',

    # 训练
    'train_epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'loss': 'MSE',
    'lradj': 'type1',
    'patience': 3,

    # 其他
    'des': 'Exchange_Exp',
}

# 这些参数是 action='store_true'，只能出现或消失，不能跟值
STORE_TRUE_ARGS = {
    'inverse', 'distil', 'use_amp', 'use_gpu', 'no_use_gpu',
    'use_multi_gpu', 'use_dtw', 'individual',
    'jitter', 'scaling', 'permutation', 'randompermutation',
    'magwarp', 'timewarp', 'windowslice', 'windowwarp',
    'rotation', 'spawner', 'dtwwarp', 'shapedtwwarp',
    'wdba', 'discdtw', 'discsdtw',
}


# ============================================================
# 2. 输出目录
# ============================================================
OUTPUT_ROOT = './exchange_result'
OUTPUT_DIR = os.path.join(OUTPUT_ROOT)
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print('=' * 80)
print(f'Output directory: {os.path.abspath(OUTPUT_DIR)}')
print('=' * 80)


# ============================================================
# 3. 构建命令（正确区分 store_true）
# ============================================================
def build_command(config):
    cmd = [sys.executable, '-u', 'run.py']
    for k, v in config.items():
        if k in STORE_TRUE_ARGS:
            if v:
                cmd.append(f'--{k}')
        else:
            cmd.append(f'--{k}')
            cmd.append(str(v))
    return cmd


cmd = build_command(CONFIG)
print('\nCommand:')
print(' '.join(cmd))
print('=' * 80)
print('Training started...\n')

start_time = time.time()
# 关键：强制子进程使用 UTF-8，避免 Windows GBK 报错
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUTF8'] = '1'
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    bufsize=1,
)

log_lines = []
for line in process.stdout:
    print(line, end='')
    log_lines.append(line)

process.wait()
train_time = time.time() - start_time

log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.writelines(log_lines)

print(f'\nTraining finished in {train_time:.2f} seconds')
print(f'Log saved to: {log_path}')


# ============================================================
# 4. 解析损失曲线
# ============================================================
epoch_pattern = re.compile(
    r'Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*'
    r'Train Loss:\s*([\d.eE+-]+)\s+'
    r'Vali Loss:\s*([\d.eE+-]+)\s+'
    r'Test Loss:\s*([\d.eE+-]+)'
)

epochs, train_losses, val_losses, test_losses = [], [], [], []
for line in log_lines:
    m = epoch_pattern.search(line)
    if m:
        epochs.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
        val_losses.append(float(m.group(3)))
        test_losses.append(float(m.group(4)))

best_epoch, best_val = None, None
if val_losses:
    best_idx = int(np.argmin(val_losses))
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

print(f'\nParsed {len(epochs)} epochs')
if best_epoch is not None:
    print(f'Best epoch: {best_epoch}, best val loss: {best_val:.6f}')


# ============================================================
# 5. 解析测试指标
# ============================================================
metric_pattern = re.compile(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')
test_mse_log, test_mae_log = None, None
for line in log_lines:
    m = metric_pattern.search(line)
    if m:
        test_mse_log = float(m.group(1))
        test_mae_log = float(m.group(2))

if test_mse_log is not None:
    print(f'Logged test MSE: {test_mse_log:.6f}, MAE: {test_mae_log:.6f}')


# ============================================================
# 6. 定位结果目录并加载预测
# ============================================================
pattern = (f'./results/long_term_forecast_{CONFIG["model_id"]}_'
           f'{CONFIG["model"]}_{CONFIG["data"]}_*{CONFIG["des"]}_0')
matches = glob.glob(pattern)

preds, trues, metrics_arr = None, None, None
if matches:
    result_dir = max(matches, key=os.path.getctime)
    print(f'\nResult directory: {result_dir}')

    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    metrics_path = os.path.join(result_dir, 'metrics.npy')

    if os.path.exists(pred_path) and os.path.exists(true_path):
        preds = np.load(pred_path)
        trues = np.load(true_path)
        print(f'Preds shape: {preds.shape}, Trues shape: {trues.shape}')

    if os.path.exists(metrics_path):
        metrics_arr = np.load(metrics_path)
        print(f'metrics.npy [mae, mse, rmse, mape, mspe]: {metrics_arr}')
else:
    print('\nWarning: no result directory found.')


# ============================================================
# 7. 生成图表
# ============================================================
def save_fig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {name}')


print('\nGenerating plots...')

# 图 1：损失曲线（必做）
if epochs:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=7)
    plt.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=7)
    plt.plot(epochs, test_losses, '^-', label='Test Loss', linewidth=2, markersize=7)
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.8,
                    label=f'Best epoch={best_epoch} (val={best_val:.4f})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'Loss Curve - {CONFIG["model"]} on Exchange '
              f'(seq_len={CONFIG["seq_len"]}, pred_len={CONFIG["pred_len"]})', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_fig('01_loss_curve.png')

# 图 2：单样本预测 vs 真实
if preds is not None and trues is not None:
    plt.figure(figsize=(14, 5))
    plt.plot(trues[0, :, 0], label='Ground Truth', linewidth=2, color='black')
    plt.plot(preds[0, :, 0], label='Prediction', linewidth=2,
             color='tab:orange', linestyle='--')
    plt.xlabel('Forecast Horizon', fontsize=12)
    plt.ylabel('Normalized Value', fontsize=12)
    plt.title('Prediction vs Ground Truth (Sample 0, Variable 0)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_fig('02_sample_prediction.png')

# 图 3：8 个变量全部展示
if preds is not None and trues is not None:
    n_vars = trues.shape[-1]
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.array(axes).flatten()
    for c in range(n_vars):
        ax = axes[c]
        ax.plot(trues[0, :, c], label='True', linewidth=1.6, color='black')
        ax.plot(preds[0, :, c], label='Pred', linewidth=1.6,
                color='tab:orange', linestyle='--')
        ax.set_title(f'Variable {c}', fontsize=11)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(fontsize=9)
    for c in range(n_vars, len(axes)):
        axes[c].axis('off')
    plt.suptitle('All Variables Forecast (Sample 0)', fontsize=14)
    save_fig('03_all_variables.png')

# 图 4：散点图
if preds is not None and trues is not None:
    plt.figure(figsize=(7, 7))
    plt.scatter(trues.flatten(), preds.flatten(), s=1, alpha=0.2, color='steelblue')
    lo = float(min(trues.min(), preds.min()))
    hi = float(max(trues.max(), preds.max()))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal')
    plt.xlabel('Ground Truth', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    plt.title('Prediction vs Ground Truth Scatter', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_fig('04_scatter.png')

# 图 5：误差分布
if preds is not None and trues is not None:
    errors = (preds - trues).flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=100, alpha=0.75, color='steelblue', edgecolor='white')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Error Distribution', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_fig('05_error_hist.png')

# 图 6：每变量 MSE 柱状图
if preds is not None and trues is not None:
    per_var_mse = ((preds - trues) ** 2).mean(axis=(0, 1))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(per_var_mse)), per_var_mse,
                   color='steelblue', edgecolor='black')
    plt.xticks(range(len(per_var_mse)),
               [f'Var {i}' for i in range(len(per_var_mse))])
    plt.xlabel('Variable', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Per-Variable MSE on Test Set', fontsize=13)
    for b, v in zip(bars, per_var_mse):
        plt.text(b.get_x() + b.get_width() / 2, v, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    save_fig('06_per_variable_mse.png')

print(f'All plots saved to: {os.path.abspath(PLOTS_DIR)}')


# ============================================================
# 8. 保存实验配置表
# ============================================================
config_df = pd.DataFrame([{'Parameter': k, 'Value': v} for k, v in CONFIG.items()])
config_csv = os.path.join(OUTPUT_DIR, 'experiment_config.csv')
config_df.to_csv(config_csv, index=False, encoding='utf-8-sig')
print(f'\nConfig table saved: {config_csv}')


# ============================================================
# 9. 保存实验结果表
# ============================================================
if preds is not None and trues is not None:
    errors = preds - trues
    eps = 1e-8

    mse = float((errors ** 2).mean())
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(mse))
    mape = float((np.abs(errors) / (np.abs(trues) + eps)).mean() * 100)

    results = {
        'Task': CONFIG['task_name'],
        'Model': CONFIG['model'],
        'Dataset': 'Exchange',
        'Seq Len': CONFIG['seq_len'],
        'Pred Len': CONFIG['pred_len'],
        'Num Variables': int(trues.shape[-1]),
        'Test Samples': int(trues.shape[0]),
        'MSE': round(mse, 6),
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'MAPE (%)': round(mape, 4),
        'Best Epoch': best_epoch if best_epoch is not None else 'N/A',
        'Best Val Loss': round(best_val, 6) if best_val is not None else 'N/A',
        'Total Epochs Run': len(epochs),
        'Training Time (s)': round(train_time, 2),
        'Top k': CONFIG['top_k'],
        'd_model': CONFIG['d_model'],
        'd_ff': CONFIG['d_ff'],
        'e_layers': CONFIG['e_layers'],
        'Learning Rate': CONFIG['learning_rate'],
        'Batch Size': CONFIG['batch_size'],
    }

    results_df = pd.DataFrame([results])
    results_csv = os.path.join(OUTPUT_DIR, 'experiment_results.csv')
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f'Results table saved: {results_csv}')

    results_txt = os.path.join(OUTPUT_DIR, 'experiment_results.txt')
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('Experiment Results\n')
        f.write('=' * 60 + '\n')
        for k, v in results.items():
            f.write(f'{k:25s}: {v}\n')
    print(f'Results text saved: {results_txt}')

    np.save(os.path.join(OUTPUT_DIR, 'pred.npy'), preds)
    np.save(os.path.join(OUTPUT_DIR, 'true.npy'), trues)
    print(f'Predictions saved to: {os.path.abspath(OUTPUT_DIR)}')

else:
    print('\nNo predictions loaded, skipping results table.')


# ============================================================
# 10. 完成
# ============================================================
print('\n' + '=' * 80)
print('ALL DONE')
print('=' * 80)
print(f'Output directory: {os.path.abspath(OUTPUT_DIR)}')
print(f'  training_log.txt')
print(f'  experiment_config.csv')
print(f'  experiment_results.csv / .txt')
print(f'  pred.npy / true.npy')
print(f'  plots/01_loss_curve.png')
print(f'  plots/02_sample_prediction.png')
print(f'  plots/03_all_variables.png')
print(f'  plots/04_scatter.png')
print(f'  plots/05_error_hist.png')
print(f'  plots/06_per_variable_mse.png')