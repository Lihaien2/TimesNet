import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 项目根目录 = 本脚本上级目录的上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_loss_curves(model_ids):
    save_path = os.path.join(RESULTS_DIR, 'loss_curves.png')
    plt.figure(figsize=(12, 6))
    colors = {'TimesNet': 'tab:blue', 'GRU': 'tab:orange',
              'CNN': 'tab:green', 'DLinear': 'tab:red'}

    for model_name, model_id in model_ids:
        path = os.path.join(RESULTS_DIR,
                            f'loss_history_{model_id}_{model_name}.json')
        if not os.path.exists(path):
            print(f'[WARN] 未找到: {path}')
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        epochs = range(1, len(data['train']) + 1)
        color = colors.get(model_name, 'gray')
        plt.plot(epochs, data['train'], label=f'{model_name} Train',
                 color=color, linestyle='-', linewidth=1.5)
        plt.plot(epochs, data['vali'], label=f'{model_name} Vali',
                 color=color, linestyle='--', linewidth=1.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training & Validation Loss Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'损失曲线已保存: {save_path}')


def plot_metrics_comparison(model_ids):
    save_path = os.path.join(RESULTS_DIR, 'metrics_comparison.png')
    metrics_name = ['MSE', 'MAE', 'RMSE', 'Direction_Acc',
                    'cumulative_return', 'annual_return',
                    'sharpe_ratio', 'max_drawdown', 'win_rate', 'n_trades']

    metrics_dict = {}
    for model_name, model_id in model_ids:
        path = os.path.join(RESULTS_DIR,
                            f'metrics_{model_id}_{model_name}.json')
        if not os.path.exists(path):
            print(f'[WARN] 未找到: {path}')
            continue
        with open(path, encoding='utf-8') as f:
            metrics_dict[model_name] = json.load(f)

    if not metrics_dict:
        print('没有找到任何指标文件')
        return

    model_names = list(metrics_dict.keys())
    colors = {'TimesNet': 'tab:blue', 'GRU': 'tab:orange',
              'CNN': 'tab:green', 'DLinear': 'tab:red'}

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_name):
        values = [metrics_dict[m].get(metric, 0) for m in model_names]
        bar_colors = [colors.get(m, 'gray') for m in model_names]
        bars = axes[i].bar(model_names, values, color=bar_colors)
        axes[i].set_title(metric, fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, values):
            txt = f'{val:.4f}' if abs(val) < 1000 else f'{val:.0f}'
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(), txt,
                         ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'指标对比图已保存: {save_path}')


def plot_predictions(model_ids, save_path='./results/predictions.png',
                     n_stocks=3, max_points=300):
    """
    按股票分组画预测对比。选样本数最多的 n_stocks 只股票，每只一个子图。
    """
    data = {}
    for model_name, model_id in model_ids:
        pred_path = f'./results/preds_{model_id}_{model_name}.npy'
        sid_path = f'./results/stock_ids_{model_id}_{model_name}.npy'
        if not os.path.exists(pred_path) or not os.path.exists(sid_path):
            print(f'[WARN] 缺少 {pred_path} 或 {sid_path}')
            continue
        data[model_name] = {
            'preds': np.load(pred_path),
            'trues': np.load(f'./results/trues_{model_id}_{model_name}.npy'),
            'sids': np.load(sid_path),
        }
        # 尝试加载 code_to_id 映射
        code_path = f'./results/code_to_id_{model_id}_{model_name}.json'
        if os.path.exists(code_path):
            with open(code_path) as f:
                code_map = json.load(f)
            data[model_name]['code_map'] = {int(v): k
                                            for k, v in code_map.items()}

    if not data:
        print('没有找到 preds/trues/stock_ids')
        return

    any_model = list(data.keys())[0]
    sids = data[any_model]['sids']
    code_map = data[any_model].get('code_map', {})

    unique_sids, counts = np.unique(sids, return_counts=True)
    order = np.argsort(-counts)
    top_sids = unique_sids[order[:n_stocks]]

    n_rows = len(top_sids)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, sid in zip(axes, top_sids):
        mask = sids == sid
        n_show = min(max_points, int(mask.sum()))

        true_seq = data[any_model]['trues'][mask][:n_show]
        ax.plot(true_seq, 'k-', label='Ground Truth', linewidth=2)

        for model_name, d in data.items():
            pred_seq = d['preds'][d['sids'] == sid][:n_show]
            ax.plot(pred_seq, '--', label=model_name, alpha=0.75)

        code_str = code_map.get(int(sid), f'ID {int(sid)}')
        ax.set_title(f'Stock {code_str}  (n={int(mask.sum())})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Close Price (元)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'预测对比图已保存: {save_path}')


if __name__ == '__main__':
    model_ids = [
        ('TimesNet', 'SSE50_TimesNet'),
        ('GRU', 'SSE50_GRU'),
        ('CNN', 'SSE50_CNN'),
        ('DLinear', 'SSE50_DLinear'),
    ]
    plot_loss_curves(model_ids)
    plot_metrics_comparison(model_ids)
    plot_predictions(model_ids)