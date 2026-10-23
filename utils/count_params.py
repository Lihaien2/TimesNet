"""
统计 TimesNet、GRU、CNN、DLinear 四个模型的参数量。
用法：
    python utils/count_params.py
"""

import sys
import os

# 保证能 import models/ 里的模块
sys.path.append(os.getcwd())

import torch


class Configs:
    """
    对应 scripts/long_term_forecast/SSE50_script/*.sh 里的超参数。
    如果 .sh 脚本里的参数有改动，这里也要同步。
    """
    # 任务配置
    task_name = 'long_term_forecast'
    seq_len = 60
    label_len = 0
    pred_len = 1
    features = 'MS'

    # 输入输出维度
    enc_in = 10
    dec_in = 10
    c_out = 1

    # 嵌入层配置
    embed = 'timeF'
    freq = 'd'
    dropout = 0.1

    # 模型通用配置
    d_model = 32
    d_ff = 64
    e_layers = 2
    d_layers = 1
    n_heads = 8
    factor = 1
    activation = 'gelu'
    distil = True

    # TimesNet 专用
    top_k = 3
    num_kernels = 6

    # DLinear 专用
    moving_avg = 25
    individual = False

    # 其他占位
    expand = 2
    d_conv = 4
    channel_independence = 1
    decomp_method = 'moving_avg'
    use_norm = 1
    down_sampling_layers = 0
    down_sampling_window = 1
    down_sampling_method = None
    seg_len = 96
    patch_len = 16
    node_dim = 10
    gcn_depth = 2
    gcn_dropout = 0.3
    propalpha = 0.3
    conv_channel = 32
    skip_channel = 32
    alpha = 0.1
    top_p = 0.5
    pos = 1
    use_D = 0
    tv_dt = 0
    tv_B = 0
    tv_C = 0


def count_module_params(model):
    """返回 {子模块名: 参数量} 的字典，用于分项打印。"""
    result = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            result[name] = n
    return result


def main():
    configs = Configs()

    # 延迟 import，避免某模型依赖缺失时整个脚本崩溃
    model_map = {}
    try:
        from models.TimesNet import Model as TimesNet
        model_map['TimesNet'] = TimesNet
    except Exception as e:
        print(f'[WARN] TimesNet 导入失败: {e}')

    try:
        from models.GRU import Model as GRU
        model_map['GRU'] = GRU
    except Exception as e:
        print(f'[WARN] GRU 导入失败: {e}')

    try:
        from models.CNN import Model as CNN
        model_map['CNN'] = CNN
    except Exception as e:
        print(f'[WARN] CNN 导入失败: {e}')

    try:
        from models.DLinear import Model as DLinear
        model_map['DLinear'] = DLinear
    except Exception as e:
        print(f'[WARN] DLinear 导入失败: {e}')

    if not model_map:
        print('没有成功导入任何模型，请检查 models/ 目录。')
        return

    print('=' * 70)
    print(f'配置: d_model={configs.d_model}, d_ff={configs.d_ff}, '
          f'e_layers={configs.e_layers}, seq_len={configs.seq_len}, '
          f'pred_len={configs.pred_len}')
    print(f'      enc_in={configs.enc_in}, c_out={configs.c_out}, '
          f'top_k={configs.top_k}, num_kernels={configs.num_kernels}')
    print('=' * 70)

    total_table = {}

    for name, ModelClass in model_map.items():
        try:
            model = ModelClass(configs)
        except Exception as e:
            print(f'\n[{name}] 构造失败: {e}')
            continue

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_table[name] = total

        print(f'\n[{name}]')
        print(f'  总参数量    : {total:>12,}')
        print(f'  可训练参数  : {trainable:>12,}')
        print(f'  分项:')
        for sub_name, n in count_module_params(model).items():
            print(f'    {sub_name:28s}: {n:>12,}')

    # 汇总表
    print('\n' + '=' * 70)
    print('汇总表')
    print('=' * 70)
    print(f'{"模型":<12s} {"参数量":>15s} {"与 CNN 的倍数":>15s}')

    # 以最小的模型为基准计算倍数
    if total_table:
        base = min(total_table.values())
        for name, n in total_table.items():
            ratio = n / base if base > 0 else 0
            print(f'{name:<12s} {n:>15,} {ratio:>14.2f}x')

    print()


if __name__ == '__main__':
    main()