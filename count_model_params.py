#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计 CNN、DLinear、GRU、TimesNet 的参数量。

目录结构：
TimesNet/
├── run.py
├── count_model_params.py
├── models/
│   ├── CNN.py
│   ├── DLinear.py
│   ├── GRU.py
│   └── TimesNet.py
└── layers/

在项目根目录运行：
    python count_model_params.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


# 获取项目根目录，并添加到 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_params(model):
    """返回总参数量和可训练参数量。"""
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


def main():
    # 与 SSE50 实验配置保持一致
    # 如果你的 .sh 参数不同，请以实际运行配置为准
    configs = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=60,
        label_len=0,
        pred_len=1,
        enc_in=10,
        dec_in=10,
        c_out=1,
        embed="timeF",
        freq="d",
        d_model=32,
        d_ff=64,
        e_layers=2,
        d_layers=1,
        dropout=0.1,
        top_k=5,
        num_kernels=6,
        num_class=2,
    )

    # 从 models 子目录导入
    from models.CNN import Model as CNNModel
    from models.DLinear import Model as DLinearModel
    from models.GRU import Model as GRUModel
    from models.TimesNet import Model as TimesNetModel

    model_classes = [
        ("CNN", CNNModel),
        ("DLinear", DLinearModel),
        ("GRU", GRUModel),
        ("TimesNet", TimesNetModel),
    ]

    results = []

    print("=" * 65)
    print("SSE50 四个模型参数量统计")
    print("=" * 65)
    print(f"seq_len     = {configs.seq_len}")
    print(f"pred_len    = {configs.pred_len}")
    print(f"enc_in      = {configs.enc_in}")
    print(f"d_model     = {configs.d_model}")
    print(f"d_ff        = {configs.d_ff}")
    print(f"e_layers    = {configs.e_layers}")
    print(f"top_k       = {configs.top_k}")
    print(f"num_kernels = {configs.num_kernels}")
    print("-" * 65)

    for model_name, model_class in model_classes:
        print(f"正在统计 {model_name} ...")

        model = model_class(configs).cpu()

        total, trainable = count_params(model)

        results.append(
            (model_name, total, trainable)
        )

        print(
            f"{model_name:10s} "
            f"总参数量={total:,} "
            f"({total / 1e6:.6f} M)"
        )

        print(
            f"{'':10s} "
            f"可训练参数量={trainable:,} "
            f"({trainable / 1e6:.6f} M)"
        )

        del model

    print("\n" + "=" * 65)
    print("参数量统计汇总")
    print("=" * 65)

    print(
        f"{'模型':<12}"
        f"{'总参数量':>18}"
        f"{'可训练参数量':>20}"
    )

    print("-" * 50)

    for name, total, trainable in results:
        print(
            f"{name:<12}"
            f"{total:>18,}"
            f"{trainable:>20,}"
        )

    print("\n可直接复制到实验报告的 Markdown 表格：")
    print("| 模型 | 总参数量 | 可训练参数量 |")
    print("|---|---:|---:|")

    for name, total, trainable in results:
        print(
            f"| {name} | {total:,} | {trainable:,} |"
        )


if __name__ == "__main__":
    main()