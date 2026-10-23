#!/bin/bash
# 自动切换到项目根目录
cd "$(dirname "$0")/.."

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/SSE50/ \
  --data_path panel.csv \
  --model_id SSE50_test \
  --model TimesNet \
  --data multi_stock \
  --features MS \
  --target target \
  --freq d \
  --seq_len 60 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 1 \
  --d_layers 1 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --num_kernels 6 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 1 \
  --train_epochs 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss MSE \
  --des 'test' \
  --itr 1