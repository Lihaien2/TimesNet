@echo off
set CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_15_3 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --d_ff 64 --top_k 5 --num_kernels 6 --train_epochs 10 --batch_size 32 --learning_rate 0.0001 --des "Exchange_Exp" --itr 1 --num_workers 0