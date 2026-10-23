import sys, os
sys.path.append(os.getcwd())
import torch
from models.GRU import Model as GRUModel
from models.CNN import Model as CNNModel
from models.DLinear import Model as DLinearModel


class Configs:
    task_name = 'long_term_forecast'
    seq_len = 60
    label_len = 0
    pred_len = 1
    enc_in = 10
    dec_in = 10
    c_out = 1
    d_model = 32
    d_ff = 64
    e_layers = 2
    d_layers = 1
    dropout = 0.1
    embed = 'timeF'
    freq = 'd'          # ★ freq='d' → TimeFeatureEmbedding 期望 3 维
    top_k = 5
    num_kernels = 6


configs = Configs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

B = 8
x_enc = torch.randn(B, 60, 10).to(device)
x_mark_enc = torch.zeros(B, 60, 3).to(device)   # ★ 改 4 为 3
x_dec = torch.randn(B, 1, 10).to(device)
x_mark_dec = torch.zeros(B, 1, 3).to(device)    # ★ 改 4 为 3

models = [
    ('GRU', GRUModel),
    ('CNN', CNNModel),
    ('DLinear', DLinearModel),
]

for name, ModelClass in models:
    try:
        model = ModelClass(configs).to(device)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{name:8s} 输出形状={tuple(out.shape)}  参数量={n_params:,}')
    except Exception as e:
        print(f'{name:8s} 失败: {repr(e)}')

print('\n三个模型都输出 (8, 1, 1) 即正确。')