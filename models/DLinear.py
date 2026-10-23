import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    DLinear 改造版，支持 features='MS'。
    原版 DLinear 输出 (B, pred_len, enc_in)，只适用于 features='M'。
    本版本在末端加一个 channel 投影，输出 (B, pred_len, c_out)。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

        # 移动平均分解，kernel_size 一般设 25
        self.decompsition = series_decomp(25)

        # 每个通道共享的线性层（非 individual 模式）
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # 用均匀权重初始化（DLinear 论文的做法）
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 末端投影：把 enc_in 个通道压缩成 c_out 个
        self.channel_proj = nn.Linear(self.enc_in, self.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, seq_len, enc_in)
        seasonal_init, trend_init = self.decompsition(x_enc)   # 各 (B, seq_len, enc_in)

        # 转成 (B, enc_in, seq_len)，让 Linear 作用于时间维
        seasonal_init = seasonal_init.permute(0, 2, 1)          # (B, enc_in, seq_len)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_out = self.Linear_Seasonal(seasonal_init)      # (B, enc_in, pred_len)
        trend_out = self.Linear_Trend(trend_init)               # (B, enc_in, pred_len)

        x_out = seasonal_out + trend_out                        # (B, enc_in, pred_len)
        x_out = x_out.permute(0, 2, 1)                          # (B, pred_len, enc_in)

        # 投影到 c_out 通道
        dec_out = self.channel_proj(x_out)                      # (B, pred_len, c_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None