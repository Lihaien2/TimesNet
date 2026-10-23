import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    GRU 基线模型。
    架构: DataEmbedding -> GRU -> 取最后时间步 -> Linear 投影到 pred_len 步
    适用于: long_term_forecast, 单步或多步预测
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        # 特征嵌入: (B, seq_len, enc_in) -> (B, seq_len, d_model)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # GRU 编码器
        self.gru = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
        )

        # 投影: (B, d_model) -> (B, pred_len * c_out)
        self.projection = nn.Linear(
            configs.d_model, configs.pred_len * configs.c_out
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, seq_len, enc_in)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B, seq_len, d_model)

        # GRU
        gru_out, _ = self.gru(enc_out)  # (B, seq_len, d_model)

        # 取最后一个时间步
        last = gru_out[:, -1, :]  # (B, d_model)

        # 投影到 pred_len * c_out 然后 reshape
        out = self.projection(last)  # (B, pred_len * c_out)
        out = out.view(-1, self.pred_len, self.c_out)  # (B, pred_len, c_out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # (B, pred_len, c_out)
        # 其他任务暂不支持
        return None