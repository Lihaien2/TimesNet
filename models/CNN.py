import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    CNN 基线模型（1D 卷积）。
    架构: DataEmbedding -> 多层 Conv1d -> 全局平均池化 -> Linear 投影
    适用于: long_term_forecast, 单步或多步预测

    使用 Conv1d 沿时间维度滑动，kernel_size=3，padding=1，保持时间长度不变，
    最后用 AdaptiveAvgPool1d(1) 压成 (B, d_model, 1)，再投影。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        # 特征嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # 卷积模块：e_layers 层 Conv1d，每层 kernel=3, padding=1
        conv_layers = []
        for i in range(configs.e_layers):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=configs.d_model,
                    out_channels=configs.d_model,
                    kernel_size=3,
                    padding=1,
                )
            )
            conv_layers.append(nn.BatchNorm1d(configs.d_model))
            conv_layers.append(nn.ReLU())
            if configs.dropout > 0:
                conv_layers.append(nn.Dropout(configs.dropout))
        self.conv = nn.Sequential(*conv_layers)

        # 全局平均池化: (B, d_model, T) -> (B, d_model, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 投影
        self.projection = nn.Linear(
            configs.d_model, configs.pred_len * configs.c_out
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, seq_len, enc_in)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B, seq_len, d_model)

        # 转成 (B, d_model, seq_len) 供 Conv1d 使用
        enc_out = enc_out.permute(0, 2, 1)  # (B, d_model, seq_len)

        # 卷积
        conv_out = self.conv(enc_out)  # (B, d_model, seq_len)

        # 全局平均池化
        pooled = self.pool(conv_out).squeeze(-1)  # (B, d_model)

        # 投影
        out = self.projection(pooled)  # (B, pred_len * c_out)
        out = out.view(-1, self.pred_len, self.c_out)  # (B, pred_len, c_out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None