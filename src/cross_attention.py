import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = dropout
        self.d_k = d_model // h
        self.d_v = d_model // h

        # 定义4个线性层：查询、键、值和最终输出
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        self.attn = F.softmax(scores, dim=-1)
        output = torch.matmul(self.attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)

        return self.dropout_layer(self.linears[-1](output))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FusionCoder(nn.Module):
    def __init__(self, d_model, attn, ff, dropout=0.1):
        super(FusionCoder, self).__init__()
        self.attn = attn
        self.ff = ff
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        # 对查询和键（即 event 和 query）应用交叉注意力
        attn_output = self.attn(query, key, key)
        attn_output = self.layer_norm(attn_output + query)
        ff_output = self.ff(attn_output)
        return self.layer_norm(ff_output + attn_output)
