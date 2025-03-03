import torch.nn as nn
import torch


class GatedResidualAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, residual, attn_output):
        # 动态门控融合
        combined = torch.cat([residual, attn_output], dim=-1)
        gate = self.gate(combined)
        return residual * gate + attn_output * (1 - gate)

class KeywordAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_heads=1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, bert_output, keyword_mask):
        # 显式保留中间变量
        residual = bert_output  # [batch, seq, hidden]
        query = self.query(bert_output)

        # 注意力计算
        attn_output, _ = self.attention(
            query.transpose(0, 1),
            residual.transpose(0, 1),  # 使用residual作为key/value
            residual.transpose(0, 1),
            key_padding_mask=~keyword_mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)

        # 返回结果和中间变量
        output = self.layer_norm(residual + self.dropout(attn_output))
        output = self.layer_norm(residual + self.gate * self.dropout(attn_output))
        return (
            self.layer_norm(residual + self.dropout(attn_output)),  # 主输出
            attn_output.detach(),  # 监控用，阻断梯度
            residual.detach()  # 监控用，阻断梯度
        )