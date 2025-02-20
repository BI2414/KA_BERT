import torch
import torch.nn as nn
import torch.nn.functional as F


class KeywordAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(KeywordAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
        """
        # 为了适应 nn.MultiheadAttention 的输入格式，转换为 (seq_len, batch_size, hidden_size)
        hidden_states = hidden_states.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        attention_mask = attention_mask.unsqueeze(0).repeat(hidden_states.size(0), 1,
                                                            1)  # (seq_len, batch_size, seq_len)

        # Attention 计算
        attn_output, attn_output_weights = self.attention(hidden_states, hidden_states, hidden_states,
                                                          key_padding_mask=attention_mask)

        # 加入 Dropout 和 LayerNorm
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + hidden_states)  # 残差连接
        return attn_output.transpose(0, 1), attn_output_weights  # (batch_size, seq_len, hidden_size)
