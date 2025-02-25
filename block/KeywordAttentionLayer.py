import torch.nn as nn


class KeywordAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads)
        self.norm = nn.LayerNorm(hidden_size)  # 新增层归一化

    def forward(self, bert_output, keyword_mask):
        key_padding_mask = ~keyword_mask.bool()  # 正确掩码格式
        attn_output, _ = self.attention(
            bert_output, bert_output, bert_output,
            key_padding_mask=key_padding_mask
        )
        # 残差连接 + 层归一化
        output = self.norm(bert_output + attn_output)
        return output