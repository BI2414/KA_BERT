import torch.nn as nn


class KeywordAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_heads=1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)  # 独立查询向量生成
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads)

    def forward(self, bert_output, keyword_mask):
        # 生成基于关键词的查询向量
        query = self.query(bert_output)  # [batch_size, seq_len, hidden_size]

        # 注意力计算（Query=关键词增强向量，Key=原始输出）
        attn_output, _ = self.attention(
            query.transpose(0, 1),  # MultiheadAttention 需要 [seq_len, batch, hidden_size]
            bert_output.transpose(0, 1),
            bert_output.transpose(0, 1),
            key_padding_mask=~keyword_mask.bool()
        )
        return attn_output.transpose(0, 1)  # 恢复为 [batch, seq_len, hidden_size]