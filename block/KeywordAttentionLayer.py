import torch.nn as nn


class KeywordAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_heads=1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads)
    
    def forward(self, bert_output, keyword_mask):

        # 生成基于关键词的查询向量
        query = self.query(bert_output)  # [batch_size, seq_len, hidden_size]

        # 注意力计算
        attn_output, _ = self.attention(
            query.transpose(0, 1),  # [seq_len, batch, hidden_size]
            bert_output.transpose(0, 1),
            bert_output.transpose(0, 1),
            key_padding_mask=~keyword_mask.bool()  # 注意：keyword_mask 需要是 bool 类型
        )
        attn_output = attn_output.transpose(0, 1)  # 恢复为 [batch, seq_len, hidden_size]

        # 确保 attn_output 的形状与 bert_output 一致
        assert attn_output.shape == bert_output.shape, "attn_output 形状不匹配"
        return attn_output