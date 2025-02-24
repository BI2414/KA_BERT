import torch.nn as nn
class KeywordAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_heads=1):
        super(KeywordAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads)
        self.hidden_size = hidden_size

    def forward(self, bert_output, keyword_mask):
        """
        bert_output: BERT模型的输出，[batch_size, seq_len, hidden_size]
        keyword_mask: 用于标记关键词的mask，[batch_size, seq_len]
        """
        # 注意力掩码，关键词标记为1，其他为0
        attention_mask = keyword_mask.unsqueeze(1).expand(-1, bert_output.size(1), -1)

        # 注意力计算，使用自注意力机制
        output, _ = self.attention(bert_output, bert_output, bert_output, key_padding_mask=attention_mask)
        return output
