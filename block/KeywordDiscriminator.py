import torch
import torch.nn as nn
import torch.nn.functional as F

class KeywordDiscriminator(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        """
        关键词鉴别网络
        :param hidden_size: BERT 的隐藏层大小（通常为 768）
        :param num_heads: 注意力头的数量
        :param dropout: Dropout 概率
        """
        super(KeywordDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # 全连接层，用于输出关键词的重要性分数
        self.fc = nn.Linear(hidden_size, 1)

        # Layer normalization 和 Dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, token_embeddings, attention_mask=None):
        """
        :param token_embeddings: BERT 的词嵌入表示，形状为 (seq_len, batch_size, hidden_size)
        :param attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
        :return: 关键词的重要性分数，形状为 (batch_size, seq_len)
        """
        # 确保 attention_mask 是 boolean 类型
        attention_mask = attention_mask.bool()

        # 多头注意力机制
        attn_output, _ = self.multihead_attn(token_embeddings, token_embeddings, token_embeddings,
                                             key_padding_mask=attention_mask)
        attn_output = self.dropout_layer(attn_output)
        attn_output = self.layer_norm(token_embeddings + attn_output)  # 残差连接

        # 计算关键词的重要性分数
        keyword_scores = self.fc(attn_output)  # (seq_len, batch_size, 1)
        keyword_scores = keyword_scores.squeeze(-1).transpose(0, 1)  # (batch_size, seq_len)

        # 对分数进行归一化（Softmax）
        if attention_mask is not None:
            attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
        keyword_scores = F.softmax(keyword_scores + attention_mask, dim=-1)

        return keyword_scores