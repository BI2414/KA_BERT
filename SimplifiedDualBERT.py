import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import BertModel, BertPreTrainedModel

# 定义交叉注意力层
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, query, key_value):
        query = query.permute(1, 0, 2)    # [seq_len, batch, hidden]
        key_value = key_value.permute(1, 0, 2)
        attn_output, _ = self.attn(query, key_value, key_value)
        return attn_output.permute(1, 0, 2)

class SimplifiedDualBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

        # 使用BERT模型作为编码器
        self.query_bert = BertModel(config)
        self.doc_bert = BertModel(config)

        self.query_proj = nn.Linear(config.hidden_size, args.embed_dim)
        self.doc_proj = nn.Linear(config.hidden_size, args.embed_dim)

        self.cross_attn = CrossAttention(args.embed_dim)
        self.init_weights()

    def encode_query(self, input_ids, attention_mask):
        # query_inputs shape: [batch, seq_len]
        outputs = self.query_bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]的输出作为查询表示
        cls_output = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]
        return self.query_proj(cls_output)            # [batch, embed_dim]

    def encode_doc(self, input_ids, attention_mask):
        # doc_inputs shape: [batch, num_candidates, seq_len]
        batch_size, num_cand, seq_len = input_ids.shape
        # 将二维展平为 (batch*num_candidates, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.doc_bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_output = outputs.pooler_output          # [batch*num_candidates, hidden_size]
        projected = self.doc_proj(pooled_output)         # [batch*num_candidates, embed_dim]
        # 恢复三维结构
        return projected.view(batch_size, num_cand, -1)   # [batch, num_candidates, embed_dim]

    def forward(self, query_inputs, doc_inputs):
        # query_inputs: {'input_ids': [batch, seq_len], 'attention_mask': [batch, seq_len]}
        # doc_inputs: {'input_ids': [batch, num_candidates, seq_len], 'attention_mask': [batch, num_candidates, seq_len]}
        query_embeds = self.encode_query(**query_inputs)  # [batch, embed_dim]
        doc_embeds = self.encode_doc(**doc_inputs)          # [batch, num_candidates, embed_dim]

        # 扩展查询表示以匹配候选数量
        q_expanded = query_embeds.unsqueeze(1).expand(-1, doc_embeds.size(1), -1)  # [batch, num_candidates, embed_dim]
        cross_embeds = self.cross_attn(q_expanded, doc_embeds)  # [batch, num_candidates, embed_dim]

        # 计算相似度得分
        scores = torch.matmul(query_embeds.unsqueeze(1), cross_embeds.transpose(1, 2)).squeeze(1)
        return scores, None, None

    def compute_loss(self, scores, labels, mu, logvar):
        # 使用BCEWithLogitsLoss代替CrossEntropyLoss
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(scores, labels)

