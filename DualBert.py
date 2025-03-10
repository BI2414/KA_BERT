import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import BertModel, BertPreTrainedModel
from block.KeywordAttentionLayer import KeywordAttentionLayer  # 假设已实现

# 在DualBert.py中定义Adapter模块
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.up_proj(self.activation(self.down_proj(x)))  # ✅ 确保 residual 连接

# 添加交叉注意力层
class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, query, key_value):
        query = query.permute(1, 0, 2)  # [seq_len, batch, hidden]
        key_value = key_value.permute(1, 0, 2)
        attn_output, _ = self.attn(query, key_value, key_value)
        return attn_output.permute(1, 0, 2)

class EnhancedDualBERT(BertPreTrainedModel):
    """
    改进的Dual-BERT召回模型，集成STAMP的关键模块
    主要特性：
    - 双塔独立编码器
    - 关键词注意力机制
    - 自适应噪声注入
    - 门控特征融合
    - 投影降维
    """

    def __init__(self, config, args):
        super().__init__(config)  # 先调用父类初始化
        self.args = args

        # 初始化双编码器
        self.query_bert = BertModel(config)
        self.doc_bert = BertModel(config)

        # 共享底层参数（需确保query_bert和doc_bert已实例化）
        if args.share_low_layers > 0:
            for i in range(args.share_low_layers):
                # 直接共享参数，而非替换层对象
                self.doc_bert.encoder.layer[i].load_state_dict(
                    self.query_bert.encoder.layer[i].state_dict()
                )

        # 添加Adapter
        self.query_bert = self._add_adapters(self.query_bert, args.adapter_size)
        self.doc_bert = self._add_adapters(self.doc_bert, args.adapter_size)

        # --------------------------
        # STAMP改进模块
        # --------------------------
        # 关键词注意力层（Query和Doc共享）
        self.keyword_attention = KeywordAttentionLayer(config.hidden_size)

        # 噪声生成网络
        self.noise_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        )

        # 门控融合模块
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        ) if args.use_gate else None

        # --------------------------
        # 投影与正则化
        # --------------------------
        # 降维投影
        self.query_proj = nn.Linear(config.hidden_size, args.embed_dim)
        self.doc_proj = nn.Linear(config.hidden_size, args.embed_dim)

        # 初始化参数
        self.init_weights()

        self.attn_gate = MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,  # 可调整头数
            dropout=0.1
        ) if args.use_gate else None
        self.cross_attn = CrossAttention(config.hidden_size)

    def _add_adapters(self, bert_model, adapter_size):
        """为BERT模型的每一层添加Adapter模块"""
        for layer in bert_model.encoder.layer:
            layer.adapter = Adapter(bert_model.config.hidden_size, adapter_size)
        return bert_model

    def _apply_keyword_attention(self, hiddens, attention_mask):
        """应用关键词注意力增强"""
        # 生成动态掩码
        with torch.no_grad():
            dummy_outputs = self.query_bert(
                inputs_embeds=hiddens,
                attention_mask=attention_mask,
                output_attentions=True
            )
            attn_weights = dummy_outputs.attentions[-1]
            keyword_mask = self._generate_keyword_mask(attn_weights)

        return self.keyword_attention(hiddens, keyword_mask)

    def _generate_keyword_mask(self, attn_weights, topk=8):
        """动态生成关键词掩码（同STAMP逻辑）"""
        # attn_weights形状: [batch, heads, seq_len, seq_len]
        importance = attn_weights.mean(dim=1)  # 平均多头注意力
        importance = importance.mean(dim=-1)  # 被关注度 [batch, seq_len]

        # 选取topk重要位置
        _, topk_indices = importance.topk(topk, dim=-1)
        mask = torch.zeros_like(importance, dtype=torch.bool)
        for b in range(mask.size(0)):
            mask[b, topk_indices[b]] = True
        return mask

    def _apply_noise(self, hiddens):
        """使用注意力门控融合噪声"""
        mu, logvar = torch.chunk(self.noise_net(hiddens), 2, dim=-1)
        noise = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        if self.attn_gate is not None:
            # 将原始表示作为Query，噪声作为Key和Value
            hiddens_ = hiddens.permute(1, 0, 2)  # [seq_len, batch, hidden]
            noise_ = noise.permute(1, 0, 2)
            attn_output, _ = self.attn_gate(
                query=hiddens_,
                key=noise_,
                value=noise_,
                need_weights=False
            )
            attn_output = attn_output.permute(1, 0, 2)  # 恢复维度
            return attn_output + hiddens, mu, logvar  # 残差连接

        return hiddens + noise, mu, logvar

    def encode_query(self, input_ids, attention_mask):
        # 基础编码
        outputs = self.query_bert(input_ids=input_ids, attention_mask=attention_mask)
        hiddens = outputs.last_hidden_state
        attention_mask = attention_mask.to(dtype=torch.float)  # ✅ 解决错误
        print("Query BERT hidden size:", self.query_bert.config.hidden_size)
        print("Doc BERT hidden size:", self.doc_bert.config.hidden_size)
        # 遍历每一层并应用Adapter
        for layer in self.query_bert.encoder.layer:
            # 原始BERT层的前向传播
            hiddens = layer.adapter(hiddens)  # ✅ 确保 Adapter 不改变 hidden_size

            # 应用Adapter并残差连接
            hiddens = layer.adapter(hiddens)
        # 其他处理（如噪声注入、关键词注意力等）
        hiddens = self._apply_keyword_attention(hiddens, attention_mask)
        hiddens, mu, logvar = self._apply_noise(hiddens)
        return self.query_proj(hiddens[:, 0]), mu, logvar

    def encode_doc(self, input_ids, attention_mask):
        # 处理三维输入 (batch_size, num_candidates, seq_len)
        batch_size, num_cand, seq_len = input_ids.size()
        attention_mask = attention_mask.to(dtype=torch.float)  # ✅ 解决错误
        # 展平维度 (batch_size*num_cand, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        # 通过BERT模型
        outputs = self.doc_bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            return_dict=True
        )

        # 获取池化输出并恢复维度
        pooled_output = outputs.pooler_output

        # 添加文档投影降维
        pooled_output = self.doc_proj(pooled_output)  # [batch_size*num_cand, embed_dim]

        # 恢复三维结构
        return pooled_output.view(batch_size, num_cand, -1)  # [batch, num_cand, embed_dim]

    def forward(self, query_inputs, doc_inputs):
        q_embeds = self.encode_query(**query_inputs)
        d_embeds = self.encode_doc(**doc_inputs)
        print(q_embeds.shape, d_embeds.shape)  # Debug 输出

        # 交叉注意力增强
        cross_embeds = self.cross_attn(q_embeds.unsqueeze(1), d_embeds)
        scores = torch.matmul(q_embeds.unsqueeze(1), cross_embeds.transpose(1, 2)).squeeze(1)
        return scores


    def compute_loss(self, scores, labels, mu, logvar):
        contrast_loss = F.cross_entropy(scores, labels)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return contrast_loss + self.args.kl_weight * kl_loss