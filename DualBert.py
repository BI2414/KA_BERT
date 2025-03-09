import torch
import torch.nn as nn
from click.core import F
from transformers import BertModel, BertPreTrainedModel
from block.KeywordAttentionLayer import KeywordAttentionLayer  # 假设已实现


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
        super().__init__(config)
        self.args = args

        # --------------------------
        # 双编码器架构
        # --------------------------
        # Query侧编码器
        self.query_bert = BertModel(config)
        # Doc侧编码器（与Query共享底层参数）
        self.doc_bert = BertModel(config)
        if args.share_low_layers > 0:  # 部分层共享
            for i in range(args.share_low_layers):
                self.doc_bert.encoder.layer[i] = self.query_bert.encoder.layer[i]

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
        """噪声注入与门控融合"""
        # 生成噪声参数
        mu, logvar = torch.chunk(self.noise_net(hiddens), 2, dim=-1)
        noise = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # 门控融合
        if self.gate is not None:
            gate = self.gate(torch.cat([hiddens, noise], dim=-1))
            return gate * hiddens + (1 - gate) * noise
        return hiddens + noise

    def encode_query(self, input_ids, attention_mask):
        """查询编码流程"""
        # 基础编码
        outputs = self.query_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hiddens = outputs.last_hidden_state

        # STAMP增强处理
        hiddens = self._apply_keyword_attention(hiddens, attention_mask)
        hiddens = self._apply_noise(hiddens)

        # 投影降维
        return self.query_proj(hiddens[:, 0])  # [batch, embed_dim]

    def encode_doc(self, input_ids, attention_mask):
        """文档编码流程（与Query对称但独立）"""
        outputs = self.doc_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hiddens = outputs.last_hidden_state

        hiddens = self._apply_keyword_attention(hiddens, attention_mask)
        hiddens = self._apply_noise(hiddens)

        return self.doc_proj(hiddens[:, 0])

    def forward(self, query_inputs, doc_inputs):
        """
        训练前向传播
        :param query_inputs: {input_ids: [batch, seq_len], attention_mask: [batch, seq_len]}
        :param doc_inputs: 同query_inputs结构
        :return: 相似度矩阵 [batch, batch]
        """
        # 编码Query和Doc
        q_embeds = self.encode_query(**query_inputs)  # [batch, dim]
        d_embeds = self.encode_doc(**doc_inputs)  # [batch, dim]

        # 计算相似度矩阵
        return torch.matmul(q_embeds, d_embeds.T)  # [batch, batch]

    def compute_loss(self, scores, labels, mu=None, logvar=None):
        """
        综合损失计算
        :param scores: 相似度矩阵
        :param labels: 真实标签
        :param mu: 噪声均值（来自noise_net）
        :param logvar: 噪声方差
        """
        # 对比损失
        contrast_loss = F.cross_entropy(scores, labels)

        # KL正则化损失
        kl_loss = 0
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return contrast_loss + self.args.kl_weight * kl_loss