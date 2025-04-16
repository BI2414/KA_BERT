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
        self.down = nn.Linear(hidden_size, adapter_size)
        self.up = nn.Linear(adapter_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.up(F.gelu(self.down(x)))
        return residual + x  # 更稳定的残差连接

# 添加交叉注意力层
class CrossAttention(nn.Module):
    def __init__(self, bert_hidden_size, embed_dim):  # 同时接收 BERT 原始维度和投影维度
        super().__init__()
        # 将投影后的维度（embed_dim）映射回 BERT 原始维度（bert_hidden_size）
        self.query_proj = nn.Linear(embed_dim, bert_hidden_size)
        self.key_proj = nn.Linear(embed_dim, bert_hidden_size)
        self.attn = MultiheadAttention(bert_hidden_size, num_heads=4)

    def forward(self, query, key_value):
        # 维度变换
        query = self.query_proj(query)
        key = self.key_proj(key_value)
        # 调整维度顺序 [seq_len, batch, features]
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = key  # 使用与 key 相同的值

        # 计算注意力
        attn_output, _ = self.attn(query, key, value)
        return attn_output.permute(1, 0, 2)  # 恢复 [batch, seq_len, features]

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
        # 修改投影层
        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_size, args.embed_dim),
            nn.LayerNorm(args.embed_dim)
        )
        self.doc_proj = nn.Sequential(
            nn.Linear(config.hidden_size, args.embed_dim),
            nn.LayerNorm(args.embed_dim)
        )
        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.05)

        # 初始化参数
        self.init_weights()

        self.attn_gate = MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,  # 可调整头数
            dropout=0.1
        ) if args.use_gate else None
        # 获取 BERT 的原始隐藏层维度（通常为768）
        bert_hidden_size = config.hidden_size
        # 传递两个参数：bert_hidden_size 和 embed_dim
        self.cross_attn = CrossAttention(bert_hidden_size, args.embed_dim)
        # 在初始化时注册 Adapter 到每一层
        for layer in self.query_bert.encoder.layer:
            layer.register_forward_hook(self._add_adapter_hook)

    @staticmethod
    def _add_adapter_hook(module, input, output):
        """在每层 BERT 的输出后自动添加 Adapter"""
        hidden_states = output[0]
        adapted = module.adapter(hidden_states)
        return (adapted,) + output[1:]  # 保持其他输出不变

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

    def _generate_keyword_mask(self, attn_weights, topk=4):
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
        outputs = self.query_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,  # ✅ 使用四维掩码
            output_hidden_states=True
        )

        # 直接获取最后一层输出
        hiddens = outputs.last_hidden_state
        # 其他处理（如噪声注入、关键词注意力等）
        hiddens = self._apply_keyword_attention(hiddens, attention_mask)
        hiddens, mu, logvar = self._apply_noise(hiddens)
        # 在 encode_query 中添加调试信息
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
        query_inputs["attention_mask"] = query_inputs["attention_mask"].to(dtype=torch.float)
        doc_inputs["attention_mask"] = doc_inputs["attention_mask"].to(dtype=torch.float)

        q_embeds, mu, logvar = self.encode_query(**query_inputs)  # [batch, embed_dim]
        d_embeds = self.encode_doc(**doc_inputs)  # [batch, num_cand, embed_dim]
        # 计算相似度得分
        scores = torch.matmul(
            q_embeds.unsqueeze(1),  # [batch, 1, embed_dim]
            d_embeds.transpose(1, 2)  # [batch, embed_dim, num_cand]
        ).squeeze(1)  # [batch, num_cand]

        # 或者方法2：如果您确实需要交叉注意力
        # cross_embeds = self.cross_attn(
        #     q_embeds.unsqueeze(1),  # [batch, 1, embed_dim]
        #     d_embeds  # [batch, num_cand, embed_dim]
        # )  # [batch, 1, embed_dim]
        # scores = torch.matmul(
        #     cross_embeds,
        #     d_embeds.transpose(1, 2)
        # ).squeeze(1)  # [batch, num_cand]

        # 应用温度系数
        scores = scores / self.temperature

        return scores, mu, logvar

    def compute_loss(self, scores, labels, mu, logvar):
        """计算损失函数，输入:
        - scores: [batch_size, num_candidates]
        - labels: [batch_size, num_candidates]
        """
        # 多标签二元交叉熵损失
        contrast_loss = F.binary_cross_entropy_with_logits(
            scores,  # 直接使用2D scores
            labels.float(),
            pos_weight=torch.tensor([5.0], device=scores.device))  # 正样本加权

        # KL散度正则项
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return contrast_loss + self.args.kl_weight * kl_loss