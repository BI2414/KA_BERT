import os
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from math import sqrt
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
from block.KeywordAttentionLayer import KeywordAttentionLayer


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)

class NewBert(nn.Module):
    def __init__(self, args):
        super(NewBert, self).__init__()
        self.model_name = args["model"]
        self.bert_model = BertForSequenceClassification.from_pretrained(
            args["model"], num_labels=args["n_class"], output_attentions=True, output_hidden_states=True
        )

        self.noise_net = nn.Sequential(nn.Linear(args["hidden_size"],
                                                 args["hidden_size"]),
                                       nn.ReLU(),
                                       nn.Linear(args["hidden_size"],
                                                 args["hidden_size"] * 2))
        config = self.bert_model.config
        self.config = config
        self.dropout = config.hidden_dropout_prob  # 0.1
        self.args = args

        # 添加 KeywordAttentionLayer
        self.keyword_attention = KeywordAttentionLayer(hidden_size=args["hidden_size"])

        if self.args["gate"]:
            self.Gate = nn.Sequential(nn.Linear(2 * args["hidden_size"],
                                                args["hidden_size"]),
                                      nn.ReLU(),
                                      nn.Linear(args["hidden_size"],
                                                2))

    def generate_keyword_mask(self, attention_weights, hidden_states, topk=3):
        """修正后的梯度计算"""
        importance_attn = attention_weights.mean(dim=1).mean(dim=-1)

        if self.training:
            # 使用autograd.grad替代backward()
            cls_embed = hidden_states[:, 0, :]
            grads = torch.autograd.grad(
                outputs=cls_embed.mean(),
                inputs=hidden_states,
                create_graph=True,
                retain_graph=True
            )[0]
            importance_grad = torch.norm(grads, dim=-1)
            importance = 0.7 * importance_attn + 0.3 * importance_grad
        else:
            importance = importance_attn

        _, topk_indices = importance.topk(topk, dim=-1)
        keyword_mask = torch.zeros_like(importance, dtype=torch.bool)
        keyword_mask.scatter_(1, topk_indices, True)
        return keyword_mask

    def forward(self, input_ids, attention_mask, token_type_ids, labels, keyword_mask):
        # 复用第一次前向传播的结果
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        attention_weights = outputs.attentions[-1]
        hidden_states = outputs.hidden_states[-1]

        # 生成关键词掩码
        keyword_mask = self.generate_keyword_mask(
            attention_weights=attention_weights,
            hidden_states=hidden_states,
            topk=3
        )

        # 获取 BERT 输出
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        # 获取最后一层的隐藏状态
        hiddens = outputs.hidden_states[-1]  # 最后一层隐状态

        if self.args["aug"]:
            # 使用 keyword_mask 增强隐藏层
            if keyword_mask is not None:
                # 接收三个返回值
                hiddens, attn_out, residual = self.keyword_attention(hiddens, keyword_mask)

                # 安全计算比例
                if self.training:
                    # 按样本计算后取平均
                    residual_norms = torch.norm(residual, dim=(1, 2))  # [batch]
                    attn_norms = torch.norm(attn_out, dim=(1, 2))  # [batch]
                    ratios = attn_norms / (residual_norms + 1e-6)
                    print(f"样本平均残差比: {ratios.mean().item():.2f}±{ratios.std().item():.2f}")

            embeddings = self.bert_model.get_input_embeddings()
            hiddens = outputs.hidden_states[-1]
            inputs_embeds = embeddings(input_ids)

            # 生成噪声后，应用 keyword_mask 保护关键位置
            if self.args['uniform']:
                uniform_noise = torch.empty(inputs_embeds.shape).uniform_(0.9995, 1.0005).to(self.args['device'])
                noise = uniform_noise
            else:
                # 生成噪声逻辑保持不变
                mask = attention_mask.view(-1)
                indices = (mask == 1)
                mu_logvar = self.noise_net(hiddens)
                mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
                zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                noise = zs
                if keyword_mask is not None:
                    # 扩展 keyword_mask 的维度以匹配 noise 的形状 [batch, seq_len, hidden_size]
                    keyword_mask_expanded = keyword_mask.unsqueeze(-1).expand_as(noise)
                    noise = noise.masked_fill(keyword_mask_expanded, 1.0)  # 关键位置噪声置为1

                prior_mu = torch.ones_like(mu)
                # If p < 0.5, sqrt makes variance the larger
                prior_var = torch.ones_like(mu) * sqrt(self.dropout / (1-self.dropout))
                prior_logvar = torch.log(prior_var)

                kl_criterion = GaussianKLLoss()
                h = mu.size(-1)
                _mu = mu.view(-1, h)[indices]  # 或者使用 mu.reshape(-1, h)[indices]
                _log_var = log_var.view(-1, h)[indices]
                _prior_mu = prior_mu.view(-1, h)[indices]
                _prior_logvar = prior_logvar.view(-1, h)[indices]

                kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            # 统一应用噪声保护
            if keyword_mask is not None:
                keyword_mask_expanded = keyword_mask.unsqueeze(-1).expand_as(noise)
                noise = noise.masked_fill(keyword_mask_expanded, 1.0)
            # Random discarding to aug
            rands = torch.randperm(inputs_embeds.size(0))[:self.args["zero_peturb"]].tolist()

            for index in rands:
                embed_ = inputs_embeds[index, :, :]
                length = torch.randint(1, 4, (1,)).item()  # 生成1-3
                for iter in range(length):
                    index_ = torch.randint(1, inputs_embeds.shape[1], (1,)).item()
                    vec = torch.rand(1, inputs_embeds.shape[-1]).to(self.args["device"])
                    embed_[index_] = vec
            #噪声增强的模型

            inputs = {"inputs_embeds": inputs_embeds * noise,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "labels":labels}

            noise_outputs = self.bert_model(**inputs, output_hidden_states = True)
            noise_loss = noise_outputs[0]
            #原始模型
            new_inputs = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "labels":labels}

            outputs = self.bert_model(**new_inputs, output_hidden_states = True)
            nll = outputs[0]

            if self.args["gate"]:
                last_noise = noise_outputs.hidden_states[-1]
                last = outputs.hidden_states[-1]
                cls_noise = last_noise[:, 0, :]  # [batch, hidden]
                cls = last[:, 0, :]  # [batch, hidden]
                cls_total = torch.cat((cls_noise, cls), dim=1)
                res = self.Gate(cls_total)  # [batch, 2]
                temperature =0.5
                Gates = F.softmax(res / temperature, dim=-1)

                # 获取样本级损失
                noise_logits = noise_outputs.logits
                nll_logits = outputs.logits

                noise_losses = F.cross_entropy(
                    noise_logits.view(-1, self.args["n_class"]),
                    labels.view(-1),
                    reduction='none'
                ).view_as(labels)  # [batch]

                nll_losses = F.cross_entropy(
                    nll_logits.view(-1, self.args["n_class"]),
                    labels.view(-1),
                    reduction='none'
                ).view_as(labels)  # [batch]

                # 动态加权
                loss = (noise_losses * Gates[:, 0] + nll_losses * Gates[:, 1]).mean()
                # loss = nll + 0.001 * noise_loss
            else:
                loss = nll + 0.001 * noise_loss #loss 为公式10的前半部分 nll为loss(p,h) noise_loss为loss(p`,h`)
            if self.args['uniform']:
                return (loss, 0 * loss, outputs.logits)
            else:
                return (loss, kl, outputs.logits)
        else:
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids}
            # output_hidden_states = True 输出隐含状态
            outputs = self.bert_model(**inputs, labels=labels)
            return outputs.loss, 0, outputs.logits
            