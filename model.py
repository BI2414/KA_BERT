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
                                                3))

    def forward(self, input_ids, attention_mask, token_type_ids, labels, keyword_mask):

        # 第一次调用 BERT 获取注意力权重和隐藏状态
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True  # 必须显式指定
            )
            attention_weights = outputs.attentions[-1]  # 取最后一层注意力权重

            # 假设选择每个位置中注意力权重最高的token作为关键词
            keyword_mask = (attention_weights.mean(dim=1) > 0.5).any(dim=1)  # [batch, seq_len]
        # 获取 BERT 输出
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        # 第二次调用 BERT 获取隐藏状态
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True  # 必须显式指定
        )

        # 获取最后一层的隐藏状态
        hiddens = outputs.hidden_states[-1]  # 最后一层隐状态

        if self.args["aug"]:
            # 使用 keyword_mask 增强隐藏层
            if keyword_mask is not None:
                hiddens = self.keyword_attention(hiddens, keyword_mask)  # 修改位置
            embeddings = self.bert_model.get_input_embeddings()
            encoder = self.bert_model.bert
            with torch.no_grad():
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}

                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]
            inputs_embeds = embeddings(input_ids)


            if self.args['uniform']:
                # low is 0.95, high is 1.05 Try to produce softer noise as much as possible，to avoid semantic space collapse
                uniform_noise = torch.empty(inputs_embeds.shape).uniform_(0.9995, 1.0005).to(self.args['device'])
                noise = uniform_noise
            else:
                mask = attention_mask.view(-1)
                indices = (mask == 1)
                # 确保掩码的形状与目标张量一致
                mu_logvar = self.noise_net(hiddens)
                mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
                zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                noise = zs

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

            # Random discarding to aug
            rands = list(set([random.randint(1, inputs_embeds.shape[0] - 1) for i in range(self.args["zero_peturb"])]))

            for index in rands:
                embed_ = inputs_embeds[index, :, :]
                length = random.randint(1, 3)
                for iter in range(length):
                    index_ = random.randint(1, inputs_embeds.shape[1] - 1)
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

            if self.args["gate"]: #等于adapter
                # 获取CLS
                last_noise = noise_outputs.hidden_states[-1]
                last = outputs.hidden_states[-1]
                cls_noise = last_noise[:, :1, :].squeeze()  # 获取噪声向量的CLS
                cls = last[:, :1, :].squeeze()  # 获取原始向量的CLS
                cls_total = torch.cat((cls_noise, cls), dim=1)
                cls_total = torch.mean(cls_total, dim=0).unsqueeze(dim=0)  # 合并并计算平均
                # 将CLS通过Gate网络
                res = self.Gate(cls_total)
                temperature = 0.5
                Gates = F.softmax(res / temperature, dim=-1).squeeze()
                # Gates = F.softmax(res, dim=-1).squeeze()  # Gate 权重，通过softmax标准化
                # Adapter 动态调整损失
                print("CLS Total:", cls_total.mean().item())
                print("Gates:", Gates)
                loss = noise_loss * Gates[0] + nll * Gates[1]  # 动态调整损失
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
            