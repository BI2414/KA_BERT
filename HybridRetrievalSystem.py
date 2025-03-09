import os
import json
import torch
import numpy as np
from math import sqrt
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer
from tqdm import tqdm
import jieba  # 中文分词工具
from src.config import get_argparse

args = get_argparse().parse_args()
args = vars(args)
class HybridRetrievalSystem:
    def __init__(self, args):
        # 初始化配置
        self.args = args
        self.device = args["device"]

        # 第一阶段：BM25初始化
        self._init_bm25(args["corpus_path"])

        # 第二阶段：改进的STAMP模型
        self.stamp_model = NewBert(args).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(args["model"])

        # 加载预计算嵌入（如果存在）
        self._precompute_stamp_embeddings()

    def _init_bm25(self, corpus_path):
        """加载语料库并构建BM25索引"""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = [line.strip() for line in f]

        # 中文分词处理
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 构建ID映射
        self.doc_ids = {i: doc for i, doc in enumerate(self.corpus)}

    def _precompute_stamp_embeddings(self):
        """预计算所有文档的STAMP增强嵌入"""
        cache_path = os.path.join(self.args["cache_dir"], "stamp_embeddings.pt")

        if os.path.exists(cache_path):
            self.stamp_embeddings = torch.load(cache_path)
        else:
            self.stamp_model.eval()
            embeddings = []

            with torch.no_grad():
                for doc in tqdm(self.corpus, desc="预计算STAMP嵌入"):
                    inputs = self.tokenizer(
                        doc,
                        max_length=self.args["max_length"],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    # 获取增强后的CLS嵌入
                    outputs = self.stamp_model(**inputs, labels=None)
                    cls_embed = outputs[1][:, 0, :].cpu()
                    embeddings.append(cls_embed)

            self.stamp_embeddings = torch.stack(embeddings).squeeze(1)
            torch.save(self.stamp_embeddings, cache_path)

    def _bm25_retrieve(self, query, top_k=1000):
        """第一阶段：BM25召回"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return top_indices

    def _stamp_rerank(self, query, candidate_indices, top_k=100):
        """第二阶段：STAMP增强重排序"""
        # 编码查询
        query_inputs = self.tokenizer(
            query,
            max_length=self.args["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # 获取查询嵌入
        with torch.no_grad():
            query_outputs = self.stamp_model(**query_inputs, labels=None)
            query_embed = query_outputs[1][:, 0, :]

        # 获取候选嵌入
        candidate_embeds = self.stamp_embeddings[candidate_indices].to(self.device)

        # 计算相似度
        similarities = torch.matmul(query_embed, candidate_embeds.T).squeeze(0)

        # 组合结果
        sorted_indices = torch.argsort(similarities, descending=True)[:top_k]
        return [self.doc_ids[candidate_indices[i]] for i in sorted_indices.cpu().numpy()]

    def retrieve(self, query, bm25_top_k=1000, final_top_k=100):
        """两阶段混合召回"""
        # 第一阶段：BM25粗筛
        bm25_candidates = self._bm25_retrieve(query, top_k=bm25_top_k)

        # 第二阶段：STAMP精排
        final_results = self._stamp_rerank(query, bm25_candidates, top_k=final_top_k)

        return final_results

    def adaptive_noise_injection(self, inputs_embeds, attention_mask):
        """动态噪声注入（与原始STAMP模型集成）"""
        # 生成关键词掩码
        with torch.no_grad():
            outputs = self.stamp_model.bert_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True
            )
            attention_weights = outputs.attentions[-1]
            keyword_mask = self.stamp_model.generate_keyword_mask(attention_weights)

        # 应用噪声
        if self.args["aug"]:
            noise = self.stamp_model.noise_net(outputs.hidden_states[-1])
            mu, log_var = torch.chunk(noise, 2, dim=-1)
            zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

            # 基于关键词掩码的差异扰动
            keyword_mask_expanded = keyword_mask.unsqueeze(-1).expand_as(zs)
            zs = torch.where(keyword_mask_expanded, zs * 0.3, zs * 1.0)

            return inputs_embeds * zs
        else:
            return inputs_embeds




# 初始化系统
retrieval_system = HybridRetrievalSystem(args)

# 使用示例
if __name__ == "__main__":
    query = "如何申请信用卡"
    results = retrieval_system.retrieve(query)

    print("混合召回结果：")
    for i, doc in enumerate(results[:10]):
        print(f"{i + 1}. {doc}")