# import os
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
# import DualBert
# from nltk.corpus import wordnet
# import random
# import jieba
# import synonyms
# import nltk
# # nltk.download('wordnet')
#
#
# # 初始化分词工具
# jieba.initialize()
#
# # 改进的增强函数（支持中文）
# def augment_positive(text,
#                      num_augments=3,  # 生成多个增强样本
#                      delete_prob=0.15,
#                      swap_prob=0.15,
#                      insert_prob=0.1,
#                      replace_prob=0.15):
#     """多策略文本增强（同义词替换+删除+交换+插入）"""
#
#     def synonym_replace(words):
#         """随机替换一个词为同义词"""
#         if len(words) < 2:
#             return words
#         replace_idx = random.randint(0, len(words) - 1)
#         synonyms_list = synonyms.nearby(words[replace_idx])[0]
#         if len(synonyms_list) > 1:
#             words[replace_idx] = synonyms_list[1]  # 选择最相似的同义词
#         return words
#
#     def random_delete(words):
#         """随机删除一个词"""
#         if len(words) > 2 and random.random() < delete_prob:
#             del words[random.randint(0, len(words) - 1)]
#         return words
#
#     def random_swap(words):
#         """随机交换相邻两个词"""
#         if len(words) > 2 and random.random() < swap_prob:
#             idx = random.randint(0, len(words) - 2)
#             words[idx], words[idx + 1] = words[idx + 1], words[idx]
#         return words
#
#     def random_insert(words):
#         """随机插入一个词（复制已有词）"""
#         if len(words) > 1 and random.random() < insert_prob:
#             insert_idx = random.randint(0, len(words) - 1)
#             words.insert(insert_idx, words[insert_idx])  # 复制已有词
#         return words
#
#     words = list(jieba.cut(text))
#     augmented_texts = set()  # 用于存储增强样本，避免重复
#
#     for _ in range(num_augments):
#         new_words = words[:]  # 复制原文本
#
#         if random.random() < replace_prob:
#             new_words = synonym_replace(new_words)
#         if random.random() < delete_prob:
#             new_words = random_delete(new_words)
#         if random.random() < swap_prob:
#             new_words = random_swap(new_words)
#         if random.random() < insert_prob:
#             new_words = random_insert(new_words)
#
#         augmented_text = ''.join(new_words)
#         if augmented_text != text:
#             augmented_texts.add(augmented_text)  # 避免生成与原文本相同的增强样本
#
#     return list(augmented_texts)  # 返回多个增强版本
#
#
# class BQPairwiseDataset(Dataset):
#     """支持一对多训练的BQ数据集加载器"""
#
#     def __init__(self, data_path, tokenizer, mode='train',
#                  max_len=128, num_pos=3, num_neg=8):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.num_pos = num_pos
#         self.num_neg = num_neg
#         self.mode = mode
#         self.data = self._load_data(data_path, mode)
#         self.all_neg_pool = self._build_global_neg_pool()  # 全局负样本池
#
#     def _build_global_neg_pool(self):
#         """构建全局负样本池（其他Query的正样本）"""
#         return [p for item in self.data for p in item['pos']]
#
#     def _load_data(self, path, mode):
#         """修复标题行处理的数据加载方法"""
#         query_dict = {}
#         line_count = 0
#         file_path = os.path.join(path, f"{mode}.tsv")
#
#         with open(file_path, 'r', encoding='utf-8') as f:
#             # 明确跳过标题行 -----------------------------------------------------
#             first_line = f.readline().strip()
#             if first_line.lower() == 'sentence1\tsentence2\tlabel':
#                 print(f"已跳过标题行: {first_line}")
#             else:
#                 # 如果第一行不是标题，则回退到文件开头
#                 f.seek(0)
#
#             # 处理数据行 ---------------------------------------------------------
#             for line_num, line in enumerate(f, 1):
#                 line = line.strip()
#                 if not line:
#                     continue
#
#                 # 严格的三列格式检查 ----------------------------------------------
#                 parts = line.split('\t')
#                 if len(parts) != 3:
#                     print(f"跳过格式错误行 #{line_num}: {line}")
#                     continue
#
#                 query, candidate, label_str = parts
#
#                 # 标签合法性检查 -------------------------------------------------
#                 try:
#                     label = int(label_str)
#                     if label not in (0, 1):
#                         raise ValueError
#                 except ValueError:
#                     print(f"非法标签值 '{label_str}'，行号 {line_num}: {line}")
#                     continue  # 跳过非法标签行
#
#                 # 存储到字典 -----------------------------------------------------
#                 if query not in query_dict:
#                     query_dict[query] = {'pos': [], 'neg': []}
#                 if label == 1:
#                     query_dict[query]['pos'].append(candidate)
#                 else:
#                     query_dict[query]['neg'].append(candidate)
#
#         valid_data = []
#         if self.mode == "dev":
#             for q, v in query_dict.items():
#                 # 放宽过滤条件：只要求至少有1个正样本和负样本
#                 if len(v['pos']) >= self.num_pos and len(v['neg']) >= 1:
#                     valid_data.append({'query': q, 'pos': v['pos'], 'neg': v['neg']})
#         else:
#             for q, v in query_dict.items():
#                 if len(v['pos']) >= self.num_pos and len(v['neg']) >=self.num_pos:
#                     valid_data.append({'query': q, 'pos': v['pos'], 'neg': v['neg']})
#         print(f"有效query数量: {len(valid_data)}")
#         return valid_data
#
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         query = item['query']
#         positives = list(set(item['pos']))  # 去重
#         negatives = list(set(item['neg']))  # 去重
#
#         # ================= 生成多个正样本增强 =================
#         augmented_positives = set()
#         for text in positives:
#             augmented_versions = augment_positive(text, num_augments=1)  # 每个正样本生成 3 个增强版本
#             augmented_positives.update(augmented_versions)
#
#         unique_positives = list(positives) + list(augmented_positives)  # 保持原始顺序
#
#         # ================= 改进的负样本采样 =================
#         if self.mode == "dev" :
#             # 合并本地和全局负样本池
#             local_negs = set(negatives)
#             global_negs = set(self.all_neg_pool) - set(positives)  # 排除当前query的正样本
#         else:
#             local_negs = set(negatives)
#             global_negs = set(random.sample(self.all_neg_pool, min(len(self.all_neg_pool), self.num_neg // 2)))
#
#         all_negs = list(local_negs | global_negs)  # 合并并去重
#
#
#         # ================= 严格采样（避免重复） =================
#         if len(unique_positives) >= self.num_pos:
#             pos_samples = random.sample(unique_positives, self.num_pos)
#         else:
#             pos_samples = unique_positives
#
#         if len(all_negs) >= self.num_neg:
#             neg_samples = random.sample(all_negs, self.num_neg)
#         else:
#             neg_samples = all_negs
#
#
#         # 强制采样固定数量（允许重复）
#         pos_samples = random.choices(positives, k=self.num_pos) if pos_samples else [""] * self.num_pos
#         neg_samples = random.choices(negatives, k=self.num_neg) if neg_samples else [""] * self.num_neg
#
#         # 合并候选
#         candidates = pos_samples + neg_samples
#         labels = [1] * self.num_pos + [0] * self.num_neg
#
#         return {
#             'query': query,
#             'candidates': candidates,
#             'labels': labels
#         }
