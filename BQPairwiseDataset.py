import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import DualBert


class BQPairwiseDataset(Dataset):
    """支持一对多训练的BQ数据集加载器"""

    def __init__(self, data_path, tokenizer, mode='train',
                 max_len=128, num_pos=10, num_neg=10):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.data = self._load_data(data_path, mode)

    def _load_data(self, path, mode):
        """带标题行处理的加载方法"""
        query_dict = {}
        line_count = 0
        file_path = os.path.join(path, f"{mode}.tsv")

        with open(file_path, 'r', encoding='utf-8') as f:
            # 显式跳过标题行
            header = f.readline().strip().split('\t')
            if header == ['query', 'candidate', 'label']:
                print(f"检测到标题行，已跳过")
            else:
                # 如果不是标题格式，则回退到文件开头
                f.seek(0)

            for line_num, line in enumerate(f, 1):
                # if  line_count >= 2000:
                #     break  # 达到最大行数后停止读取
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 3:
                    print(f"跳过格式错误行 #{line_num}: {line}")
                    continue

                query, candidate, label_str = parts
                try:
                    label = int(label_str)
                except ValueError:
                    print(f"非法标签值 '{label_str}'，行号 {line_num}: {line}")
                    continue

                # 存储到query字典
                if query not in query_dict:
                    query_dict[query] = {'pos': [], 'neg': []}
                query_dict[query]['pos'].append(candidate) if label == 1 else query_dict[query]['neg'].append(candidate)

                line_count += 1  # 增加行数计数

        # 过滤有效数据并转换格式
        return [
            {'query': q, 'pos': v['pos'], 'neg': v['neg']}
            for q, v in query_dict.items()
            if len(v['pos']) > 0 and len(v['neg']) > 0
        ]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        positives = item['pos']
        negatives = item['neg']

        # 动态计算最小可用样本数
        min_count = min(len(positives), len(negatives))
        if min_count == 0:  # 理论上不会发生（加载时已过滤）
            return self.__getitem__(random.randint(0, len(self.data) - 1))  # 随机换一个样本

        # 强制1:1比例
        num_pairs = min(self.num_pos, self.num_neg, min_count)

        # 无重复采样（如果数据不足则重复）
        pos_samples = random.sample(positives, num_pairs) if len(positives) >= num_pairs \
            else random.choices(positives, k=num_pairs)
        neg_samples = random.sample(negatives, num_pairs) if len(negatives) >= num_pairs \
            else random.choices(negatives, k=num_pairs)

        # 合并候选（1:1比例）
        candidates = pos_samples + neg_samples
        labels = [1] * num_pairs + [0] * num_pairs

        return {
            'query': query,
            'candidates': candidates,
            'labels': labels
        }