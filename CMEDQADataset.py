# CMEDQADataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class CMEDQADataset(Dataset):
    """cMedQA2数据集专用处理类（无动态采样）"""

    def __init__(self, question_id_to_text, ans_id_to_text, data_path, mode='train', max_candidates=200):
        """
        Args:
            question_id_to_text: 问题ID到文本的字典
            ans_id_to_text: 答案ID到文本的字典
            data_path: 数据集根目录
            mode: train/dev/test
            max_candidates: 评估时的最大候选数（防止内存溢出）
        """
        self.mode = mode
        self.question_id_to_text = question_id_to_text
        self.ans_id_to_text = ans_id_to_text
        self.max_cand = max_candidates
        self.data_path = data_path

        # 加载原始数据
        if mode == 'train':
            self._process_train()
        else:
            self._process_eval()

        self._validate_data_consistency()  # ✅ 添加数据验证

    def _process_train(self):
        """训练数据处理：每行直接作为一个样本（1正1负）"""
        self.samples = []

        # 显式指定列数据类型为字符串（避免自动推断）
        self.data = pd.read_csv(
            f"{self.data_path}/train_candidates.txt",
            names=['question_id', 'pos_ans_id', 'neg_ans_id'],
            dtype={'question_id': str, 'pos_ans_id': str, 'neg_ans_id': str},  # ✅ 强制转换为字符串
            sep=',',  # 明确分隔符
            header=0,  # ✅ 跳过首行（标题行）
            engine='python',  # 避免C引擎解析问题
            on_bad_lines='warn'  # 跳过错误行
        )
        print(self.data.columns)  # 查看实际列名

        # 过滤包含非数字ID的行
        self.data = self.data[
            self.data['question_id'].str.isdigit() &
            self.data['pos_ans_id'].str.isdigit() &
            self.data['neg_ans_id'].str.isdigit()
            ]

        # 过滤无效行（可选）
        self.data = self.data.dropna()  # 删除包含空值的行

        for _, row in self.data.iterrows():
            self.samples.append({
                'question_id': row['question_id'].strip(),  # 去除首尾空格
                'pos_ans_id': row['pos_ans_id'].strip(),
                'neg_ans_id': row['neg_ans_id'].strip()
            })

    def _process_eval(self):
        """评估数据处理：聚合每个问题的所有候选答案"""
        self.query_dict = defaultdict(lambda: {'candidates': [], 'labels': []})

        # 显式指定列数据类型
        self.data = pd.read_csv(
            f"{self.data_path}/{self.mode}_candidates.txt",
            names=['question_id', 'ans_id', 'cnt', 'label'],
            dtype={'question_id': str, 'ans_id': str, 'cnt': int, 'label': int},  # ✅ 强制类型
            sep=',',
            header=0,  # ✅ 跳过首行（标题行）
            engine='python',
            on_bad_lines='warn'
        )
        print(self.data.columns)  # 查看实际列名
        # 过滤包含非数字ID的行
        self.data = self.data[
            self.data['question_id'].str.isdigit() &
            self.data['ans_id'].str.isdigit()
            ]

        # 删除空值和无效标签
        self.data = self.data.dropna()
        self.data = self.data[self.data['label'].isin([0, 1])]  # 只保留合法标签

        for _, row in self.data.iterrows():
            question_id = row['question_id'].strip()
            ans_id = row['ans_id'].strip()
            self.query_dict[question_id]['candidates'].append(ans_id)
            self.query_dict[question_id]['labels'].append(row['label'])

        self.question_ids = list(self.query_dict.keys())

    def __len__(self):
        return len(self.samples) if self.mode == 'train' else len(self.question_ids)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 训练模式：每个样本包含1正1负
            sample = self.samples[idx]
            query_text = self.question_id_to_text[sample['question_id']]
            pos_text = self.ans_id_to_text[sample['pos_ans_id']]
            neg_text = self.ans_id_to_text[sample['neg_ans_id']]
            return {
                'query': query_text,
                'candidates': [pos_text, neg_text],  # 固定顺序：正样本在前
                'labels': [1, 0]  # 对应正负标签
            }
        else:
            # 评估模式：返回所有候选（截断到max_cand）
            question_id = self.question_ids[idx]
            candidates = self.query_dict[question_id]['candidates'][:self.max_cand]
            labels = self.query_dict[question_id]['labels'][:self.max_cand]
            return {
                'query': self.question_id_to_text[question_id],
                'candidates': [self.ans_id_to_text.get(ans_id, "[UNK]") for ans_id in candidates],
                'labels': labels
            }

    def _validate_data_consistency(self):
        def is_valid_question_id(s):
            # 检查是否为数字且在映射字典中存在
            if not s.isdigit():
                return False
            return s in self.question_id_to_text
        def is_valid_ans_id(s):
            # 检查是否为数字且在映射字典中存在
            if not s.isdigit():
                return False
            return s in self.ans_id_to_text

        # 训练数据验证
        if self.mode == 'train':
            for idx, sample in enumerate(self.samples[:100]):
                question_id = sample['question_id']
                pos_ans_id = sample['pos_ans_id']
                neg_ans_id = sample['neg_ans_id']
                assert is_valid_question_id(question_id), f"训练数据第 {idx} 行: question_id={question_id} 无效"
                assert is_valid_ans_id(pos_ans_id), f"训练数据第 {idx} 行: pos_ans_id={pos_ans_id} 无效"
                assert is_valid_ans_id(neg_ans_id), f"训练数据第 {idx} 行: neg_ans_id={neg_ans_id} 无效"

def cmedqa_collate_fn(batch, tokenizer, max_len=128):
    """定制化批次处理函数"""
    batch_queries = [item['query'] for item in batch]
    batch_candidates = [item['candidates'] for item in batch]
    batch_labels = [item['labels'] for item in batch]

    # Tokenize查询
    query_enc = tokenizer(
        batch_queries,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Tokenize候选（三维结构：[batch, num_cand, seq_len]）
    flat_candidates = [c for sublist in batch_candidates for c in sublist]
    cand_enc = tokenizer(
        flat_candidates,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 重组为三维张量
    num_cand = len(batch_candidates[0])  # 训练时为2，评估时可变
    cand_input_ids = cand_enc['input_ids'].view(len(batch), num_cand, -1)
    cand_attention_mask = cand_enc['attention_mask'].view(len(batch), num_cand, -1)

    return {
        'query': {
            'input_ids': query_enc['input_ids'],
            'attention_mask': query_enc['attention_mask']
        },
        'candidates': {
            'input_ids': cand_input_ids,
            'attention_mask': cand_attention_mask
        },
        'labels': torch.FloatTensor(batch_labels)  # [batch, num_cand]
    }