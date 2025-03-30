# CMEDQADataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import os
import pickle
from hashlib import md5
from tqdm import tqdm

class CMEDQADataset(Dataset):
    """cMedQA2数据集专用处理类（无动态采样）"""
    # 全局缓存字典（避免重复加载）
    # 第一层缓存：全局问题和答案文本映射
    _qid_to_text = None
    _aid_to_text = None

    def __init__(
            self,
            data_path,
            mode='train',
            max_candidates=200,
            tokenizer=None,  # ✅ 允许为None
            max_len=128,  # ✅ 默认值
            cache_dir=".cache"
    ):
        """
        Args:
            data_path: 数据集根目录（包含question.csv, answer.csv等）
            mode: train/dev/test
            max_candidates: 评估时最大候选数
            tokenizer: 分词器（用于缓存扩展）
            max_len: 序列最大长度（用于缓存扩展）
            cache_dir: 缓存文件存储目录
        """
        self.mode = mode
        self.data_path = data_path
        self.max_cand = max_candidates
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # === 第一层缓存：加载全局文本映射 ===#
        if CMEDQADataset._qid_to_text is None:
            # 加载问题文本映射
            q_path = os.path.join(data_path, "question.csv")
            questions = pd.read_csv(
                q_path,
                names=['id', 'text'],
                dtype={'id': str, 'text': str},
                on_bad_lines='skip'
            )
            CMEDQADataset._qid_to_text = dict(zip(questions['id'], questions['text']))

            # 加载答案文本映射
            a_path = os.path.join(data_path, "answer.csv")
            answers = pd.read_csv(
                a_path,
                names=['id', 'question_id', 'text'],
                dtype={'id': str, 'question_id': str, 'text': str},
                on_bad_lines='skip'
            )
            CMEDQADataset._aid_to_text = dict(zip(answers['id'], answers['text']))

        self.qid_to_text = CMEDQADataset._qid_to_text
        self.aid_to_text = CMEDQADataset._aid_to_text

        # === 第二层缓存：处理后的样本数据 ===#
        # 生成唯一缓存文件名（包含分词器哈希和关键参数）
        tokenizer_hash = md5(pickle.dumps(tokenizer)).hexdigest()[:8]
        cache_name = f"{mode}_mc{max_candidates}_ml{max_len}_tok{tokenizer_hash}.pkl"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            # 从缓存加载预处理后的 Tensor 数据
            with open(cache_path, "rb") as f:
                self.tokenized_samples = pickle.load(f)  # 直接加载分词结果
        else:
            # 无缓存时处理原始数据并分词
            if mode == 'train':
                self._process_train()
            else:
                self._process_eval()
            # self._validate_data_consistency()

            self.tokenized_samples = []
            for sample in tqdm(self.samples, desc="Tokenizing Data"):
                # 获取问题和候选答案文本
                query_text = self.qid_to_text[sample['question_id']]

                # 根据模式获取候选答案ID列表
                if self.mode == 'train':
                    # 训练模式：candidates 是 [pos_ans_id, neg_ans_id]
                    candidate_ids = sample['candidates']
                else:
                    # 评估模式：candidates 是多个答案ID
                    candidate_ids = sample['candidates']

                # 转换为文本
                candidate_texts = [
                    self.aid_to_text.get(ans_id, "[UNK]")
                    for ans_id in candidate_ids
                ]

                # Tokenize 查询和候选
                query_enc = tokenizer(
                    query_text,
                    max_length=max_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                cand_enc = tokenizer(
                    candidate_texts,
                    max_length=max_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                    return_attention_mask=True
                )

                # 存储为字典（避免重复分词）
                self.tokenized_samples.append({
                    'query_input_ids': query_enc['input_ids'].squeeze(0),  # [seq_len]
                    'query_attention_mask': query_enc['attention_mask'].squeeze(0),
                    'cand_input_ids': cand_enc['input_ids'],  # [num_cand, seq_len]
                    'cand_attention_mask': cand_enc['attention_mask'],
                    'labels': torch.FloatTensor(sample['labels'])
                })

            # 保存预处理结果
            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_samples, f)


    def _process_train(self):
        """训练数据处理：每行直接作为一个样本（1正1负）"""
        self.samples = []

        # 显式指定列数据类型为字符串（避免自动推断）
        self.data = pd.read_csv(
            # f"{self.data_path}/train_candidates.txt",
            f"{self.data_path}/train.txt",
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

        # 构建统一格式的 samples
        for _, row in self.data.iterrows():
            self.samples.append({
                'question_id': row['question_id'].strip(),
                'candidates': [  # ✅ 关键修改：构造 candidates 列表
                    row['pos_ans_id'].strip(),
                    row['neg_ans_id'].strip()
                ],
                'labels': [1, 0]  # ✅ 标签直接存储
            })

    def _process_eval(self):
        """评估数据处理：聚合每个问题的所有候选答案"""
        self.query_dict = defaultdict(lambda: {'candidates': [], 'labels': []})
        self.samples = []  # ✅ 新增：初始化 samples 列表

        # 显式指定列数据类型
        self.data = pd.read_csv(
            # f"{self.data_path}/{self.mode}_candidates.txt",
            f"{self.data_path}/{self.mode}.txt",
            names=['question_id', 'ans_id', 'cnt', 'label'],
            dtype={'question_id': str, 'ans_id': str, 'cnt': int, 'label': int},
            sep=',',
            header=0,
            engine='python',
            on_bad_lines='warn'
        )

        # 过滤包含非数字ID的行
        self.data = self.data[
            self.data['question_id'].str.isdigit() &
            self.data['ans_id'].str.isdigit()
            ]

        # 删除空值和无效标签
        self.data = self.data.dropna()
        self.data = self.data[self.data['label'].isin([0, 1])]

        # 构建 samples 列表
        for question_id, group in self.data.groupby('question_id'):
            candidates = group['ans_id'].tolist()
            labels = group['label'].tolist()
            self.samples.append({
                'question_id': question_id,
                'candidates': candidates[:self.max_cand],
                'labels': labels[:self.max_cand]
            })

        self.question_ids = [sample['question_id'] for sample in self.samples]

    def __len__(self):
        return len(self.tokenized_samples)  # 直接返回分词后样本数

    def __getitem__(self, idx):
        # 直接返回预处理的 Tensor 数据
        return self.tokenized_samples[idx]

    def _validate_data_consistency(self):
        def is_valid_question_id(s):
            # 检查是否为数字且在映射字典中存在
            if not s.isdigit():
                return False
            return s in self.qid_to_text
        def is_valid_ans_id(s):
            # 检查是否为数字且在映射字典中存在
            if not s.isdigit():
                return False
            return s in self.aid_to_text

        # 训练数据验证
        if self.mode == 'train':
            for idx, sample in enumerate(self.samples[:100]):
                question_id = sample['question_id']
                pos_ans_id = sample['pos_ans_id']
                neg_ans_id = sample['neg_ans_id']
                assert is_valid_question_id(question_id), f"训练数据第 {idx} 行: question_id={question_id} 无效"
                assert is_valid_ans_id(pos_ans_id), f"训练数据第 {idx} 行: pos_ans_id={pos_ans_id} 无效"
                assert is_valid_ans_id(neg_ans_id), f"训练数据第 {idx} 行: neg_ans_id={neg_ans_id} 无效"

    @classmethod
    def clear_cache(cls, cache_dir=".cache"):
        """清空所有缓存文件（用于数据更新后）"""
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, f))
def cmedqa_collate_fn(batch):
    """合并预处理的 Tensor 数据"""
    return {
        'query': {
            'input_ids': torch.stack([item['query_input_ids'] for item in batch]),  # [batch, seq_len]
            'attention_mask': torch.stack([item['query_attention_mask'] for item in batch])
        },
        'candidates': {
            'input_ids': torch.stack([item['cand_input_ids'] for item in batch]),  # [batch, num_cand, seq_len]
            'attention_mask': torch.stack([item['cand_attention_mask'] for item in batch])
        },
        'labels': torch.stack([item['labels'] for item in batch])  # [batch, num_cand]
    }


# 示例用法
if __name__ == "__main__":
    # 首次加载会生成缓存
    train_set = CMEDQADataset(
        data_path="data/wjh/graduate/AugData/cMedQA2",
        mode='train',
        cache_dir=".cache"
    )

    # 后续加载直接读取缓存
    dev_set = CMEDQADataset(
        data_path="data/wjh/graduate/AugData/cMedQA2",
        mode='dev',
        cache_dir=".cache"
    )

    # 清空缓存（当原始数据更新时调用）
    CMEDQADataset.clear_cache()