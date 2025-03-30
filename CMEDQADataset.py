import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class CMEDQADataset(Dataset):
    _cache_loaded = False
    _q_encodings = None
    _a_encodings = None
    _qid_map = None
    _aid_map = None

    def __init__(self, data_path, mode='train', max_candidates=200,
                 tokenizer=None, max_len=128, cache_dir=".cache",
                 device='cpu'):
        self.mode = mode
        self.max_cand = max_candidates
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.cache_dir = cache_dir

        # 初始化缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 加载或创建预编码数据
        if not self._cache_loaded:
            self._load_or_create_encodings(data_path)

        # 加载样本数据
        self.samples = self._load_samples(data_path, mode)

    def _load_or_create_encodings(self, data_path):
        """处理预编码的核心逻辑"""
        # 缓存文件路径
        q_cache = os.path.join(self.cache_dir, f"questions_{self.max_len}.pt")
        a_cache = os.path.join(self.cache_dir, f"answers_{self.max_len}.pt")
        map_cache = os.path.join(self.cache_dir, "id_maps.pkl")

        # 尝试加载缓存
        if all(os.path.exists(f) for f in [q_cache, a_cache, map_cache]):
            print("Loading precomputed encodings from cache...")
            CMEDQADataset._q_encodings = torch.load(q_cache)
            CMEDQADataset._a_encodings = torch.load(a_cache)
            with open(map_cache, 'rb') as f:
                maps = pickle.load(f)
                CMEDQADataset._qid_map = maps['qid_map']
                CMEDQADataset._aid_map = maps['aid_map']
            return

        print("Precomputing encodings...")
        # 加载原始数据
        q_df = pd.read_csv(os.path.join(data_path, "question.csv"),
                           names=['id', 'text'], dtype=str,
                           on_bad_lines='skip').dropna()
        a_df = pd.read_csv(os.path.join(data_path, "answer.csv"),
                           names=['id', 'question_id', 'text'], dtype=str,
                           on_bad_lines='skip').dropna()

        # 构建ID映射
        CMEDQADataset._qid_map = {qid: idx for idx, qid in enumerate(q_df['id'])}
        CMEDQADataset._aid_map = {aid: idx for idx, aid in enumerate(a_df['id'])}

        # 并行编码函数

        def _parallel_encode(texts):
            return self.tokenizer.batch_encode_plus(
                texts,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

        # 使用多进程加速
        with ProcessPoolExecutor() as executor:
            q_future = executor.submit(_parallel_encode, q_df['text'].tolist())
            a_future = executor.submit(_parallel_encode, a_df['text'].tolist())

            q_enc = q_future.result()
            a_enc = a_future.result()

        # 转换为优化格式
        CMEDQADataset._q_encodings = {
            'input_ids': q_enc['input_ids'].to(torch.int32),
            'attention_mask': q_enc['attention_mask'].to(torch.uint8)
        }
        CMEDQADataset._a_encodings = {
            'input_ids': a_enc['input_ids'].to(torch.int32),
            'attention_mask': a_enc['attention_mask'].to(torch.uint8)
        }

        # 保存缓存
        torch.save(CMEDQADataset._q_encodings, q_cache)
        torch.save(CMEDQADataset._a_encodings, a_cache)
        with open(map_cache, 'wb') as f:
            pickle.dump({
                'qid_map': CMEDQADataset._qid_map,
                'aid_map': CMEDQADataset._aid_map
            }, f)

        self._cache_loaded = True

    def _load_samples(self, data_path, mode):
        """加载样本关系的优化实现"""
        cache_file = os.path.join(self.cache_dir, f"{mode}_samples.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # 原始数据加载逻辑
        if mode == 'train':
            df = pd.read_csv(
                os.path.join(data_path, "train.txt"),
                sep=',',
                dtype={'question_id': str, 'pos_ans_id': str, 'neg_ans_id': str},
                header=0
            ).dropna()
            samples = [{
                'qid': row['question_id'].strip(),
                'candidates': [row['pos_ans_id'].strip(), row['neg_ans_id'].strip()],
                'labels': [1, 0]
            } for _, row in df.iterrows()]
        else:
            df = pd.read_csv(
                os.path.join(data_path, f"{mode}.txt"),
                sep=',',
                dtype={'question_id': str, 'ans_id': str, 'label': int},
                header=0
            ).dropna()
            samples = []
            for qid, group in df.groupby('question_id'):
                samples.append({
                    'qid': qid.strip(),
                    'candidates': group['ans_id'].str.strip().tolist()[:self.max_cand],
                    'labels': group['label'].tolist()[:self.max_cand]
                })

        # 缓存样本数据
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 获取问题编码
        q_idx = self._qid_map[sample['qid']]
        q_input_ids = self._q_encodings['input_ids'][q_idx].long()
        q_attention_mask = self._q_encodings['attention_mask'][q_idx].long()

        # 获取候选答案编码
        cand_indices = [self._aid_map[aid] for aid in sample['candidates']]
        a_input_ids = self._a_encodings['input_ids'][cand_indices].long()
        a_attention_mask = self._a_encodings['attention_mask'][cand_indices].long()

        return {
            'query_input_ids': q_input_ids,
            'query_attention_mask': q_attention_mask,
            'cand_input_ids': a_input_ids,
            'cand_attention_mask': a_attention_mask,
            'labels': torch.FloatTensor(sample['labels'])
        }


def optimized_collate_fn(batch):
    """优化后的批次处理函数"""
    return {
        'query': {
            'input_ids': torch.stack([item['query_input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['query_attention_mask'] for item in batch])
        },
        'candidates': {
            'input_ids': torch.stack([item['cand_input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['cand_attention_mask'] for item in batch])
        },
        'labels': torch.stack([item['labels'] for item in batch])
    }