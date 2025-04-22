# CMEDQADataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import os
import pickle
from hashlib import md5
from tqdm import tqdm
import gc


class CMEDQADataset(Dataset):
    """处理cMedQA2数据集的类（支持分批分词以避免内存不足）"""
    _qid_to_text = None
    _aid_to_text = None

    def __init__(
            self,
            data_path,
            mode='train',
            max_candidates=200,
            tokenizer=None,
            max_len=128,
            cache_dir=".cache",
            chunk_size=1000000  # 新增批次大小参数
    ):
        self.mode = mode
        self.data_path = data_path
        self.max_cand = max_candidates
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size  # 控制分词批次大小
        os.makedirs(cache_dir, exist_ok=True)

        # 加载全局文本映射（保持不变）
        if CMEDQADataset._qid_to_text is None:
            q_path = os.path.join(data_path, "question.csv")
            questions = pd.read_csv(
                q_path,
                names=['id', 'text'],
                dtype={'id': str, 'text': str},
                on_bad_lines='skip'
            )
            CMEDQADataset._qid_to_text = dict(zip(questions['id'], questions['text']))

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

        # 生成缓存文件名
        # tokenizer_hash = md5(pickle.dumps(tokenizer)).hexdigest()[:8]
        cache_name = f"{mode}_mc{max_candidates}_ml{max_len}_tok{1000000}.pkl"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.tokenized_samples = pickle.load(f)
        else:
            if mode == 'train':
                self._process_train()
            else:
                self._process_eval()

            # 分批分词处理
            self.tokenized_samples = []
            for i in tqdm(range(0, len(self.samples), self.chunk_size),
                          desc=f"Tokenizing {mode} data in batches"):
                batch = self.samples[i:i + self.chunk_size]
                batch_tokenized = []

                for sample in batch:
                    query_text = self.qid_to_text[sample['question_id']]
                    candidate_ids = sample['candidates']
                    candidate_texts = [self.aid_to_text.get(aid, "[UNK]") for aid in candidate_ids]

                    # 分词处理
                    query_enc = self.tokenizer(
                        query_text,
                        max_length=self.max_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    cand_enc = self.tokenizer(
                        candidate_texts,
                        max_length=self.max_len,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )

                    # 转换为numpy数组节省内存
                    batch_tokenized.append({
                        'query_input_ids': query_enc['input_ids'].squeeze(0).numpy(),
                        'query_attention_mask': query_enc['attention_mask'].squeeze(0).numpy(),
                        'cand_input_ids': cand_enc['input_ids'].numpy(),
                        'cand_attention_mask': cand_enc['attention_mask'].numpy(),
                        'labels': sample['labels']
                    })

                self.tokenized_samples.extend(batch_tokenized)
                del batch, batch_tokenized
                gc.collect()

            # 保存处理结果
            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_samples, f)

    def __len__(self):
        # 添加防御性编程检查
        if not hasattr(self, 'tokenized_samples'):
            raise RuntimeError("Dataset not initialized properly")
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        item = self.tokenized_samples[idx]
        # 将numpy数组转换回Tensor
        return {
            'query_input_ids': torch.from_numpy(item['query_input_ids']),
            'query_attention_mask': torch.from_numpy(item['query_attention_mask']),
            'cand_input_ids': torch.from_numpy(item['cand_input_ids']),
            'cand_attention_mask': torch.from_numpy(item['cand_attention_mask']),
            'labels': torch.FloatTensor(item['labels'])
        }

    # 其余方法保持不变（_process_train, _process_eval等）
    # 注意在_process_train和_process_eval中增加分块读取逻辑

    def _process_train(self):
        """训练数据处理（优化内存使用）"""
        self.samples = []
        chunk_size = 10000  # 分块读取

        reader = pd.read_csv(
            f"{self.data_path}/train_candidates.txt",
            # f"{self.data_path}/train.txt",
            names=['question_id', 'pos_ans_id', 'neg_ans_id'],
            dtype={'question_id': str, 'pos_ans_id': str, 'neg_ans_id': str},
            sep=',',
            header=0,
            engine='python',
            on_bad_lines='warn',
            chunksize=chunk_size
        )

        for chunk in reader:
            # 过滤和处理数据
            chunk = chunk[
                chunk['question_id'].str.isdigit() &
                chunk['pos_ans_id'].str.isdigit() &
                chunk['neg_ans_id'].str.isdigit()
                ].dropna()

            for _, row in chunk.iterrows():
                self.samples.append({
                    'question_id': row['question_id'].strip(),
                    'candidates': [row['pos_ans_id'].strip(), row['neg_ans_id'].strip()],
                    'labels': [1, 0]
                })
            del chunk
            gc.collect()

    def _process_eval(self):
        """评估数据处理（优化内存使用）"""
        self.samples = []
        chunk_size = 10000
        #
        reader = pd.read_csv(
            f"{self.data_path}/{self.mode}_candidates.txt",
            # f"{self.data_path}/{self.mode}.txt",
            names=['question_id', 'ans_id', 'cnt', 'label'],
            dtype={'question_id': str, 'ans_id': str, 'cnt': int, 'label': int},
            sep=',',
            header=0,
            engine='python',
            on_bad_lines='warn',
            chunksize=chunk_size
        )

        for chunk in reader:
            chunk = chunk[
                chunk['question_id'].str.isdigit() &
                chunk['ans_id'].str.isdigit()
                ].dropna()

            for question_id, group in chunk.groupby('question_id'):
                candidates = group['ans_id'].tolist()[:self.max_cand]
                labels = group['label'].tolist()[:self.max_cand]
                pad_length = self.max_cand - len(candidates)

                if pad_length > 0:
                    candidates += ['0'] * pad_length
                    labels += [0] * pad_length

                self.samples.append({
                    'question_id': question_id,
                    'candidates': candidates,
                    'labels': labels
                })
            del chunk
            gc.collect()


# collate函数需要相应调整
def cmedqa_collate_fn(batch):
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
# 示例用法
if __name__ == "__main__":
    # 首次加载会生成缓存
    import pandas as pd

    # 处理训练集
    train_path = "data/wjh/graduate/AugData/cMedQA2/train_candidates.txt"
    q_df = pd.read_csv(
        train_path,
        sep=',',
        names=['question_id', 'pos_ans_id', 'neg_ans_id'],  # 显式指定列名
        dtype=str,
        header=None,  # 确保不将第一行作为标题
        on_bad_lines='skip'
    ).dropna()

    # 获取后200万条记录
    last_2m_rows = q_df.tail(2000000)

    # 将后200万条记录保存为csv文件
    last_2m_rows.to_csv(
        "data/wjh/graduate/AugData/cMedQA2/q.csv",
        index=False,  # 不保存索引列
        header=False  # 不生成标题行（保持与原文件一致）
    )

    # 处理验证集
    dev_path = "data/wjh/graduate/AugData/cMedQA2/dev_candidates.txt"
    v_df = pd.read_csv(
        dev_path,
        sep=',',
        names=['question_id', 'ans_id', 'cnt', 'label'],  # 假设开发集有4列
        dtype=str,
        header=None,
        on_bad_lines='skip',
        nrows=10000
    ).dropna()

    v_df.to_csv(
        "data/wjh/graduate/AugData/cMedQA2/v.csv",
        index=False,
        header=False
    )
