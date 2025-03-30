import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import os
import pickle
from hashlib import md5
from tqdm import tqdm

class CMEDQADataset(Dataset):
    """cMedQA2数据集专用处理类（支持批量处理）"""
    _qid_to_text = None
    _aid_to_text = None

    def __init__(self, data_path, mode='train', max_candidates=200, tokenizer=None, max_len=128, cache_dir=".cache", batch_size=32):
        self.mode = mode
        self.data_path = data_path
        self.max_cand = max_candidates
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = cache_dir
        self.batch_size = batch_size  # 批量处理大小
        os.makedirs(cache_dir, exist_ok=True)

        # === 加载问题和答案映射 ===#
        if CMEDQADataset._qid_to_text is None:
            q_path = os.path.join(data_path, "question.csv")
            questions = pd.read_csv(q_path, names=['id', 'text'], dtype={'id': str, 'text': str}, on_bad_lines='skip')
            CMEDQADataset._qid_to_text = dict(zip(questions['id'], questions['text']))

            a_path = os.path.join(data_path, "answer.csv")
            answers = pd.read_csv(a_path, names=['id', 'question_id', 'text'], dtype={'id': str, 'question_id': str, 'text': str}, on_bad_lines='skip')
            CMEDQADataset._aid_to_text = dict(zip(answers['id'], answers['text']))

        self.qid_to_text = CMEDQADataset._qid_to_text
        self.aid_to_text = CMEDQADataset._aid_to_text

        # === 加载缓存 ===#
        tokenizer_hash = md5(pickle.dumps(tokenizer)).hexdigest()[:8]
        cache_name = f"{mode}_mc{max_candidates}_ml{max_len}_bs{batch_size}_tok.pkl"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.tokenized_samples = pickle.load(f)
        else:
            self._process_data()
            self._batch_tokenize()
            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_samples, f)

    def _process_data(self):
        self.samples = []
        if self.mode == 'train':
            data_file = os.path.join(self.data_path, "train_candidates.txt")
            # data_file = os.path.join(self.data_path, "train.txt")
            data = pd.read_csv(data_file, names=['question_id', 'pos_ans_id', 'neg_ans_id'], dtype=str, sep=',', header=0, on_bad_lines='warn')
            data = data.dropna()
            for _, row in data.iterrows():
                self.samples.append({
                    'question_id': row['question_id'].strip(),
                    'candidates': [row['pos_ans_id'].strip(), row['neg_ans_id'].strip()],
                    'labels': [1, 0]
                })
        else:
            data_file = os.path.join(self.data_path, f"{self.mode}_candidates.txt")
            # data_file = os.path.join(self.data_path, f"{self.mode}.txt")
            data = pd.read_csv(data_file, names=['question_id', 'ans_id', 'cnt', 'label'], dtype={'question_id': str, 'ans_id': str, 'label': int}, sep=',', header=0, on_bad_lines='warn')
            data = data.dropna()
            for question_id, group in data.groupby('question_id'):
                candidates = group['ans_id'].tolist()
                labels = group['label'].tolist()
                self.samples.append({
                    'question_id': question_id,
                    'candidates': candidates[:self.max_cand],
                    'labels': labels[:self.max_cand]
                })

    def _batch_tokenize(self):
        self.tokenized_samples = []
        queries, all_candidates, all_labels = [], [], []

        for sample in self.samples:
            queries.append(self.qid_to_text[sample['question_id']])
            all_candidates.append([self.aid_to_text.get(ans_id, "[UNK]") for ans_id in sample['candidates']])
            all_labels.append(sample['labels'])

        for i in tqdm(range(0, len(queries), self.batch_size), desc="Tokenizing Data in Batches"):
            batch_queries = queries[i:i + self.batch_size]
            batch_candidates = all_candidates[i:i + self.batch_size]
            batch_labels = all_labels[i:i + self.batch_size]

            query_enc = self.tokenizer(batch_queries, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
            cand_enc = self.tokenizer([c for sublist in batch_candidates for c in sublist], max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
            cand_input_ids = cand_enc['input_ids'].view(len(batch_candidates), -1, self.max_len)
            cand_attention_mask = cand_enc['attention_mask'].view(len(batch_candidates), -1, self.max_len)

            for j in range(len(batch_queries)):
                self.tokenized_samples.append({
                    'query_input_ids': query_enc['input_ids'][j],
                    'query_attention_mask': query_enc['attention_mask'][j],
                    'cand_input_ids': cand_input_ids[j],
                    'cand_attention_mask': cand_attention_mask[j],
                    'labels': torch.FloatTensor(batch_labels[j])
                })

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        return self.tokenized_samples[idx]

    @classmethod
    def clear_cache(cls, cache_dir=".cache"):
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, f))

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

if __name__ == "__main__":
    train_set = CMEDQADataset("data/wjh/graduate/AugData/cMedQA2", mode='train', cache_dir=".cache", batch_size=32)
    dev_set = CMEDQADataset("data/wjh/graduate/AugData/cMedQA2", mode='dev', cache_dir=".cache", batch_size=32)
    CMEDQADataset.clear_cache()
