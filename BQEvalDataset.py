class EvalDataset(Dataset):
    """用于模拟真实召回场景的验证数据集"""

    def __init__(self, data_path, tokenizer, mode='dev',
                 max_len=128, candidate_pool_size=10000):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载基础数据
        self.data = self._load_base_data(data_path, mode)

        # 构建全局候选池
        self.global_candidates = self._build_global_pool(candidate_pool_size)

    def _load_base_data(self, path, mode):
        """加载原始标注数据"""
        file_path = os.path.join(path, f"{mode}.tsv")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            header = f.readline().strip().split('\t')
            if header != ['query', 'candidate', 'label']:
                f.seek(0)

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query, candidate, label = parts
                    data.append((query, candidate, int(label)))

        # 按query聚合
        query_dict = {}
        for q, c, l in data:
            if q not in query_dict:
                query_dict[q] = {'pos': [], 'neg': []}
            query_dict[q]['pos'].append(c) if l == 1 else query_dict[q]['neg'].append(c)

        return [
            {'query': q, 'pos': v['pos'], 'neg': v['neg']}
            for q, v in query_dict.items()
            if len(v['pos']) > 0  # 至少有一个正样本
        ]

    def _build_global_pool(self, pool_size):
        """构建全局负样本池"""
        all_negatives = []
        for item in self.data:
            all_negatives.extend(item['neg'])

        # 去重并采样
        unique_negs = list(set(all_negatives))
        return random.sample(unique_negs, min(pool_size, len(unique_negs)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]