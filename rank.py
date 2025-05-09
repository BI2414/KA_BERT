# rank.py
import torch
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import normalize  # 新增归一化函数导入
from src.config import get_argparse
from tqdm import tqdm  # 新增

class BertRanker:
    def __init__(self, model_path, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用设备: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(args["model"])
        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.args = args

    def encode(self, texts):
        batch_size = self.args["test_batch_size"]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            token_type_ids = inputs["token_type_ids"].to(self.device)

            with torch.no_grad():
                reps = self.model.get_encoded_representation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                reps = reps.cpu().numpy()
                reps = normalize(reps, axis=1)
                embeddings.extend(reps)
        return embeddings

    def rank_candidates(self, query, candidates):
        if not candidates:
            return {
                "candidates": [],
                "scores": []
            }
        query_embedding = self.encode([query])[0]
        candidate_embeddings = self.encode(candidates)
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        sorted_indices = similarities.argsort()[::-1]
        return {
            "candidates": [candidates[i] for i in sorted_indices],
            "scores": similarities[sorted_indices].tolist()
        }

def load_ground_truth(filepath):
    import pandas as pd
    from collections import defaultdict

    df = pd.read_csv(filepath, sep="\t", quoting=3, on_bad_lines='skip')

    # 初始化并查集
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # 构建等价类
    for _, row in df[df["label"] == 1].iterrows():
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        union(s1, s2)

    # 按 root 分组
    groups = defaultdict(set)
    for s in parent:
        root = find(s)
        groups[root].add(s)

    # 构建 ground_truth 映射：每个句子对应它等价类中除自己外的所有句子
    ground_truth = defaultdict(list)
    for group in groups.values():
        for s in group:
            ground_truth[s] = [other for other in group if other != s]

    return ground_truth


def calculate_metrics(ranked_results, ground_truth):
    total = len(ranked_results)
    recall_at_10 = 0
    mrr = 0.0

    for item in ranked_results:
        query = item["query"]
        candidates = item["sorted_candidates"][:10]
        true_answers = ground_truth.get(query, [])

        found = any(cand in true_answers for cand in candidates)
        recall_at_10 += int(found)

        for rank, cand in enumerate(item["sorted_candidates"], 1):
            if cand in true_answers:
                mrr += 1.0 / rank
                break

    return {
        "Recall@10": recall_at_10 / total,
        "MRR": mrr / total
    }


# 加载召回结果并排序
if __name__ == "__main__":
    recall_file = "recall_results.json"
    ground_truth_file = "data/wjh/graduate/AugData/BQ/train.tsv"
    model_path = "data/wjh/graduate/data/save/bert_base_BQ.pt"

    args = get_argparse().parse_args()
    args = vars(args)

    # 加载召回结果
    with open("recall_results.json", "r", encoding="utf-8") as f:
        recall_results = json.load(f)

    # 计算 BM25+Bool 阶段指标
    bm25_results = []
    for item in recall_results:
        bm25_results.append({
            "query": item["query"],
            "sorted_candidates": item["candidates"]  # 注意：这里用原始 top100 排序
        })

    ground_truth = load_ground_truth(ground_truth_file)
    bm25_metrics = calculate_metrics(bm25_results, ground_truth)
    print("BM25 Recall@10:", bm25_metrics["Recall@10"])
    print("BM25 MRR:", bm25_metrics["MRR"])

    # 计算 BERT rerank 阶段指标
    ranker = BertRanker(model_path, args)
    ranked_results = []
    for item in tqdm(recall_results, desc="Ranking queries"):
        ranked = ranker.rank_candidates(item["query"], item["candidates"])
        ranked_results.append({
            "query": item["query"],
            "sorted_candidates": ranked["candidates"]
        })

    with open("ranked_results.json", "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, ensure_ascii=False, indent=2)

    # 加载召回结果
    with open("ranked_results.json", "r", encoding="utf-8") as f:
        ranked_results = json.load(f)

    bert_metrics = calculate_metrics(ranked_results, ground_truth)
    print("BERT Recall@10:", bert_metrics["Recall@10"])
    print("BERT MRR:", bert_metrics["MRR"])
