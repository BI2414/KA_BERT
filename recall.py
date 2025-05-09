import jieba
import math
from collections import defaultdict
import pandas as pd
import json
def _load_stopwords(file_path="stopwords.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

class ImprovedBM25WithBoolean:
    def __init__(self, corpus):
        self.corpus = corpus  # 语料库，格式为列表，每个元素为文档字符串
        self.preprocessed_corpus = self._preprocess_corpus()
        self.vocab = self._build_vocab()
        self.doc_count = len(corpus)
        self.avg_dl = sum(len(doc) for doc in self.preprocessed_corpus) / self.doc_count
        self.k1 = 1.2
        self.b = 0.75
        self.nt = self._compute_term_frequencies()  # 词项总频次（用于IWF）
        self.total_terms = sum(self.nt.values())     # 语料库总词数
        self.inverted_index = self._build_inverted_index()

    def _preprocess_corpus(self):
        """预处理：分词、去停用词"""
        stopwords = _load_stopwords()  # 替换原有简单停用词
        processed = []
        for doc in self.corpus:
            words = list(jieba.cut(doc))
            words = [w for w in words if w not in stopwords]
            processed.append(words)
        return processed

    def _build_vocab(self):
        """构建词汇表"""
        vocab = set()
        for doc in self.preprocessed_corpus:
            vocab.update(doc)
        return list(vocab)

    def _compute_term_frequencies(self):
        """统计每个词在语料库中的总频次（nt_i）"""
        nt = defaultdict(int)
        for doc in self.preprocessed_corpus:
            for word in doc:
                nt[word] += 1
        return nt

    def _build_inverted_index(self):
        """构建倒排索引"""
        inverted_index = defaultdict(list)
        for doc_id, doc in enumerate(self.preprocessed_corpus):
            for word in set(doc):
                inverted_index[word].append(doc_id)
        return inverted_index

    def _compute_iwf(self, word):
        """计算改进的逆词频（IWF）"""
        nt_i = self.nt.get(word, 0)
        return math.log((self.total_terms - nt_i + 0.5) / (nt_i + 0.5))

    def _bm25_score(self, query, doc_id):
        """计算改进BM25的分数"""
        doc = self.preprocessed_corpus[doc_id]
        dl = len(doc)
        K = self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
        score = 0.0
        for word in query:
            tf = doc.count(word)
            if tf == 0:
                continue
            iwf = self._compute_iwf(word)
            numerator = tf * (self.k1 + 1)
            denominator = tf + K
            score += iwf * (numerator / denominator)
        return score

    def _boolean_search(self, query, logic="OR"):
        terms = [word for word in jieba.cut(query) if word in self.vocab]
        if not terms:
            return []
        # OR逻辑：任一term匹配即可
        doc_ids = set()
        for term in terms:
            doc_ids.update(self.inverted_index.get(term, []))
        return list(doc_ids)

    def hybrid_search(self, query, top_k=10, alpha=0.5):
        """混合检索：BM25 + 布尔检索"""
        # 布尔检索
        boolean_docs = self._boolean_search(query)
        boolean_scores = {doc_id: 1.0 for doc_id in boolean_docs}

        # BM25检索
        query_terms = list(jieba.cut(query))
        bm25_scores = {}
        for doc_id in range(self.doc_count):
            score = self._bm25_score(query_terms, doc_id)
            if score > 0:
                bm25_scores[doc_id] = score

        # 合并结果（加权求和）
        combined_scores = defaultdict(float)
        for doc_id in boolean_scores:
            combined_scores[doc_id] += alpha * boolean_scores[doc_id]
        for doc_id in bm25_scores:
            combined_scores[doc_id] += (1 - alpha) * bm25_scores[doc_id]

        # 去重并按分数排序
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        # ========== 新增去重逻辑 ==========
        seen_texts = set()  # 用于文本去重
        filtered_docs = []

        for doc_id, score in sorted_docs:
            candidate = self.corpus[doc_id]
            # 过滤与query完全相同的候选
            if candidate == query:
                continue
            # 去重相似候选（可选）
            if candidate not in seen_texts:
                seen_texts.add(candidate)
                filtered_docs.append(doc_id)
            if len(filtered_docs) >= top_k:
                break

        return filtered_docs[:top_k]
# 在文件顶部加载语料库
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]
model = ImprovedBM25WithBoolean(corpus)  # 全局模型

def get_query_results(query, top_k=50):
    doc_ids = model.hybrid_search(query, top_k=top_k)
    return {
        "query": query,
        "candidates": [corpus[doc_id] for doc_id in doc_ids],
        "doc_ids": doc_ids
    }

# 示例用法
# 示例：批量处理所有查询
if __name__ == "__main__":
    # 加载测试集查询（假设测试集为test.tsv）
    test_df = pd.read_csv("data/wjh/graduate/AugData/BQ/sampled_recall_test_data.tsv", sep="\t")
    test_queries = test_df["sentence1"].unique().tolist()  # 获取所有唯一查询

    all_recall_results = []
    for query in test_queries:
        result = get_query_results(query, top_k=500)
        all_recall_results.append(result)

    # 保存召回结果（供排序阶段使用）

    with open("recall_results.json", "w", encoding="utf-8") as f:
        json.dump(all_recall_results, f, ensure_ascii=False)

    # query = "为什么借款后一直没有给我回拨电话"
    # results = get_query_results(query, top_k=150)
    # print("检索结果（文档ID）:", results)
    # print("匹配文档:", [corpus[doc_id] for doc_id in results])