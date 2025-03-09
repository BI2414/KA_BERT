from rank_bm25 import BM25Okapi
import numpy as np
import jieba  # 中文分词工具（英文场景可替换为nltk）


class BM25Retriever:
    def __init__(self, corpus, tokenize_fn=None):
        """
        BM25粗筛器初始化
        :param corpus: 知识库文档列表，格式为["文本1", "文本2", ...]
        :param tokenize_fn: 自定义分词函数，默认使用jieba分词
        """
        self.corpus = corpus
        self.tokenize_fn = tokenize_fn or self._default_tokenizer
        self._build_index()

    def _default_tokenizer(self, text):
        """默认中文分词器"""
        return [word for word in jieba.cut(text) if word.strip()]

    def _build_index(self):
        """构建BM25索引"""
        # 知识库文档分词
        self.tokenized_corpus = [self.tokenize_fn(doc) for doc in self.corpus]

        # 初始化BM25模型
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 预计算文档长度（用于后续改进）
        self.doc_lengths = np.array([len(d) for d in self.tokenized_corpus])

    def search(self, query, top_k=100, return_scores=False):
        """
        BM25粗筛查询
        :param query: 查询文本
        :param top_k: 返回候选数量
        :param return_scores: 是否返回分数
        """
        # 查询文本分词
        tokenized_query = self.tokenize_fn(query)

        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取Top-K候选索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 构建返回结果
        results = []
        for idx in top_indices:
            item = {
                "doc_id": idx,
                "text": self.corpus[idx],
                "score": scores[idx]
            }
            results.append(item)

        return (results, scores) if return_scores else results

    def save_index(self, path):
        """保存索引（用于后续STAMP模型集成）"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                'doc_lengths': self.doc_lengths
            }, f)

    @classmethod
    def load_index(cls, path, tokenize_fn=None):
        """加载预建索引"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.corpus = data['corpus']
        instance.tokenized_corpus = data['tokenized_corpus']
        instance.doc_lengths = data['doc_lengths']
        instance.bm25 = BM25Okapi(instance.tokenized_corpus)
        instance.tokenize_fn = tokenize_fn or instance._default_tokenizer
        return instance


# 示例用法 -------------------------------------------------
if __name__ == "__main__":
    # 示例知识库（实际应替换为真实数据）
    corpus = [
        "如何办理信用卡",
        "信用卡年费是多少",
        "信用卡逾期如何处理",
        "储蓄卡开户流程",
        "网上银行登录问题"
    ]

    # 初始化BM25检索器
    retriever = BM25Retriever(corpus)

    # 执行查询
    query = "怎么办信用卡"
    results = retriever.search(query, top_k=2)

    # 打印结果
    print("BM25粗筛结果：")
    for res in results:
        print(f"[Doc {res['doc_id']}] Score: {res['score']:.2f}\tText: {res['text']}")