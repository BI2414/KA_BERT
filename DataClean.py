import os
import pandas as pd
from collections import defaultdict
import pickle

def save_query_dict(query_dict, file_path):
    """
    将 query_dict 存储到本地
    :param query_dict: 重组后的数据字典
    :param file_path: 存储路径（如 'data/query_dict.pkl'）
    """
    with open(file_path, 'wb') as f:
        pickle.dump(query_dict, f)

def load_query_dict(file_path):
    """
    从本地加载 query_dict
    :param file_path: 存储路径（如 'data/query_dict.pkl'）
    :return: 重组后的数据字典
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_local_bq_data(data_dir):
    """
    从本地加载 BQ 数据集
    :param data_dir: 数据集目录（包含 train.tsv, dev.tsv, test.tsv）
    :return: 重组后的数据字典 {query: [(candidate1, score1), (candidate2, score2), ...]}
    """
    # 定义文件路径

    # train_path = os.path.join(data_dir, 'train.tsv')
    # dev_path = os.path.join(data_dir, 'dev.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')

    # 加载数据
    train_data = pd.read_csv(test_path, sep='\t', header=0, quoting=3)
    # 重组为 {query: [(candidate, score), ...]} 格式
    query_dict = defaultdict(list)
    for _, row in train_data.iterrows():
        query = row['sentence1']
        candidate = row['sentence2']
        score = row['label']
        query_dict[query].append((candidate, score))

    return query_dict

if __name__ == '__main__':
    # 示例调用
    data_dir = 'data/wjh/graduate/AugData/BQ'  # 替换为您的实际路径
    query_dict = load_local_bq_data(data_dir)

    # 打印示例
    for query, candidates in list(query_dict.items())[:2]:
        print(f"Query: {query}")
        for cand, score in candidates:
            print(f"  Candidate: {cand}, Score: {score}")

    save_query_dict(query_dict, 'data/wjh/graduate/recallData/test.pkl')  # 存储
    # query_dict = load_query_dict('data/query_dict.pkl')  # 加载