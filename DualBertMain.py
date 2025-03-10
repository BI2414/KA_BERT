import sys
sys.dont_write_bytecode = True
import argparse
import os
import nni
import time
import pickle
import random
import setproctitle
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from transformers import (AdamW, AlbertTokenizer, AlbertModel,
                          get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, BertTokenizer, BertConfig)

from src.config import get_argparse
import json
from src.metrics import flat_accuracy, flat_f1
from model import NewBert
from logger import logger
from src.utils import read_examples
from src.utils import convert_examples_to_features
import DualBert
from sklearn.metrics import roc_auc_score
from BQPairwiseDataset import BQPairwiseDataset


args = get_argparse().parse_args()
args = vars(args)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 禁止并行
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
procname = str(args["name"]) + "test" if args["test"] else str(args["name"])
setproctitle.setproctitle('wjh_{}'.format(procname))

# 定义gpu设备
# device = torch.cuda.current_device()
# args["device"] = device
# print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args['device'] =device
# print(device)  # 输出当前设备
def collate_fn(batch, tokenizer, max_len=128):
    """自定义批次处理"""
    batch_queries = [item['query'] for item in batch]
    batch_candidates = [item['candidates'] for item in batch]
    batch_labels = [item['labels'] for item in batch]
    # 添加维度验证
    assert all(len(x['labels']) == (args.num_pos + args.num_neg) for x in batch)
    "候选数量不一致"
    # 编码查询
    query_enc = tokenizer(
        batch_queries,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 编码候选（三维结构：[batch, num_cand, seq_len]）
    flat_candidates = [c for sublist in batch_candidates for c in sublist]
    cand_enc = tokenizer(
        flat_candidates,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 重组为三维结构
    num_cand = len(batch_candidates[0])
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
        'labels': torch.FloatTensor(batch_labels)
    }


def train(model, train_loader, val_loader, args):
    """训练函数"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_score = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # 训练阶段
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # 数据迁移到设备
            query_inputs = {
                'input_ids': batch['query']['input_ids'].to(args.device),
                'attention_mask': batch['query']['attention_mask'].to(args.device)
            }
            doc_inputs = {
                'input_ids': batch['candidates']['input_ids'].to(args.device),
                'attention_mask': batch['candidates']['attention_mask'].to(args.device)
            }
            labels = batch['labels'].to(args.device)

            # 前向传播
            scores, mu, logvar = model(query_inputs, doc_inputs)  # 接收mu和logvar

            # 计算损失
            loss = model.compute_loss(scores, labels, mu, logvar)  # 传递参数
            loss.backward()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # 验证阶段
        val_metrics = evaluate(model, val_loader, args)

        # 保存最佳模型
        if val_metrics['recall@10'] > best_score:
            best_score = val_metrics['recall@10']
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pt")

        # 打印日志
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Val Recall@10: {val_metrics['recall@10']:.4f}")
        print(f"Val MRR@10: {val_metrics['mrr@10']:.4f}\n")


def evaluate(model, data_loader, args, top_k=(1, 5, 10)):
    """评估函数"""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            query_inputs = {
                'input_ids': batch['query']['input_ids'].to(args.device),
                'attention_mask': batch['query']['attention_mask'].to(args.device)
            }
            doc_inputs = {
                'input_ids': batch['candidates']['input_ids'].to(args.device),
                'attention_mask': batch['candidates']['attention_mask'].to(args.device)
            }

            # 计算相似度
            scores = model(query_inputs, doc_inputs).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels)

    # 合并结果
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 计算指标
    metrics = {}
    for k in top_k:
        recall, mrr = calculate_metrics(scores, labels, k)
        metrics[f'recall@{k}'] = recall
        metrics[f'mrr@{k}'] = mrr

    return metrics


def calculate_metrics(scores, labels, top_k):
    """计算Recall@K和MRR@K"""
    recall = 0
    mrr = 0
    for i in range(scores.shape[0]):
        # 获取每个query的排序结果
        sorted_indices = np.argsort(-scores[i])
        relevant = np.where(labels[i][sorted_indices] == 1)[0]

        # Recall@K
        if len(relevant) > 0 and relevant[0] < top_k:
            recall += 1

        # MRR@K
        if len(relevant) > 0:
            first_relevant = relevant[0] + 1  # 位置从1开始计数
            mrr += 1 / first_relevant if first_relevant <= top_k else 0

    recall = recall / scores.shape[0]
    mrr = mrr / scores.shape[0]
    return recall, mrr


def test(model, test_loader, args):
    """测试函数（与evaluate相同，增加结果保存）"""
    metrics = evaluate(model, test_loader, args)

    # 保存结果
    with open(f"{args.save_dir}/test_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    # 参数配置
    class Args:
        data_path = "data/wjh/graduate/AugData/BQ"
        model_name = "data/wjh/graduate/data/bert-base-chinese"
        save_dir = "data/wjh/graduate/data/save"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 32
        num_pos = 2
        num_neg = 6
        max_len = 128
        lr = 2e-5
        epochs = 5
        embed_dim = 128
        share_low_layers = 6
        use_gate = True
        kl_weight = 0.1
        adapter_size = 64  # Adapter的中间维度
        use_cross_attn = True  # 启用交叉注意力
        contrastive_margin = 0.2  # 对比损失边界


    args = Args()

    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = DualBert.EnhancedDualBERT.from_pretrained(
        args.model_name,
        config=BertConfig.from_pretrained(args.model_name),
        args=args
    ).to(args.device)

    # 数据加载
    train_dataset = BQPairwiseDataset(args.data_path, tokenizer, mode='train')
    val_dataset = BQPairwiseDataset(args.data_path, tokenizer, mode='dev')
    # test_dataset = BQPairwiseDataset(args.data_path, tokenizer, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len)
    )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len)
    # )

    # 训练流程
    train(model, train_loader, val_loader, args)

    # 最终测试
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt"))
    # test(model, test_loader, args)