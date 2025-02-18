from datasets import load_dataset
import numpy as np

# 加载 GLUE MRPC 数据集
dataset = load_dataset("glue", "mrpc")

from transformers import BertTokenizer

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义分词函数
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True, max_length=128)

# 对数据集进行分词处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 删除不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

from transformers import BertForSequenceClassification

# 加载 BERT 模型用于句子分类任务
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


from torch.utils.data import DataLoader

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16)


import torch
from torch.optim import AdamW

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练函数
def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# 训练模型
epochs = 3
for epoch in range(epochs):
    avg_train_loss = train_model(model, train_loader, optimizer)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}')


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 评估函数
def evaluate_model_with_print(model, eval_loader, raw_dataset):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_ids = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.append(label_ids.cpu().numpy())

            if i == 0:
                prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                true_label = label_ids.cpu().numpy()[0]
                print(f"Sentence 1: {raw_dataset[i]['sentence1']}")
                print(f"Sentence 2: {raw_dataset[i]['sentence2']}")
                print(f"True label: {true_label}")
                print(f"Predicted label: {prediction}")

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    return accuracy, precision, recall, f1

# 运行评估
accuracy, precision, recall, f1 = evaluate_model_with_print(model, eval_loader, dataset['validation'])
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# 保存模型和分词器
model.save_pretrained('bert-text-matching-model')
tokenizer.save_pretrained('bert-text-matching-model')
