import csv

def load_true_labels(text_file):
    """
    从 text.txt 文件中加载真实标签。

    参数:
        text_file (str): text.txt 文件的路径。

    返回:
        list: 真实标签列表（数字编码：2=ENTAILMENT, 1=NEUTRAL, 0=CONTRADICTION）。
    """
    true_labels = []
    with open(text_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            label = row['entailment_label'].strip().upper()  # 获取真实标签并转换为大写
            if label == 'ENTAILMENT':
                true_labels.append(2)
            elif label == 'NEUTRAL':
                true_labels.append(1)
            elif label == 'CONTRADICTION':
                true_labels.append(0)
            else:
                raise ValueError(f"未知标签: {label}")
    return true_labels

def calculate_accuracy(prediction_file, true_labels):
    """
    计算 SICK.tsv 文件的准确率。

    参数:
        prediction_file (str): 预测文件的路径（SICK.tsv）。
        true_labels (list): 真实标签列表。

    返回:
        float: 准确率。
    """
    correct_predictions = 0
    total_samples = 0

    with open(prediction_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')  # 使用 DictReader 按列名访问数据
        for row, true_label in zip(reader, true_labels):
            predicted_label = int(row['prediction'])  # 预测标签
            if predicted_label == true_label:
                correct_predictions += 1
            total_samples += 1

    if total_samples == 0:
        raise ValueError("文件为空或没有样本。")

    accuracy = correct_predictions / total_samples
    return accuracy

# 示例调用
text_file = 'data/wjh/graduate/AugData/SICK/test.txt'  # 真实标签文件
prediction_file = 'data/wjh/graduate/data/submit/SICK.tsv'  # 预测文件

# 加载真实标签
true_labels = load_true_labels(text_file)

# 计算准确率
accuracy = calculate_accuracy(prediction_file, true_labels)
print(f"准确率: {accuracy:.4f}")