import csv

# 读取 SICK.txt 文件
train_data = []
val_data = []
test_data = []
with open('../data/wjh/graduate/AugData/SICK/SICK.txt', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    headers = next(reader)  # 读取表头
    for row in reader:
        if row[-1] == 'TRAIN':
            train_data.append(row)
        elif row[-1] == 'TRIAL':
            val_data.append(row)
        elif row[-1] == 'TEST':
            test_data.append(row)

# 保存训练集、验证集和测试集
def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)  # 写入表头
        writer.writerows(data)

save_data(train_data, '../data/wjh/graduate/AugData/SICK/train.txt')
save_data(val_data, '../data/wjh/graduate/AugData/SICK/val.txt')
save_data(test_data, '../data/wjh/graduate/AugData/SICK/test.txt')

print("数据集按官方划分完成！")