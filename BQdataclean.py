import pandas as pd
import os

# 文件路径配置
base_dir = 'data/wjh/graduate/AugData/BQ'
file_path1 = os.path.join(base_dir, 'train.tsv')  # 训练集
file_path2 = os.path.join(base_dir, 'dev.tsv')    # 验证集
file_path3 = os.path.join(base_dir, 'test.tsv')   # 测试集

# 读取CSV文件（使用逗号分隔）

df_train = pd.read_csv(file_path1, sep='\t', header=0, quoting=3)
df_dev = pd.read_csv(file_path2, sep='\t', header=0, quoting=3)
df_test = pd.read_csv(file_path3, sep='\t', header=0, quoting=2)
# 打印列名以验证
print("列名验证:", df_train.columns.tolist())


# 合并数据集并处理列名空格（可选）
# combined_df = pd.concat([df_train, df_dev], ignore_index=True)
# combined_df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
combined_df = pd.concat([df_train], ignore_index=True)
combined_df.columns = df_train.columns.str.strip()  # 去除列名前后空格

# 提取句子
sentences = []
for _, row in combined_df.iterrows():
    sentences.append(row["sentence1"].strip())
    sentences.append(row["sentence2"].strip())

# 去重并保存
unique_sentences = list(set(sentences))

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(unique_sentences))
print(f"成功生成语料库！唯一句子数量: {len(unique_sentences)}")


# 预览
print("\n预览前5条：")
for idx, s in enumerate(unique_sentences[:5], 1):
    print(f"{idx}. {s}")

