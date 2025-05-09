import pandas as pd
import os

# 文件路径配置
base_dir = 'data/wjh/graduate/AugData/BQ'
train_path = os.path.join(base_dir, 'train.tsv')  # 训练集路径
output_path = os.path.join(base_dir, 'sampled_recall_test_data.tsv')  # 抽样结果保存路径

# ========== 核心代码 ==========
try:
    # 1. 读取训练数据
    df_train = pd.read_csv(train_path, sep='\t', quoting=3)

    # 2. 筛选label=1的数据
    label1_df = df_train[df_train['label'] == 1]

    # 3. 检查数据量是否足够
    if len(label1_df) < 2300:
        print(f"警告：训练集中只有 {len(label1_df)} 组label=1的数据，不足5000组！")
        print("将使用全部可用数据：", len(label1_df))
        sample_size = len(label1_df)
    else:
        sample_size = 2300

    # 4. 随机抽样 (设置随机种子保证可复现)
    sampled_df = label1_df.sample(n=sample_size, random_state=24, replace=False)

    # 5. 保存抽样结果
    sampled_df.to_csv(output_path, sep='\t', index=False, quoting=3)
    print(f"成功生成抽样数据！保存路径：{output_path}")
    print("抽样数据统计信息：")
    print("-" * 40)
    print(sampled_df.info())
    print("\n前5行样例：")
    print(sampled_df.head())

except FileNotFoundError:
    print(f"错误：训练集文件 {train_path} 不存在！")
except KeyError as e:
    print(f"数据列错误：{str(e)}，请检查数据是否包含 'label' 列！")
except Exception as e:
    print(f"未知错误：{str(e)}")