import pickle
import pandas as pd
import numpy as np
import random
from itertools import combinations

# 指定pickle文件路径
pickle_file_path = 'RML2016.10a_dict.pkl'

# 加载数据
with open(pickle_file_path, 'rb') as file:
    data_dict = pickle.load(file, encoding='latin1')

# 指定允许的调制类型
allowed_mods = ['AM-SSB', 'GFSK', 'PAM4', 'CPFSK', 'BPSK']
target_snr = 10  # 设置目标信噪比为10dB

# 筛选指定SNR和调制类型的数据（保留所有数据）
filtered_data = {}
for key, value in data_dict.items():
    mod_type, snr = key
    if snr == target_snr and mod_type in allowed_mods:
        if mod_type not in filtered_data:
            filtered_data[mod_type] = []
        filtered_data[mod_type].extend(value)  # 保留所有样本

# 生成所有可能的两两组合并分配标签
combinations_list = list(combinations(allowed_mods, 2))
sorted_combos = [tuple(sorted(combo)) for combo in combinations_list]
label_mapping = {combo: f"mix_{i + 1}" for i, combo in enumerate(sorted_combos)}


# 定义混合函数
def mix_samples(sample1, sample2, alpha=0.5):
    return alpha * sample1 + (1 - alpha) * sample2


# 创建原始数据的DataFrame
original_df = pd.DataFrame()
for mod_type, samples in filtered_data.items():
    # 将所有样本展平为二维数组
    samples_flat = np.array([sample.flatten() for sample in samples])
    # 创建原始数据的DataFrame
    temp_df = pd.DataFrame(samples_flat, columns=[f'Sample_{i}' for i in range(samples_flat.shape[1])])
    temp_df['Mod_Type'] = mod_type
    temp_df['SNR'] = target_snr
    original_df = pd.concat([original_df, temp_df], ignore_index=True)

# 创建混合数据的DataFrame
combined_df = pd.DataFrame()
for _ in range(20480):  # 生成20480个混合样本
    mod_type1, mod_type2 = random.sample(list(filtered_data.keys()), 2)
    combo_key = tuple(sorted([mod_type1, mod_type2]))
    mix_label = label_mapping[combo_key]

    # 随机选择两个样本进行混合
    sample1 = random.choice(filtered_data[mod_type1]).flatten()
    sample2 = random.choice(filtered_data[mod_type2]).flatten()

    mixed_sample = mix_samples(sample1, sample2, alpha=random.uniform(0.3, 0.7))

    # 创建混合样本的临时DataFrame
    temp_df = pd.DataFrame([mixed_sample], columns=[f'Sample_{i}' for i in range(len(mixed_sample))])
    temp_df['Mod_Type'] = mix_label
    temp_df['SNR'] = target_snr

    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# 合并原始数据和混合数据
final_df = pd.concat([original_df, combined_df], ignore_index=True)

# 保存到CSV文件
csv_file_path = f'mixed_output_data_snr_10dB.csv'
final_df.to_csv(csv_file_path, index=False)
print(f"Mixed CSV file saved: {csv_file_path}")

print("Data processing complete. Mixed CSV file saved.")