import pandas as pd
import numpy as np


def augment_data(input_file, output_file):
    # 读取原始数据
    df = pd.read_csv(input_file)

    # 分离特征和元数据
    features = df.filter(like='Sample_').values
    metadata = df[['Mod_Type', 'SNR']]

    # 定义增强函数
    def add_gaussian_noise(samples):
        """添加高斯噪声保持SNR=10dB"""
        signal_power = np.mean(samples ** 2, axis=1, keepdims=True)
        noise_power = signal_power / (10 ** (10 / 10))  # SNR=10dB
        noise = np.random.normal(0, np.sqrt(noise_power), samples.shape)
        return samples + noise

    def time_shift(samples):
        """时间偏移"""
        shifts = np.random.randint(-samples.shape[1] // 4, samples.shape[1] // 4, size=samples.shape[0])
        augmented = np.empty_like(samples)
        for i in range(samples.shape[0]):
            augmented[i] = np.roll(samples[i], shifts[i])
        return augmented

    def amplitude_scaling(samples):
        """幅度缩放"""
        scales = np.random.uniform(0.9, 1.1, size=(samples.shape[0], 1))
        return samples * scales

    def random_flip(samples):
        """随机反转"""
        flips = np.random.rand(samples.shape[0]) < 0.5
        augmented = samples.copy()
        augmented[flips] = augmented[flips][:, ::-1]
        return augmented

    # 应用增强
    augmentations = [
        add_gaussian_noise(features),
        time_shift(features),
        amplitude_scaling(features),
        random_flip(features)
    ]

    # 合并所有数据
    all_features = np.concatenate([features] + augmentations, axis=0)
    all_metadata = pd.concat([metadata] * 5, ignore_index=True)

    # 创建增强后的DataFrame
    augmented_df = pd.DataFrame(
        all_features,
        columns=df.filter(like='Sample_').columns
    )
    augmented_df = pd.concat([augmented_df, all_metadata], axis=1)

    # 保存结果
    augmented_df.to_csv(output_file, index=False)
