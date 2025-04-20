# CNN-test
我创建这个文档是为了我的毕业设计

首先基于final-dataset.py从RML2016.10a_dict.pkl中读取SNR=10dB的5种信号，并进行随机混合生成总数为5000个样本的15种信号，
其中包含了5种原始信号allowed_mods = ['AM-SSB', 'GFSK', 'PAM4', 'CPFSK', 'BPSK']，和10种混合信号mix_1 to mix_10;
最终生成csv_file_path = f'mixed_output_data_snr_10dB.csv' 
再使用augment.py进行数据增强，增强方式包括高斯噪声注入、时间偏移、幅度缩放和随机反转，对样本量进行增加，达到25000个样本。
augment_data('mixed_output_data_snr_10dB.csv', 'augmented_data_5x.csv')

基于CNN-test1.py模型和augmented_data_5x.csv进行训练，得到best_model.pth
