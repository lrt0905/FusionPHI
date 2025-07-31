import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':
    file1 = pd.read_csv("case_phage_motif_tensor.csv")
    file2 = pd.read_csv("case_host_motif_tensor_mixed.csv")

    file3 = pd.concat([file1,file2]) 

    # 提取identifier列和数值列并转换为目标格式
    result = []
    for index,row in file3.iterrows():
        identifier = row["Identifier"]
        tensor_values = torch.tensor(row.drop('Identifier').values.astype(float))
        result.append((identifier,tensor_values))

    data_pos_neg = pd.read_csv('case_phage_host_pairs.csv')

    # 创建一个字典，键为标识符，值为对应的特征向量
    identifier_to_tensor = {identifier: tensor for identifier, tensor in result}
    for key,value in identifier_to_tensor.items():
        print(f"Key:{key},Value:{value}")
    # 初始化一个新的列表，用于存储合并后的数据
    merged_data = []
    #
    # 遍历噬菌体-宿主相互作用对。
    # iterrows()是一个迭代器，它返回DataFrame的索引和每一行数据作为元组，其中index是当前行的索引，row是当前行的数据，通常以Pandas Series的形式返回。
    for index, row in data_pos_neg.iterrows():
        phage_id = row['phage_accession_id']
        host_id = row['host_species_name']
        label = row['label']

        phage_tensor = identifier_to_tensor.get(phage_id, torch.zeros(640))
        host_tensor = identifier_to_tensor.get(host_id, torch.zeros(640))

        phage_vector = phage_tensor.numpy()
        host_vector = host_tensor.numpy()

        current_row = [phage_id, host_id, label]
        for value in np.nditer(phage_vector):
            current_row.append(value)
        for value in np.nditer(host_vector):
            current_row.append(value)
        # 将当前行添加到merged_data列表中
        merged_data.append(current_row)
    for one in merged_data:
        print(one)
    # 将合并数据转换为Pandas DataFrame
    merged_df = pd.DataFrame(merged_data)
    # 定义列名
    column_names = ['phage_id', 'host_id', 'label'] + [f'phage_feature_{i}' for i in range(640)] + [f'host_feature_{i}' for i in range(640)]
    # 设置列名
    merged_df.columns = column_names
    # 导出为CSV文件
    merged_df.to_csv('motif_feature_mixed.csv', index=False)