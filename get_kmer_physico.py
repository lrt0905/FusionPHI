import logging
import os
import pandas as pd
from collections import Counter, defaultdict
from itertools import product
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np


# 辅助函数，用于清理蛋白质序列
def clean_protein_sequence(sequence):
    cleaned_sequence = ''.join('A' if aa not in 'ACDEFGHIKLMNPQRSTVWY' else aa for aa in sequence)
    return cleaned_sequence


# 定义从类 FASTA 格式的 txt 文件中读取序列的函数
def read_fasta_like(file_path):
    sequences = {}
    current_id = None
    seq_lines = []

    with open(file_path, 'r') as file: # 打开protein_dir下的文件
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(seq_lines).replace(' ', '').replace('\r', '')
                    seq_lines = []
                current_id = line[1:] # 存储描述信息
            else:
                seq_lines.append(line.replace(' ', '').replace('\r', ''))
        # 添加最后一个序列
        if current_id:
            sequences[current_id] = ''.join(seq_lines)

    return sequences

def calculate_protein_features(sequence):
    """
    计算输入序列的蛋白质特征
    :param sequence: 氨基酸序列
    :return: features字典，包含20个AAC+7个物理化学性质指标
    """
    # 如果序列为空，返回所有特征值为0的字典
    if len(sequence) == 0:
        default_features = {
            'Sequence Length': 0,
            'Isoelectric Point': 0.0,
            'Molecular Weight': 0.0,
            'Aromaticity': 0.0,
            'Instability Index': 0.0,
            'Flexibility Avg': 0.0,
            'Average Hydropathy (GRAVY)': 0.0
        }
        for aa in "ACDEFGHIKLMNPQRSTVWY":  # 标准氨基酸
            default_features[f"AAC_{aa}"] = 0.0
        return default_features

    # 对于非空序列，继续计算实际特征
    sequence = clean_protein_sequence(sequence)
    protein_analysis = ProteinAnalysis(sequence)
    sequence_length = len(sequence)
    amino_acid_composition = protein_analysis.get_amino_acids_percent() # 氨基酸组成
    isoelectric_point = protein_analysis.isoelectric_point()    # 等电点
    molecular_weight = protein_analysis.molecular_weight()      # 分子量
    aromaticity = protein_analysis.aromaticity()    # 芳香性
    instability_index = protein_analysis.instability_index()    # 稳定性

    # 确保灵活性至少有一个值，以避免短序列导致的问题
    try:
        flexibility = protein_analysis.flexibility()
    except Exception as e:
        logging.warning(f"Failed to calculate flexibility for sequence {sequence}: {e}")
        flexibility = [0.0]  # 设置默认值

    gravy = protein_analysis.gravy()

    features = {
        'Sequence Length': sequence_length,
        'Isoelectric Point': isoelectric_point,
        'Molecular Weight': molecular_weight,
        'Aromaticity': aromaticity,
        'Instability Index': instability_index,
        'Flexibility Avg': np.mean(flexibility) if flexibility else 0.0,
        'Average Hydropathy (GRAVY)': gravy
    }

    for aa, percent in amino_acid_composition.items():
        features[f"AAC_{aa}"] = percent

    return features

# 计算k-mers特征
def build_kmers(seq, k_size):
    kmers = [seq[i:i + k_size] for i in range(len(seq) - k_size + 1)]
    return kmers


def summary_kmers(kmers, k_size=4):
    kmers_stat = dict(Counter(kmers))
    total = sum(kmers_stat.values())

    all_possible_kmers = [''.join(p) for p in product('ACGT', repeat=k_size)]
    frequency_dict = defaultdict(int, {kmer: 0 for kmer in all_possible_kmers})

    for kmer, count in kmers_stat.items():
        if len(kmer) == k_size and set(kmer).issubset(set('ACGT')):
            frequency_dict[kmer] = count / total

    return frequency_dict


def calculate_kmer_features(file_path, kmer_size):
    sequences = SeqIO.to_dict(SeqIO.parse(file_path, "fasta"))
    kmer_features = {}

    for seq_id, record in sequences.items():
        seq = str(record.seq).upper()
        kmers = build_kmers(seq, kmer_size)
        kmer_freq = summary_kmers(kmers, kmer_size)
        kmer_features[seq_id] = kmer_freq

    return kmer_features

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():# 注：本代码在计算k-mer频率特征的时候请使用完整的只有一行描述信息的全基因组序列文件， 计算motif和蛋白质特征的时候可以使用包含多条orf的序列文件
    protein_dir = 'protein_mixed'
    data_file = "case_phage_host_pairs.csv"
    csv_file = 'case_kmer_protein_features_mixed.csv'
    kmer_size = 4  # 设置k-mer大小
    phage_dir = 'case_phage_no_orf'
    host_dir = 'case_host_no_orf'
    isHeader = True

    # 检查CSV文件是否存在并且不为空
    header_written = False
    existing_columns = set()  # 初始化列集合

    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            existing_data_df = pd.read_csv(csv_file)
            header_written = True
            existing_columns.update(existing_data_df.columns)  # 更新现有列名
        except pd.errors.EmptyDataError:
            existing_data_df = pd.DataFrame()
    else:
        existing_data_df = pd.DataFrame()

    # 逐行读取数据文件并处理每一行数据
    with open(data_file, "r") as file:
        lines = file.readlines()[1:]  # 跳过第一行 除了测试集以外的
        total_lines = len(lines)
        for idx, line in enumerate(lines, start=1):
            try:
                print(f"这是第{idx}个了{line}")
                # if idx==4:
                #     break
                logging.info(f"Processing line {idx} of {total_lines}")
                protein_ids = []
                protein_id_1, protein_id_2, flag = line.strip().split(",")  # 使用“*”获取可能存在的多余数据
                protein_ids.append(protein_id_1.replace(":"," "))   # 由于windows文件名中不能包含特殊字符:，导致这里需要特殊处理Firmicutes bacterium CAG:83，如果PHI对文件中宿主种名称里面不包含特殊字符的可以启用上面那行代码
                # protein_ids.append(protein_id_2.replace(" ", "_")) # protein_ids内部：[噬菌体,宿主,噬菌体,宿主]         在case中不启用这一行
                protein_ids.append(protein_id_2.replace(":"," "))
                features_list_pair = []
                kmer_features_list = []

                num = 0
                for protein_id in protein_ids:
                    sequence_file_path = os.path.join(protein_dir, f"{protein_id}.txt")
                    sequences = read_fasta_like(sequence_file_path)
                    if not sequences:  # 如果 sequences 为空，则使用默认特征值
                        logging.warning(f"No sequences found in file {sequence_file_path}. Assigning default values.")
                        default_feature = calculate_protein_features("")  # 使用空字符串计算默认特征
                        features_list_pair.append([default_feature])
                    else:
                        # 每一个orf对应一个字典，protein_features里包含许多字典
                        protein_features = [calculate_protein_features(seq) for seq in sequences.values()]
                        features_list_pair.append(protein_features)
                    # num只可能是0或1，因为只有两个protein_id，当num为0是代表噬菌体，num为1时代表是宿主
                    if num == 0:# 根据目前处理的protein_id去{phage_dir}里找到对应的基因组序列文件，给下游的kmer序列特征提取提供路径
                        kmer_file_path = os.path.join(phage_dir, f"{protein_id}.fasta")
                    else:
                        kmer_file_path = os.path.join(host_dir, f"{protein_id}.fasta")
                    num += 1
                    kmer_features = calculate_kmer_features(kmer_file_path, kmer_size)
                    kmer_features_list.extend(list(kmer_features.values()))

                combined_protein_features = []
                # 这个长度表示有宿主多少个orf
                print(len(features_list_pair[1]))
                for features_1 in features_list_pair[0]:
                    for features_2 in features_list_pair[1]:
                        combined_features = {}
                        combined_features.update({f"1_{k}": v for k, v in features_1.items()})
                        combined_features.update({f"2_{k}": v for k, v in features_2.items()})
                        # combined_protein_features = [{1_k:v,2_k:v},{1_k:v,2_k:v},...]
                        combined_protein_features.append(combined_features)

                protein_features_combined_df = pd.DataFrame(combined_protein_features)
                protein_features_combined_df.fillna(0, inplace=True)

                mean_features = protein_features_combined_df.mean()
                variance_features = protein_features_combined_df.var()
                median_features = protein_features_combined_df.median()

                combined_statistical_features = {
                    **{f"Mean_{col}": mean_val for col, mean_val in mean_features.items()},
                    **{f"Variance_{col}": var_val for col, var_val in variance_features.items()},
                    **{f"Median_{col}": med_val for col, med_val in median_features.items()}
                }

                for idx, kmer_feature in enumerate(kmer_features_list):
                    combined_statistical_features.update({f"kmer_{idx}_{k}": v for k, v in kmer_feature.items()})

                # 添加噬菌体的accession、宿主的名称以及它们能否发生相互作用的label
                combined_statistical_features['Phage_Accession'] = protein_id_1
                combined_statistical_features['Host_Name'] = protein_id_2
                combined_statistical_features['Label'] = flag

                with open(csv_file, "a") as f:
                    row_features_df = pd.DataFrame([combined_statistical_features])
                    # 将 DataFrame 写入文件，不写入表头和索引
                    if isHeader:
                        row_features_df.to_csv(f, header=True, index=False)
                        isHeader = False
                    else:
                        row_features_df.to_csv(f, header=False, index=False)

                combined_statistical_features={}
                # 标记表头已写入
                # header_written = True

            except Exception as e:
                logging.error(f"Failed to process line {idx}: {line.strip()}. Error: {e}")

if __name__ == "__main__":
    main()