"""生成蛋白质序列esm编码，并保存进相应名称的dict.json文件，后续模型直接读取即可，否则esm编码时间过长"""
"""只需执行一次，生成json文件即可"""
import json
from datetime import datetime
from itertools import product
from collections import Counter
from math import log2
import torch
import esm
# 定义氨基酸的分子量
aa_molecular_weights = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.18,
    'L': 131.18, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

# 定义氨基酸的pKa值
aa_pKa = {
    'D': {'pKa': 3.86, 'charge': -1}, 'E': {'pKa': 4.25, 'charge': -1},
    'C': {'pKa': 8.33, 'charge': -1}, 'Y': {'pKa': 10.07, 'charge': -1},
    'H': {'pKa': 6.04, 'charge': 1}, 'K': {'pKa': 10.54, 'charge': 1},
    'R': {'pKa': 12.48, 'charge': 1}
}

# 定义Kyte-Doolittle疏水性量表
kd_hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# 定义Chou-Fasman二级结构倾向性
chou_fasman = {
    'A': {'alpha': 1.45, 'beta': 0.97}, 'R': {'alpha': 0.79, 'beta': 0.90},
    'N': {'alpha': 0.73, 'beta': 0.65}, 'D': {'alpha': 0.98, 'beta': 0.80},
    'C': {'alpha': 0.77, 'beta': 1.30}, 'E': {'alpha': 1.53, 'beta': 0.26},
    'Q': {'alpha': 1.17, 'beta': 1.23}, 'G': {'alpha': 0.53, 'beta': 0.81},
    'H': {'alpha': 1.24, 'beta': 0.71}, 'I': {'alpha': 1.00, 'beta': 1.60},
    'L': {'alpha': 1.34, 'beta': 1.22}, 'K': {'alpha': 1.07, 'beta': 0.74},
    'M': {'alpha': 1.20, 'beta': 1.67}, 'F': {'alpha': 1.12, 'beta': 1.28},
    'P': {'alpha': 0.59, 'beta': 0.62}, 'S': {'alpha': 0.79, 'beta': 0.72},
    'T': {'alpha': 0.82, 'beta': 1.20}, 'W': {'alpha': 1.14, 'beta': 1.19},
    'Y': {'alpha': 0.61, 'beta': 1.29}, 'V': {'alpha': 1.14, 'beta': 1.65}
}

# 定义不稳定指数表（部分示例）
instability_index_table = {
    'A': {'A': 0, 'C': 44.94, 'D': -7.49, 'E': -6.54, 'F': 13.34},
    'C': {'A': 44.94, 'C': 0, 'D': 13.34, 'E': 13.34, 'F': 13.34},
    # 其他氨基酸组合的稳定性贡献值
}


# Load ESM-2 model
start_load_time = datetime.now()
print(f"the time of start loading model is {start_load_time}")

# 指定本地模型文件路径
model_path = "./local_model/ESM2/esm2_t33_650M_UR50D.pt"
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
end_load_time = datetime.now()
print(f"Loading model completed ! it takes {end_load_time - start_load_time}s !")

# 这个函数返回两个值：model 是加载的模型，alphabet 是用于编码序列的字母表。
batch_converter = alphabet.get_batch_converter()
esm_model.eval()  # disables dropout for deterministic results


def embed_target_with_ESM(protein_sequence: str):
    """对单序列进行编码"""
    data_to_handle = [("protein", protein_sequence)]
    # 使用batch_converter处理数据
    batch_labels, batch_strs, batch_tokens = batch_converter(data_to_handle)
    # 在不需要梯度的上下文中提取每个残基的表示
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
    # 通过平均生成每个序列的表示
    sequence_representation = token_representations.mean(1).squeeze(0)
    return sequence_representation

def calculate_mer_frequencies(sequence):
    """
    计算蛋白质序列的 1-mer 和 2-mer 频率，并按顺序返回数值列表。

    参数:
    - sequence (str): 蛋白质的氨基酸序列。

    返回:
    - freq_1mer_list (list): 1-mer 频率列表，按字母顺序排列。
    - freq_2mer_list (list): 2-mer 频率列表，按字母顺序排列。
    """
    # 定义所有可能的氨基酸
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # 初始化 1-mer 计数字典
    count_1mer = {aa: 0 for aa in amino_acids}

    # 统计 1-mer
    for aa in sequence:
        if aa in count_1mer:
            count_1mer[aa] += 1

    # 计算 1-mer 频率并按字母顺序排列
    total_1mer = len(sequence)
    freq_1mer_list = [count_1mer[aa] / total_1mer for aa in sorted(amino_acids)]

    # 生成所有可能的 2-mer 组合并按字母顺序排列
    all_2mers = sorted([''.join(pair) for pair in product(amino_acids, repeat=2)])

    # 初始化 2-mer 计数字典
    count_2mer = {pair: 0 for pair in all_2mers}

    # 统计 2-mer
    for i in range(len(sequence) - 1):
        pair = sequence[i:i + 2]
        if pair in count_2mer:
            count_2mer[pair] += 1

    # 计算 2-mer 频率并按字母顺序排列
    total_2mer = len(sequence) - 1
    freq_2mer_list = [count_2mer[pair] / total_2mer for pair in all_2mers]

    return freq_1mer_list+ freq_2mer_list

def calculate_protein_properties(sequence):
    # 分子量
    molecular_weight = sum(aa_molecular_weights.get(aa, 0) for aa in sequence)

    # 静电荷（pH=7）
    def calculate_net_charge(sequence, pH):
        net_charge = 0
        for aa in sequence:
            if aa in aa_pKa:
                pKa = aa_pKa[aa]['pKa']
                charge = aa_pKa[aa]['charge']
                if pH < pKa:
                    net_charge += charge
                else:
                    net_charge -= charge
        return net_charge
    net_charge = calculate_net_charge(sequence, 7.0)

    # 亲疏水性（平均疏水性）
    average_hydrophobicity = sum(kd_hydrophobicity.get(aa, 0) for aa in sequence) / len(sequence)

    # 消光系数
    extinction_coefficient = sequence.count('W') * 5500 + sequence.count('Y') * 1490 + sequence.count('C') * 125

    # 不稳定指数
    def calculate_instability_index(sequence):
        total = 0
        for i in range(len(sequence) - 1):
            aa1 = sequence[i]
            aa2 = sequence[i + 1]
            total += instability_index_table.get(aa1, {}).get(aa2, 0)
        return (10 / len(sequence)) * total
    instability_index = calculate_instability_index(sequence)

    # 带电荷氨基酸数量
    charged_aas = {'R', 'K', 'H', 'D', 'E'}
    charged_aa_count = sum(1 for aa in sequence if aa in charged_aas)

    # 芳香族氨基酸数量
    aromatic_aas = {'F', 'Y', 'W'}
    aromatic_aa_count = sum(1 for aa in sequence if aa in aromatic_aas)

    # 半胱氨酸数量
    cysteine_count = sequence.count('C')

    # 二级结构倾向性
    alpha_tendency = sum(chou_fasman.get(aa, {'alpha': 0})['alpha'] for aa in sequence) / len(sequence)
    beta_tendency = sum(chou_fasman.get(aa, {'beta': 0})['beta'] for aa in sequence) / len(sequence)

    # 序列复杂性（Shannon熵）
    def calculate_sequence_complexity(sequence):
        aa_counts = Counter(sequence)
        total = len(sequence)
        entropy = -sum((count / total) * log2(count / total) for count in aa_counts.values())
        return entropy
    sequence_complexity = calculate_sequence_complexity(sequence)

    # 返回所有属性
    return [
        molecular_weight, net_charge, average_hydrophobicity,  extinction_coefficient,
        instability_index, charged_aa_count, aromatic_aa_count, cysteine_count,
        alpha_tendency, beta_tendency, sequence_complexity
    ]


# dataset_list = [ "DrugBank7710","DrugBank2570"]
# dataset_list = [ "DrugBank35022"]
# dataset_list = [ "DrugBank200"]
dataset_list = [ "KIBA"]
#====================================================================================================================
# # 生成ESM特征词典
# for dataset_name in dataset_list:
#     dir_input = ('./data/{}.txt'.format(dataset_name))
#     start_handle_time = datetime.now()
#     print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
#     with open(dir_input, "r") as f:
#         data = f.read().strip().split('\n')
#     esm_encoder_dict = {}
#     for index,item in enumerate(data):
#         if data[index].split(' ')[1] not in esm_encoder_dict.keys():
#             print(f"{dataset_name}: generate {index + 1}/{len(data)} dict_values ...")
#             if len(data[index].split(' ')[3]) > 2000:
#                 feature = embed_target_with_ESM(data[index].split(' ')[3][:2000])
#             else:
#                 feature = embed_target_with_ESM(data[index].split(' ')[3])
#             # tensor转list数位会增多，这里将数位控制在4位
#             feature = [round(num, 4) for num in feature.tolist()]
#             esm_encoder_dict.update({data[index].split(' ')[1]:feature})
#     print(f"the length of {dataset_name}_dict is {len(esm_encoder_dict)} .")
#     with open(f'./pre_files/{dataset_name}_target_ESM_feature_dict.json', 'w') as f:
#         json.dump(esm_encoder_dict, f)
#     print(f'save {dataset_name}_target_ESM_feature_dict.json file success !!!')
#     end_handle_time = datetime.now()
#     print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")
#====================================================================================================================
# # 统计氨基酸频率json
# for dataset_name in dataset_list:
#     dir_input = ('./data/{}.txt'.format(dataset_name))
#     start_handle_time = datetime.now()
#     print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
#     with open(dir_input, "r") as f:
#         data = f.read().strip().split('\n')
#     mer_encoder_dict = {}
#     for index,item in enumerate(data):
#         if data[index].split(' ')[1] not in mer_encoder_dict.keys():
#             print(f"{dataset_name}: generate {index + 1}/{len(data)} dict_values ...")
#             if len(data[index].split(' ')[3]) > 2000:
#                 feature_mer = calculate_mer_frequencies(data[index].split(' ')[3][:2000])
#             else:
#                 feature_mer = calculate_mer_frequencies(data[index].split(' ')[3])
#             # tensor转list数位会增多，这里将数位控制在4位
#             feature_mer = [round(num, 4) for num in feature_mer]
#             mer_encoder_dict.update({data[index].split(' ')[1]:feature_mer})
#
#     print(f"the length of {dataset_name}_dict is {len(mer_encoder_dict)} .")
#     with open(f'./pre_files/{dataset_name}_target_mer_feature_dict.json', 'w') as f:
#         json.dump(mer_encoder_dict, f)
#     print(f'save {dataset_name}_target_mer_feature_dict.json file success !!!')
#     end_handle_time = datetime.now()
#     print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")

#====================================================================================================================
# 统计氨基酸理化属性json
for dataset_name in dataset_list:
    dir_input = ('./data/{}.txt'.format(dataset_name))
    start_handle_time = datetime.now()
    print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
    with open(dir_input, "r") as f:
        data = f.read().strip().split('\n')
    PCproperties_encoder_dict = {}
    for index,item in enumerate(data):
        if data[index].split(' ')[1] not in PCproperties_encoder_dict.keys():
            print(f"{dataset_name}: generate {index + 1}/{len(data)} dict_values ...")
            if len(data[index].split(' ')[3]) > 2000:
                feature_PCproperties = calculate_protein_properties(data[index].split(' ')[3][:2000])
            else:
                feature_PCproperties = calculate_protein_properties(data[index].split(' ')[3])
            # tensor转list数位会增多，这里将数位控制在4位
            feature_PCproperties = [round(num, 4) for num in  feature_PCproperties]
            PCproperties_encoder_dict.update({data[index].split(' ')[1]: feature_PCproperties})

    print(f"the length of {dataset_name}_dict is {len(PCproperties_encoder_dict)} .")
    with open(f'./pre_files/{dataset_name}_target_PCproperties_feature_dict.json', 'w') as f:
        json.dump(PCproperties_encoder_dict, f)
    print(f'save {dataset_name}_target_PCproperties_feature_dict.json file success !!!')
    end_handle_time = datetime.now()
    print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")
