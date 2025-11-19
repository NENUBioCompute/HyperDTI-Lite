"""生成药物SMILES序列ChemBert编码，并保存进相应名称的dict.json文件，后续模型直接读取即可，否则ChemBert编码时间过长"""
import os

"""只需执行一次，生成json文件即可"""
import json
from datetime import datetime
from transformers import RobertaTokenizer, RobertaModel
import torch
import warnings
from rdkit.Chem import Descriptors, rdMolDescriptors
import torchvision
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem


# 禁用 Beta 版本的警告
torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=FutureWarning)

# Load ChemBert model
start_load_time = datetime.now()
print(f"the time of start loading model is {start_load_time}")

# 本地模型文件路径
model_dir = "./local_model/ChemBERTa/"

# 从本地路径加载模型和Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
chem_model = RobertaModel.from_pretrained(model_dir)

end_load_time = datetime.now()
print(f"Loading ChemBert model completed ! it takes {end_load_time - start_load_time}s !")


# 对SMILES字符串进行编码
def embed_drug_with_ChemBERT(smiles):
    inputs = tokenizer(smiles, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = chem_model(**inputs)
    smiles_embedding = outputs.last_hidden_state.mean(dim=1)
    return smiles_embedding.squeeze()

def calculate_properties_sorted(smiles):
    """
    从 SMILES 字符串计算分子性质，并返回按顺序排列的物理化学属性值。
    参数:
        smiles (str): 分子的 SMILES 字符串。
    返回:
        list: 包含 11 种物理化学属性值的列表，按以下顺序排列：
              [分子量, LogP, 氢键供体数量, 氢键受体数量, 可旋转键数量, TPSA, 分子折射率, 芳香环数量, 重原子数量, 形式电荷, LogS]
              如果 SMILES 无效，返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"警告: 无效的 SMILES 字符串: {smiles}")
        return None
    try:
        # 计算分子性质
        properties = [
            round(Descriptors.MolWt(mol), 2),  # 分子量
            round(Descriptors.MolLogP(mol), 2),  # LogP
            Descriptors.NumHDonors(mol),  # 氢键供体数量
            Descriptors.NumHAcceptors(mol),  # 氢键受体数量
            Descriptors.NumRotatableBonds(mol),  # 可旋转键数量
            round(Descriptors.TPSA(mol), 2),  # TPSA
            round(Descriptors.MolMR(mol), 2),  # 分子折射率
            Descriptors.NumAromaticRings(mol),  # 芳香环数量
            Descriptors.HeavyAtomCount(mol),  # 重原子数量
            Chem.GetFormalCharge(mol),  # 形式电荷
            round(rdMolDescriptors.CalcCrippenDescriptors(mol)[1], 2),  # LogS
        ]
        return properties
    except Exception as e:
        print(f"计算分子性质时出错: {e}")
        return None


def get_maccs_fingerprint(smiles):
    """
    根据输入的 SMILES 字符串生成 MACCS 指纹。

    参数:
    - smiles: 药物的 SMILES 字符串。

    返回:
    - MACCS 指纹（二进制字符串）。
    """
    # 将 SMILES 转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES 字符串")

    # 生成 MACCS 指纹
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)

    # 将指纹转换为二进制字符串
    maccs_bitstring = maccs_fp.ToBitString()

    return maccs_bitstring


def smiles_to_ecfp(smiles, radius=2, n_bits=765):
    """
    将SMILES字符串转换为ECFP指纹。

    参数:
    smiles (str): 药物分子的SMILES字符串。
    radius (int): ECFP的半径，默认为2。
    n_bits (int): 指纹的位数，默认为1024。

    返回:
    list: ECFP指纹的位向量。
    """
    # 将SMILES字符串转换为RDKit的分子对象
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("无效的SMILES字符串")

    # 使用 MorganGenerator 生成 ECFP 指纹
    ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, useChirality=False
    )

    # 将指纹转换为位向量
    ecfp_bitvector = list(ecfp)

    return ecfp_bitvector

def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    将SMILES字符串转换为Morgan指纹。

    参数:
    - smiles (str): 药物的SMILES字符串。
    - radius (int): Morgan指纹的半径，默认为2。
    - n_bits (int): 指纹的位数，默认为2048。

    返回:
    - morgan_fp (list): Morgan指纹的位向量。
    """
    # 将SMILES字符串转换为RDKit的分子对象
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("无效的SMILES字符串")

    # 生成Morgan指纹
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    # 将指纹转换为位向量列表
    morgan_fp_list = list(morgan_fp)

    return morgan_fp_list
#====================================================================================================================
# dataset_list = [ "DrugBank7710","DrugBank2570"]
# dataset_list = [ "DrugBank35022"]
dataset_list = [ "KIBA"]
# dataset_list = [ "DrugBank200"]
#====================================================================================================================
# # 计算ChemBERTa.json
# for dataset_name in dataset_list:
#     dir_input = ('./data/{}.txt'.format(dataset_name))
#     start_handle_time = datetime.now()
#     print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
#     with open(dir_input, "r") as f:
#         data = f.read().strip().split('\n')
#     chembert_encoder_dict = {}
#     for index,item in enumerate(data):
#         if data[index].split(' ')[0] not in chembert_encoder_dict.keys():
#             print(f"{dataset_name}: generate {index + 1}/{len(data)} dict_values ...")
#             if len(data[index].split(' ')[0]) > 2000:
#                 feature_chemberta = embed_drug_with_ChemBERT(data[index].split(' ')[2][:2000])
#             else:
#                 feature_chemberta = embed_drug_with_ChemBERT(data[index].split(' ')[2])
#             # tensor转list数位会增多，这里将数位控制在4位
#             feature_chemberta = [round(num, 4) for num in feature_chemberta.tolist()]
#             chembert_encoder_dict.update({data[index].split(' ')[0]:feature_chemberta})
#     print(f"the length of {dataset_name}_dict is {len(chembert_encoder_dict)} .")
#     with open(f'./pre_files/{dataset_name}_drug_ChemBert_feature_dict.json', 'w') as f:
#         json.dump(chembert_encoder_dict, f)
#     print(f'save {dataset_name}_drug_ChemBert_feature_dict.json file success !!!')
#     end_handle_time = datetime.now()
#     print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")
#====================================================================================================================
# # 计算理化属性.json
# for dataset_name in dataset_list:
#     dir_input = ('./data/{}.txt'.format(dataset_name))
#     start_handle_time = datetime.now()
#     print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
#     with open(dir_input, "r") as f:
#         data = f.read().strip().split('\n')
#     drug_PCproperties_encoder_dict = {}
#     for index,item in enumerate(data):
#         if data[index].split(' ')[0] not in drug_PCproperties_encoder_dict.keys():
#             print(f"{dataset_name}: generate {index + 1}/{len(data)} PCproperties dict_values ...")
#             if len(data[index].split(' ')[0]) > 2000:
#                 feature_PC = calculate_properties_sorted(data[index].split(' ')[2][:2000])
#             else:
#                 feature_PC = calculate_properties_sorted(data[index].split(' ')[2])
#             # tensor转list数位会增多，这里将数位控制在4位
#             feature_PC = [round(num, 4) for num in feature_PC]
#             drug_PCproperties_encoder_dict.update({data[index].split(' ')[0]:feature_PC})
#
#     print(f"the length of {dataset_name}_dict is {len(drug_PCproperties_encoder_dict)} .")
#     with open(f'./pre_files/{dataset_name}_drug_PCproperties_feature_dict.json', 'w') as f:
#         json.dump(drug_PCproperties_encoder_dict, f)
#     print(f'save {dataset_name}_drug_PCproperties_feature_dict.json file success !!!')
#     end_handle_time = datetime.now()
#     print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")
#====================================================================================================================
# # 生成MACCS、ECFP、Morgan指纹的json
for dataset_name in dataset_list:
    dir_input = ('./data/{}.txt'.format(dataset_name))
    start_handle_time = datetime.now()
    print(f"start_time of handling dataset {dataset_name} is {start_handle_time}")
    with open(dir_input, "r") as f:
        data = f.read().strip().split('\n')
    drug_MACCS_encoder_dict = {}
    drug_ECFP_encoder_dict = {}
    drug_Morgan_encoder_dict = {}

    for index,item in enumerate(data):
        if data[index].split(' ')[0] not in drug_MACCS_encoder_dict.keys():
            #MACCS
            print(f"{dataset_name}: generate {index + 1}/{len(data)} MACCS dict_values ...")
            if len(data[index].split(' ')[0]) > 2000:
                feature_MACCS = get_maccs_fingerprint(data[index].split(' ')[2][:2000])
            else:
                feature_MACCS = get_maccs_fingerprint(data[index].split(' ')[2])
            # tensor转list数位会增多，这里将数位控制在4位
            feature_MACCS = [round(float(num), 4) for num in list(feature_MACCS)]
            drug_MACCS_encoder_dict.update({data[index].split(' ')[0]:feature_MACCS})

            #ECFP
            print(f"{dataset_name}: generate {index + 1}/{len(data)} ECFP dict_values ...")
            if len(data[index].split(' ')[0]) > 2000:
                feature_ECFP = smiles_to_ecfp(data[index].split(' ')[2][:2000])
            else:
                feature_ECFP = smiles_to_ecfp(data[index].split(' ')[2])
            # tensor转list数位会增多，这里将数位控制在4位
            feature_ECFP = [round(float(num), 4) for num in list(feature_ECFP)]
            drug_ECFP_encoder_dict.update({data[index].split(' ')[0]: feature_ECFP})

            # Morgan
            print(f"{dataset_name}: generate {index + 1}/{len(data)} Morgan dict_values ...")
            if len(data[index].split(' ')[0]) > 2000:
                feature_Morgan = smiles_to_morgan_fingerprint(data[index].split(' ')[2][:2000])
            else:
                feature_Morgan = smiles_to_morgan_fingerprint(data[index].split(' ')[2])
            # tensor转list数位会增多，这里将数位控制在4位
            feature_Morgan = [round(float(num), 4) for num in list(feature_Morgan)]
            drug_Morgan_encoder_dict.update({data[index].split(' ')[0]: feature_Morgan})

    print(f"the length of {dataset_name}_dict is {len(drug_MACCS_encoder_dict)} .")
    with open(f'./pre_files/{dataset_name}_drug_MACCS_feature_dict.json', 'w') as f:
        json.dump(drug_MACCS_encoder_dict, f)
    print(f'save {dataset_name}_drug_MACCS_feature_dict.json file success !!!')

    print(f"the length of {dataset_name}_dict is {len(drug_ECFP_encoder_dict)} .")
    with open(f'./pre_files/{dataset_name}_drug_ECFP_feature_dict.json', 'w') as f:
        json.dump(drug_ECFP_encoder_dict, f)
    print(f'save {dataset_name}_drug_ECFP_feature_dict.json file success !!!')

    print(f"the length of {dataset_name}_dict is {len(drug_Morgan_encoder_dict)} .")
    with open(f'./pre_files/{dataset_name}_drug_Morgan_feature_dict.json', 'w') as f:
        json.dump(drug_Morgan_encoder_dict, f)
    print(f'save {dataset_name}_drug_Morgan_feature_dict.json file success !!!')


    end_handle_time = datetime.now()
    print(f"the total time of handling {dataset_name} is {end_handle_time - start_handle_time}")