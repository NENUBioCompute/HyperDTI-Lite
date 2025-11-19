import torch
from torch.utils.data import Dataset
from operate_drug import get_vector_SMILES
from operate_target import get_vector_FASTA


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


"""对于每一个batch的数据，整理数据为input:[DrugIDs, ProteinIDs, SMILESs, FASTAs, vec_SMILESs, vec_FASTAs, label]"""
def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, target_ids,smiles_list, fasta_list= [], [], [], []
    MAX_len_SMILES = 100
    MAX_len_FASTA = 1000
    vec_SMILESs = torch.zeros((N, MAX_len_SMILES), dtype=torch.long)
    vec_FASTAs = torch.zeros((N, MAX_len_FASTA), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id, target_id, smiles, fasta, label = pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]

        drug_ids.append(drug_id)
        smiles_list.append(smiles)
        compound_int = torch.from_numpy(get_vector_SMILES(smiles, MAX_len_SMILES))
        vec_SMILESs[i] = compound_int

        target_ids.append(target_id)
        fasta_list.append(fasta)
        target_int = torch.from_numpy(get_vector_FASTA(fasta, MAX_len_FASTA))
        vec_FASTAs[i] = target_int

        label = float(label)
        labels_new[i] = int(label)

    return drug_ids, target_ids, smiles_list, fasta_list,  vec_SMILESs, vec_FASTAs, labels_new