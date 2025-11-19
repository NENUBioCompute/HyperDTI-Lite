import numpy as np



char_set_of_FASTA = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
length_of_FASTA_charset = len(char_set_of_FASTA)

# 将FASTA转化为向量
def get_vector_FASTA(pro_seq, dim = 1000):
    X = np.zeros(dim, np.int64())
    for i, ch in enumerate(pro_seq[:dim]):
        X[i] = char_set_of_FASTA[ch]
    return X


