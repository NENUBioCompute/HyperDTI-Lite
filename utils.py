import numpy as np
import random


def shuffle_dataset(dataset, seed=None):
    """
    打乱数据集的函数。
    参数：
        - dataset：原始数据集，可以是列表、numpy 数组或其他可迭代对象。
        - seed：随机种子，用于确保可重复性。默认为 None。
    返回：
        - shuffled_dataset：打乱后的数据集副本。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # 如果 dataset 是列表或其他可迭代对象
    if isinstance(dataset, (list, tuple)):
        shuffled_dataset = random.sample(dataset, k=len(dataset))
    # 如果 dataset 是 numpy 数组
    elif isinstance(dataset, np.ndarray):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        shuffled_dataset = dataset[indices]
    else:
        raise TypeError("Unsupported dataset type. Supported types are list, tuple, and numpy.ndarray.")

    return shuffled_dataset


def get_kfold_data(i, dataset, k=5):
    """
    获取第数据集dataset中的第 i 折数据

    参数
        - i : 要获取数据的折数，不大于k
        - dataset :  操作的数据集
        - k : 数据集划分总折数
    返回:
        -划分后的两个数据集
    """
    fold_size = len(dataset) // k
    test_start = i * fold_size
    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        test_set = dataset[test_start:test_end]
        train_set = dataset[0:test_start] + dataset[test_end:]
    elif i == 0:
        test_end = fold_size
        test_set = dataset[test_start:test_end]
        train_set = dataset[test_end:]
    else:
        test_set = dataset[test_start:]
        train_set = dataset[0:test_start]

    return train_set, test_set
