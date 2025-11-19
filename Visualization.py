# import torch
# import json
#
# from prefetch_generator import BackgroundGenerator
# from tqdm import tqdm
# from dataset import CustomDataSet, collate_fn
# from torch.utils.data import DataLoader
# from hyperparameter import hyperparameter
# from model import HyperTriplet  # 替换为你模型类所在文件名
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "18"
#
# def  have_a_try(dataset, model_loaded):
#     print(f"we will predict in {dataset} dataset  !")
#     dir_input = ('./data/{}.txt'.format(dataset))
#     with open(dir_input, "r") as f:
#         data_list = f.read().strip().split('\n')
#     print("load data finished")
#     test_dataset = CustomDataSet(data_list)
#     test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
#     tasktest_model = HyperTriplet(hp).cuda()
#     tasktest_model.load_state_dict(torch.load(model_loaded))
#     tasktest_model.eval()
#     data_bar = tqdm(enumerate(BackgroundGenerator(test_dataset_loader)), total=len(test_dataset_loader))
#     results1 = []
#     results2 = []
#     with torch.no_grad():
#         for i, data in data_bar:
#             drugIDs, proIDs, SMILESs, FASTAs, numSMILESs, numFASTAs, labels = data
#             numSMILESs = numSMILESs.cuda()
#             numFASTAs = numFASTAs.cuda()
#             output1,output2 = tasktest_model(drugIDs, proIDs,  SMILESs, FASTAs, numSMILESs, numFASTAs)
#
#             for y1, label1 in zip(output1, labels):
#                 results1.append({
#                     "vector": y1.cpu().numpy().tolist(),
#                     "label": int(label1.item())
#                 })
#
#             for y2, label2 in zip(output2, labels):
#                 results2.append({
#                     "vector": y2.cpu().numpy().tolist(),
#                     "label": int(label2.item())
#                 })
#         # 保存到 json 文件
#         with open("model_output1.json", "w") as f:
#             json.dump(results1, f, indent=2)
#         with open("model_output2.json", "w") as f:
#             json.dump(results2, f, indent=2)
#
#
#
# hp = hyperparameter()
# model = HyperTriplet(hp).cuda()
# task_dataset = "DrugBank35022"
# model_pth_file = './DrugBank35022/fold_0_valid_best_checkpoint.pth'
# have_a_try(task_dataset,model_pth_file)
#
#=====================================================================================================
#
# import torch
# import json
# from prefetch_generator import BackgroundGenerator
# from tqdm import tqdm
# from dataset import CustomDataSet, collate_fn
# from torch.utils.data import DataLoader
# from hyperparameter import hyperparameter
# from model import HyperTriplet
# import os
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def have_a_try(dataset, model_loaded):
#     print(f"we will predict in {dataset} dataset  !")
#     dir_input = ('./data/{}.txt'.format(dataset))
#     with open(dir_input, "r") as f:
#         data_list = f.read().strip().split('\n')
#     print("load data finished")
#
#     test_dataset = CustomDataSet(data_list)
#     test_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
#
#     model = HyperTriplet(hp).to(device)
#     model.load_state_dict(torch.load(model_loaded, map_location=device))
#     model.eval()
#
#     results1 = []
#     results2 = []
#     data_bar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
#
#     with torch.no_grad():
#         for i, data in data_bar:
#             # 若这里仍报 unpack 错，请 print(len(data)) 或 print(data) 检查
#             drugIDs, proIDs, SMILESs, FASTAs, numSMILESs, numFASTAs, labels = data
#             numSMILESs = numSMILESs.to(device)
#             numFASTAs = numFASTAs.to(device)
#
#             output1, output2 = model(drugIDs, proIDs, SMILESs, FASTAs, numSMILESs, numFASTAs)
#
#             for y1, label1 in zip(output1, labels):
#                 results1.append({
#                     "vector": y1.cpu().numpy().tolist(),
#                     "label": int(label1.item())
#                 })
#             for y2, label2 in zip(output2, labels):
#                 results2.append({
#                     "vector": y2.cpu().numpy().tolist(),
#                     "label": int(label2.item())
#                 })
#
#     # 保存结果
#     with open("model_output1.json", "w") as f:
#         json.dump(results1, f, indent=2)
#     with open("model_output2.json", "w") as f:
#         json.dump(results2, f, indent=2)
#     print("JSON saved!")
#
# # 运行
# hp = hyperparameter()
# task_dataset = "DrugBank2570"
# model_pth_file = './DrugBank35022/fold_3_valid_best_checkpoint.pth'
# have_a_try(task_dataset, model_pth_file)
# =====================================================================================================
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 1: 读取 JSON 数据
with open("model_output2.json", "r") as f:
    data = json.load(f)

# Step 2: 提取向量和标签
vectors = np.array([item["vector"] for item in data])
labels = np.array([item["label"] for item in data])

# Step 3: t-SNE 降维到二维
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate='auto')
vectors_2d = tsne.fit_transform(vectors)

# Step 4: 可视化
plt.figure(figsize=(8, 6))
colors = ['blue', 'red']
label_names = ['Label 0', 'Label 1']

for label in np.unique(labels):
    indices = labels == label
    plt.scatter(
        vectors_2d[indices, 0],
        vectors_2d[indices, 1],
        c=colors[label],
        label=label_names[label],
        alpha=0.6,
        edgecolors='k',
        s=30
    )

plt.title("t-SNE Visualization of model_output2.json")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_model_output2.png", dpi=300)
plt.show()
