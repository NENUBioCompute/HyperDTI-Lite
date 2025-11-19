# import os
# import numpy as np
# import torch
# from torch import nn
# import pytz
# import time
# from prefetch_generator import BackgroundGenerator
# from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
# import geoopt
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from hyperparameter import hyperparameter
# from model_euc import HyperTriplet
# from pytorchtools import EarlyStopping
# from utils import shuffle_dataset, get_kfold_data
# from dataset import CustomDataSet, collate_fn
# from datetime import datetime
# import torch.nn.functional as F
# import torch.optim as optim
# # from torch.optim.lr_scheduler import StepLR
# # from torch.nn import DataParallel
#
#
# tz = pytz.timezone('Asia/Shanghai')  # 定义东八区的时区
# start_timestamp = time.time()  # 获取程序开始时间（以时间戳格式）
# os.environ["CUDA_VISIBLE_DEVICES"] = "16"# 单卡
# # os.environ["CUDA_VISIBLE_DEVICES"] = "16, 17, 18 ,19" #多卡
#
# # def check_gradients(input_model):
# #     print("\n Gradient Check:")
# #     for name, param in input_model.named_parameters():
# #         if not param.requires_grad:
# #             print(f"{name:40}  requires_grad = False")
# #         elif param.grad is None:
# #             print(f"{name:40}  grad is None")
# #         elif torch.all(param.grad == 0):
# #             print(f"{name:40} ️ grad = 0 (no update)")
# #         else:
# #             grad_norm = param.grad.norm().item()
# #             print(f"{name:40}  grad norm = {grad_norm:.4e}")
#
#
# def test_model(model_to_be_tested, dataset_load, full_name_of_dataset, test_result_save_path, loss_test_process,
#                dataset_description="Train", record_label=True):
#     """
#     在指定数据集上测试模型，并返回指标、打印结果
#     :param model_to_be_tested: 待测模型
#     :param dataset_load: 测试的数据集
#     :param full_name_of_dataset: 数据集全名
#     :param test_result_save_path: 结果保存路径 or 文件夹
#     :param loss_test_process: 训练过程loss函数
#     :param dataset_description: ”Train“ ”Valid” ”Test“
#     :param record_label: 是否记录
#     :return:  五项指标 + loss
#     """
#     model_to_be_tested.eval()
#     test_pbar = tqdm(enumerate(BackgroundGenerator(dataset_load)), total=len(dataset_load))
#
#     loss_list_test_process = []
#     Y_test_process, P_test_process, S_test_process = [], [], []
#
#     result_file = test_result_save_path + "/{}_{}_prediction.txt".format(full_name_of_dataset, dataset_description)
#
#     # 如果文件存在，则先清除：因为下面需要 追加写；否则会迭代追加写；
#     if record_label:
#         if os.path.exists(result_file):
#             os.remove(result_file)
#     with torch.no_grad():
#         for i, data_process in test_pbar:
#             drugIDs, targetIDs, SMILESs, FASTAs, vecSMILESs, vecFASTAs, labels = data_process
#             vecSMILESs = vecSMILESs.cuda()
#             vecFASTAs = vecFASTAs.cuda()
#             labels = labels.cuda()
#             predicted_scores = model(drugIDs, targetIDs, SMILESs, FASTAs, vecSMILESs, vecFASTAs)
#
#             loss = loss_test_process(predicted_scores, labels)
#             #**************************双曲loss***************************************
#             # loss_list_test_process.append(loss.item())
#             # # 单卡
#             # predicted_anchor = hp.poincare_ball.expmap0(hp.anchor_factor * torch.ones(hp.hyperbolic_dim).cuda())
#             # predicted_dists = hp.poincare_ball.dist(predicted_scores, predicted_anchor)
#             # # #多卡
#             # # batch_size = predicted_scores.size(0)
#             # # anchor_vector = torch.ones(hp.hyperbolic_dim, device=predicted_scores.device) * hp.anchor_factor
#             # # predicted_anchor = hp.poincare_ball.expmap0(anchor_vector).expand(batch_size, -1)
#             # # predicted_dists = hp.poincare_ball.dist(predicted_scores, predicted_anchor)
#             # predicted_labels = np.where(predicted_dists.to('cpu') <= hp.threshold, 1, 0)
#             # truth_labels = labels.cpu().data.numpy()
#             # Y_test_process.extend(truth_labels)
#             # P_test_process.extend(predicted_labels)
#             # S_test_process.extend(-predicted_dists.cpu().data.numpy())
#             #******************************交叉熵loss*********************************************
#             truth_labels = labels.to('cpu').data.numpy()
#             predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()  # 口口
#             # predicted_labels = np.argmax(predicted_scores, axis=1)  # 口：0 or 1。np.argmax即为默认threshold为0.5
#             predicted_labels = (predicted_scores[:, 1] > hp.threshold).astype(int)
#             predicted_scores = predicted_scores[:, 1]
#             Y_test_process.extend(truth_labels)
#             P_test_process.extend(predicted_labels)
#             S_test_process.extend(predicted_scores)
#             loss_list_test_process.append(loss.item())
#             # ***************************************************************************
#
#             if record_label:
#                 with open(result_file, 'a') as test_process_file:
#                     for index in range(len(drugIDs)):
#                         message = f"{drugIDs[index]} {targetIDs[index]} {str(truth_labels[index])} {str(predicted_labels[index])} {truth_labels[index] == predicted_labels[index]}"
#                         test_process_file.write(message + "\n")
#
#     Accuracy = accuracy_score(Y_test_process, P_test_process)
#     Precision = precision_score(Y_test_process, P_test_process)
#     Recall = recall_score(Y_test_process, P_test_process)
#     AUC = roc_auc_score(Y_test_process, S_test_process)
#     P_of_PRC, R_of_PRC, _ = precision_recall_curve(Y_test_process, S_test_process)
#     PRC = auc(R_of_PRC, P_of_PRC)
#     average_loss_test_process = np.average(loss_list_test_process)
#
#     return Accuracy, Precision, Recall, AUC, PRC, average_loss_test_process
#
#
# def show_result(dataset, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, time1_stamp, time2_stamp):
#     """
#     :param dataset:实验数据集文件名
#     :param Accuracy_List:K折实验Acc实验结果list
#     :param Precision_List: K折实验Pre实验结果list
#     :param Recall_List: K折实验Recall实验结果list
#     :param AUC_List: K折实验AUC实验结果list
#     :param AUPR_List: K折实验PRC实验结果list
#     :param time1_stamp: 程序开始时间的时间戳格式
#     :param time2_stamp: 程序结束时间的时间戳格式
#     :return:
#     """
#     Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
#     Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
#     Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
#     AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
#     PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
#     F1_score_list = []
#     for num in range(len(Accuracy_List)):
#         F1_score_list.append(2.0 * Precision_List[num] * Recall_List[num] / (Precision_List[num] + Recall_List[num]))
#     F1_score_mean, F1_score_var = np.mean(F1_score_list), np.var(F1_score_list)
#
#     # 将时间戳转换为东八区的时间
#     start_time = datetime.fromtimestamp(time1_stamp, tz)
#     end_time = datetime.fromtimestamp(time2_stamp, tz)
#     # 计算时间差并转换为小时、分钟和秒
#     elapsed_time_seconds = time2_stamp - time1_stamp
#     hours, remainder = divmod(elapsed_time_seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#
#     with open("./{}/results.txt".format(dataset), 'a') as file_show:
#         file_show.write(f"K_fold: {hp.K_fold}\t")
#         file_show.write(f"Epoch: {hp.Epoch}\t")
#         file_show.write(f"Threshold: {hp.threshold}\t")
#         file_show.write(f"Patience: {hp.Patience}\t")
#         file_show.write(f"Batch_size: {hp.Batch_size}\n")
#
#         file_show.write(f"Hyper curvature: {hp.curvature}\t")
#         file_show.write(f"Hyper dim:{hp.hyperbolic_dim}\t")
#         file_show.write(f"Hyperbolic lr: {hp.hyperbolic_lr}\n")
#
#         file_show.write(f"Loss rate : {hp.loss_rate}\t")
#         file_show.write(f"Threshold: {hp.hyper_threshold}\t")
#         file_show.write(f"Anchor factor: {hp.anchor_factor}\n")
#
#         file_show.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
#         file_show.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
#         file_show.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
#         file_show.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
#         file_show.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
#         file_show.write('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var) + '\n')
#         file_show.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}" + '\t')
#         file_show.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}" + '\n')
#         file_show.write(f"Elapsed Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds" + '\n')
#         file_show.write('  \n')
#     print(f"The model's results on {dataset}:")
#     print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
#     print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
#     print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
#     print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
#     print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
#     print('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var))
#     print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Elapsed Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
#
#
# if __name__ == "__main__":
#
#     " selected dataset "
#     DATASET = "DrugBank35022"
#     # DATASET = "DrugBank2570"
#     # DATASET = "DrugBank200"
#     # DATASET = "DrugBank7710"
#
#     '''select seed'''
#     SEED = 42
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#
#     '''init hyper parameters'''
#     hp = hyperparameter()
#
#     '''load data'''
#     print("Train in " + DATASET)
#     dir_input = f"./data/{DATASET}.txt"
#     print("loading data ...")
#     with open(dir_input, "r") as f:
#         data = f.read().strip().split('\n')
#     data = shuffle_dataset(data, SEED)
#     print("loading finished !\ndata shuffle finished ! ")
#
#     # 定义全局指标：Acc、Pre 、Recall 、AUC 、PRC, 最后计算K折的平均指标
#     Accuracy_List_global, Precision_List_global, Recall_List_global, AUC_List_global, AUPR_List_global = [], [], [], [], []
#     K_fold = hp.K_fold
#
#     for i_fold in range(K_fold):
#         print('*' * 25, ' No.', i_fold + 1, '-fold ', '*' * 25)
#
#         """ prepare data """
#         train_dataset, test_dataset = get_kfold_data(i_fold, data)
#         TVdataset = CustomDataSet(train_dataset)  # train训练集 and valid验证集
#         test_dataset = CustomDataSet(test_dataset)
#         TVdataset_len = len(TVdataset)
#         valid_size = int(0.2 * TVdataset_len)  # 验证集的比例是 TV数据集的20%
#         train_size = TVdataset_len - valid_size
#         train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
#         train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
#         valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
#         test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
#
#         """creat our model """
#         model = HyperTriplet(hp).cuda()#单卡
#
#         # model = HyperTriplet(hp)  # 初始化模型
#         # model = DataParallel(model, device_ids=[0, 1, 2, 3])  # 多卡
#         # model = model.cuda()  # 移动到GPU
#
#         """ set Loss-function"""
#         # Loss = pytorchtools.CustomLoss(hp.poincare_ball, hp.hyperbolic_dim, hp.anchor_factor, hp.loss_rate)
#         Loss = nn.CrossEntropyLoss(weight=None)
#
#         """  select optimizer """
#         # hyperbolic_optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=hp.hyperbolic_lr)
#         hyperbolic_optimizer  = optim.Adam(model.parameters(), lr=0.0001)
#         # 动态学习率
#         # scheduler = StepLR(hyperbolic_optimizer, step_size=50, gamma=0.5)  # 阶梯学习率
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(hyperbolic_optimizer, mode='min', factor=hp.lr_factor, patience=hp.lr_patience)# 动态学习率 # 初始化学习率调度器，监控val_loss，若10个epoch内val_loss不下降，学习率减半
#
#         """ set save_path """
#         save_path = "./" + DATASET
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#
#         """ creat EarlyStop """
#         early_stopping = EarlyStopping(path=save_path, patience=hp.Patience, verbose=True, delta=0, fold=i_fold)
#
#         for epoch in range(1, hp.Epoch + 1):
#             """  TRAIN """
#             print('Training Start ...')
#             # if (epoch - 1) % 10 == 0:
#             #     print("abcd")
#             """ train dataset """
#             train_pbar = tqdm(enumerate(BackgroundGenerator(train_dataset_load)), total=len(train_dataset_load))
#             train_loss_list_on_batch = []
#             model.train()
#             for train_index, train_data in train_pbar:
#                 train_drug_ids, train_target_ids, train_SMILESs, train_FASTAs, train_SMILESs_vector, train_FASTAs_vector, train_labels = train_data
#                 train_SMILESs_vector = train_SMILESs_vector.cuda()
#                 train_FASTAs_vector = train_FASTAs_vector.cuda()
#                 train_labels = train_labels.cuda()
#                 """ step 1 . Clear grad """
#                 hyperbolic_optimizer.zero_grad()
#                 """ step 2 . Calculate output """
#                 predicted_interaction = model(train_drug_ids, train_target_ids, train_SMILESs, train_FASTAs, train_SMILESs_vector, train_FASTAs_vector)
#                 """ step 3. Calculate loss """
#                 train_loss = Loss(predicted_interaction, train_labels)
#                 train_loss_list_on_batch.append(train_loss.item())
#
#                 """ step 4 . Backpropagation """
#                 train_loss.backward()
#                 # check_gradients(model)
#                 """ step 5. Update parameters """
#                 hyperbolic_optimizer.step()
#             train_loss_on_epoch = np.average(train_loss_list_on_batch)
#
#             """ VALID """
#             valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_dataset_load)), total=len(valid_dataset_load))
#             valid_losses_in_epoch = []
#             model.eval()
#             Y, P, S = [], [], []
#             with torch.no_grad():
#                 for valid_i, valid_data in valid_pbar:
#                     valid_drug_ids, valid_target_ids, valid_SMILESs, valid_FASTAs, valid_SMILESs_vector, valid_FASTAs_vector, valid_labels = valid_data
#                     valid_SMILESs_vector = valid_SMILESs_vector.cuda()
#                     valid_FASTAs_vector = valid_FASTAs_vector.cuda()
#                     valid_labels = valid_labels.cuda()
#                     valid_scores = model(valid_drug_ids, valid_target_ids, valid_SMILESs, valid_FASTAs, valid_SMILESs_vector, valid_FASTAs_vector)
#                     valid_loss = Loss(valid_scores, valid_labels)
#                     # #*******************************双曲loss******************************
#                     # valid_losses_in_epoch.append(valid_loss.item())
#                     # #单卡
#                     # anchor = hp.poincare_ball.expmap0(hp.anchor_factor * torch.ones(hp.hyperbolic_dim).cuda())
#                     # valid_dists = hp.poincare_ball.dist(valid_scores, anchor)
#                     # # #多卡
#                     # # batch_size = valid_scores.size(0)
#                     # # anchor_vector = torch.ones(hp.hyperbolic_dim, device=valid_scores.device) * hp.anchor_factor
#                     # # anchor = hp.poincare_ball.expmap0(anchor_vector).expand(batch_size, -1)
#                     # # valid_dists = hp.poincare_ball.dist(valid_scores, anchor)
#                     # valid_predictions = np.where(valid_dists.cpu() <= hp.threshold, 1, 0)
#                     # Y.extend(valid_labels.cpu().data.numpy())
#                     # P.extend(valid_predictions)
#                     # S.extend(-valid_dists.cpu().data.numpy())
#                     # # ****************************交叉熵loss*********************************
#                     valid_labels = valid_labels.to('cpu').data.numpy()
#                     valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
#                     # valid_predictions = np.argmax(valid_scores, axis=1) # np.argmax默认threshold为0.5
#                     valid_predictions = (valid_scores[:, 1] > hp.threshold).astype(int)
#                     valid_scores = valid_scores[:, 1]
#                     valid_losses_in_epoch.append(valid_loss.item())
#                     Y.extend(valid_labels)
#                     P.extend(valid_predictions)
#                     S.extend(valid_scores)
#                     # *************************************************************
#
#             Precision_dev = precision_score(Y, P)  # 根据label和prediction计算precision
#             Recall_dev = recall_score(Y, P)  # 根据label和prediction计算recall
#             Accuracy_dev = accuracy_score(Y, P)  # 根据label和prediction计算Acc
#             AUC_dev = roc_auc_score(Y, S)  # 根据label和score计算ROC曲线的AUC数值
#             precision, recall, _ = precision_recall_curve(Y, S)  # 计算precision-recall
#             PRC_dev = auc(recall, precision)  # 计算曲线下面积
#             valid_loss_a_epoch = np.average(valid_losses_in_epoch)
#
#             epoch_len = len(str(hp.Epoch))
#             print_message = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
#                              f'train_loss: {train_loss_on_epoch:.5f} ' +
#                              f'valid_loss: {valid_loss_a_epoch:.5f} ' +
#                              f'valid_Accuracy: {Accuracy_dev:.5f} ' +
#                              f'valid_Precision: {Precision_dev:.5f} ' +
#                              f'valid_Recall: {Recall_dev:.5f} ' +
#                              f'valid_AUC: {AUC_dev:.5f} ' +
#                              f'valid_PRC: {PRC_dev:.5f} ')
#             print(print_message)
#             # 学习率更新
#             scheduler.step(valid_loss_a_epoch)
#             print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()[0]:.6f}")
#             early_stopping(valid_loss_a_epoch, model)
#             if early_stopping.early_stop:
#                 break
#
#         Acc_on_train, Pre_on_train, Recall_on_train, AUC_on_train, PRC_on_train, average_loss_on_train = test_model(model, train_dataset_load, DATASET, save_path, Loss, "Train", True)
#         Acc_on_valid, Pre_on_valid, Recall_on_valid, AUC_on_valid, PRC_on_valid, average_loss_on_valid = test_model(model, valid_dataset_load, DATASET, save_path, Loss, "Valid", True)
#         Acc_on_test, Pre_on_test, Recall_on_test, AUC_on_test, PRC_on_test, average_loss_on_test = test_model(model, test_dataset_load, DATASET, save_path, Loss, "Test", True)
#
#         """ record the i_fold result txt """
#         whole_result_file = save_path + f"/{i_fold}_fold_result_of_{DATASET}.txt"
#         with open(whole_result_file, "w") as file:
#             file.write(f"Train Acc : {Acc_on_train}" + '\n' + f"Train Pre : {Pre_on_train}" + '\n' + f"Train Recall : {Recall_on_train}" + '\n' + f"Train AUC : {AUC_on_train}" + '\n' + f"Train PRC : {PRC_on_train}" + '\n' + f"Train loss :{average_loss_on_train}" + '\n' + '\n')
#             file.write(f"Valid Acc : {Acc_on_valid}" + '\n' + f"Valid Pre : {Pre_on_valid}" + '\n' + f"Valid Recall : {Recall_on_valid}" + '\n' + f"Valid AUC : {AUC_on_valid}" + '\n' + f"Valid PRC : {PRC_on_valid}" + '\n' + f"Valid loss :{average_loss_on_valid}" + '\n' + '\n')
#             file.write(f"Test Acc : {Acc_on_test}" + '\n' + f"Test Pre : {Pre_on_test}" + '\n' + f"Test Recall : {Recall_on_test}" + '\n' + f"Test AUC : {AUC_on_test}" + '\n' + f"Test PRC : {PRC_on_test}" + '\n' + f"Test loss :{average_loss_on_test}" + '\n' + '\n')
#
#         Accuracy_List_global.append(Acc_on_test)
#         Precision_List_global.append(Pre_on_test)
#         Recall_List_global.append(Recall_on_test)
#         AUC_List_global.append(AUC_on_test)
#         AUPR_List_global.append(PRC_on_test)
#
#     # 获取程序结束时间（以时间戳格式）
#     end_timestamp = time.time()
#     show_result(DATASET, Accuracy_List_global, Precision_List_global, Recall_List_global, AUC_List_global, AUPR_List_global, start_timestamp, end_timestamp)