# import package
import numpy as np
import torch
import pickle 
from tqdm.auto import tqdm
from function import *
from torch.utils.data import random_split
from dataset import *
from model import *
from train import *
import optuna
from optuna.trial import TrialState
from torch.utils.data import Dataset,DataLoader
from rockpool.nn.networks import SynNet, WaveSenseNet
from rockpool.nn.modules import LIFExodus, LIFTorch
print('正在加载数据...')
c = 16
Input_path_train = '/home/liruixin/workspace/gait_classification/data/data_collect_训练.csv'
Input_path_test = '/home/liruixin/workspace/gait_classification/data/data_collect_测试.csv'
features_train, labels_train = data_extract_random(Input_path_train)
features_test, labels_test = data_extract_random(Input_path_test)

triggle = input ('请问您已经在本地保存了dataset吗?(y/n):')
if triggle == 'y':
    with open('/home/liruixin/workspace/gait_classification/data/dataset/train_dataset.pkl', 'rb') as file:
        train_dataset = pickle.load(file)
    with open('/home/liruixin/workspace/gait_classification/data/dataset/test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)
if triggle == 'n':   
    train_dataset = MyDataset(features_train, labels_train,c)
    test_dataset =MyDataset(features_test, labels_test,c)
    # train_dataset = Ann_dataset(features_train, labels_train)
    # test_dataset =Ann_dataset(features_test, labels_test)
    # 假设您的标准数据集对象为 dataset
    # dataset_size = len(dataset)
    # train_size = int(0.7 * dataset_size)  # 训练集占总数据集的比例，此处为 70%
    # 根据比例随机划分为训练集和测试集
    # train_dataset, test_dataset = random_split(dataset, [train_size, dataset_size - train_size])
    with open('/home/liruixin/workspace/gait_classification/data/dataset/train_dataset.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    with open('/home/liruixin/workspace/gait_classification/data/dataset/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)
pass

print('正在进行升采样...')
train_dataloader = oversample(train_dataset,batch_size=4096)
# test_dataloader = oversample(test_dataset,batch_size=len(test_dataset))
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))
print('升采样完成')
device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
# 创建模型实例
ann = Myann()

def objective(trial):
    global train_dataloader
    global test_dataloader
    thr = trial.suggest_float('thr',0.5,1.38)
    thr_out = trial.suggest_float('thr—out',1.5,4.0)
    snn = WaveSenseNet(n_channels_in=16, 
                    n_classes=4, 
                    dilations=[2,16,32],
                    threshold=thr, 
                    threshold_out=thr_out,
                    neuron_model = LIFTorch)
    model=snn
    Nout = 4
    criterion = nn.MSELoss()
    criterion.to(device)
    model.to(device)
    print('device:',device)
    opt = optim.Adam(model.parameters().astorch(), lr=0.000172)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(80):
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.transpose(1, 2)
            batch = batch.to(torch.float32).to(device)
            target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
            model.reset_state()
            opt.zero_grad()
            out_model, _, rec = model(batch, record=True)
            # peaks = torch.sum(out, dim=1)
            # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
            out = vmem_with_fire(out_model, rec).to(device)
            # peaks = torch.sum(out, dim=1).to(device)
            loss = criterion((out), target_loss)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = out.argmax(1).detach().to(device)
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item() / len(train_dataloader)

        sum_f1 = f1_score(train_targets, train_preds, average="macro")
        _, train_precision, train_recall, _ = precision_recall_fscore_support(
            train_targets, train_preds, labels=np.arange(4)
        )
        train_accuracy = accuracy_score(train_targets, train_preds)

        # print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        # print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        # print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.transpose(1, 2)
                batch = batch.to(torch.float32).to(device)
                target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
                model.reset_state()
                out_model, _, rec = model(batch, record=True)
                # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
                # peaks = torch.sum(out, dim=1).to(device)
                # pred = peaks.argmax(1).detach().to(device)
                out = vmem_with_fire(out_model, rec).to(device)
                loss = criterion((out), target_loss)
                pred = out.argmax(1).detach().to(device)
                test_loss += loss.item() / len(test_dataloader)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(4)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"Val accuracy = {test_accuracy},f1 = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")

        if trial.should_prune()  or (epoch >= 4 and f1 < 0.2):
            raise optuna.exceptions.TrialPruned()
    return test_accuracy
    

if __name__ == "__main__":
    # 创建Optuna的Study对象
    study = optuna.create_study(direction="maximize")

    # 运行优化过程，最多进行100次试验，超时时间为600秒
    study.optimize(objective, n_trials=200)

    # 获取被剪枝的试验和完成的试验
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # 打印研究统计信息
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    # 打印最佳试验结果
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
