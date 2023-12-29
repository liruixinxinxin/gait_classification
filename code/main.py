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
from torch.utils.data import Dataset,DataLoader
from rockpool.nn.networks import SynNet, WaveSenseNet
from rockpool.nn.modules import LIFExodus, LIFTorch
print('正在加载数据...')
c = 14
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
    train_dataset = BSA_Dataset(features_train, labels_train,c)
    test_dataset =BSA_Dataset(features_test, labels_test,c)
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
train_dataloader = oversample(train_dataset,batch_size=2048)
test_dataloader = oversample(test_dataset,batch_size=len(test_dataset)//2)
# test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))
print('升采样完成')
device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
# 创建模型实例
ann = Myann()

snn = synnet

if __name__ == '__main__':
    print('训练开始')
    # ann_train(device=device,
    #           train_dataloader=train_dataloader,
    #           test_dataloader=test_dataloader,
    #           ann=ann,
    #           len_train = len(train_dataset),
    #           len_test = len(test_dataset),
    #           data_channel=c)
    
    snn_train_spike(device=device,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              model=snn,
              thr_out=thr_out)
    # snn_train(device=device,
    #           train_dataloader=train_dataloader,
    #           test_dataloader=test_dataloader,
    #           model=snn,
    #           thr_out=thr_out)

    # snn_train_vmem(device=device,
    #         train_dataloader=train_dataloader,
    #         test_dataloader=test_dataloader,
    #         model=snn,
    #         thr_out=thr_out)
