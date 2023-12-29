# 本次训练将数据编码后导入ANN
import pandas as pd
import numpy as np
import torch
import random
import pickle
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from torch.utils.data import Dataset,DataLoader
from function import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm.auto import tqdm
from collections import Counter 
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

b = 4096
t = 100
n_class = 4

def ann_train(device,train_dataloader,test_dataloader,ann,len_train,len_test,data_channel):
    epochs = 2000
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    ann.to(device)
    # opt = optim.Adam(ann.parameters(), lr=0.000172)
    opt = optim.Adam(ann.parameters(), lr=0.000172, weight_decay=0.001)
    
    for epoch in range(epochs):
        ann.train()
        total_loss = 0.0
        correct_predictions = 0
        
        for inputs, labels in tqdm(train_dataloader):
            inputs = (inputs.reshape(-1,1,t,data_channel)).to(device)
            # inputs[inputs == -1] = 2
            labels = labels.to(device)
            opt.zero_grad()
            outputs = (ann(inputs)).to(device)
            # outputs = torch.sum(out,dim=1)
            # labels = labels.to(torch.long)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            # _, label = torch.max(labels, 1)
            # 添加L2正则化损失项
            l2_regularization_loss = 0.0
            for param in ann.parameters():
                l2_regularization_loss += torch.norm(param, p=2)

            total_loss += loss.item() + 0.1 * l2_regularization_loss.item()
            correct_predictions += torch.sum(predicted == labels).item()
            
            
            # total_loss += loss.item()
            # correct_predictions += torch.sum(predicted == labels).item()
            
            loss.backward()
            opt.step()
        
        # 计算测试集的损失和准确率
        test_loss = 0.0
        test_correct_predictions = 0
        
        ann.eval()
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.reshape(-1,1,t,data_channel).to(device)
                # inputs[inputs == -1] = 2
                labels = labels.to(device)
                outputs = ann(inputs).to(device)
                # outputs = torch.sum(outputs,dim=1)
                labels = labels.to(torch.long)
                loss = criterion(outputs, labels).to(device)
                _, predicted = torch.max(outputs, 1)
                # _, label = torch.max(labels, 1)
                
                test_loss += loss.item()
                test_correct_predictions += torch.sum(predicted == labels).item()
        
        # 输出训练集和测试集的损失和准确率
        train_loss = total_loss / len_train
        train_accuracy = correct_predictions / len_train
        test_loss /= len_test
        test_accuracy = test_correct_predictions / len_test

        print("Epoch", epoch+1)
        print("Train Set:")
        print("Loss:", train_loss)
        print("Accuracy:", train_accuracy)
        print("Test Set:")
        print("Loss:", test_loss)
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        ann.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.reshape(-1,1,t,data_channel).to(device)
                labels.to(device)
                outputs = ann(inputs).to(device)
                # outputs = torch.sum(outputs,dim=1)
                _, predicted = torch.max(outputs, 1)
                # _, label = torch.max(labels, 1)
                true_labels.extend(labels.tolist())
                predicted_labels.extend(predicted.tolist())
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)
        
        
def snn_train(device, train_dataloader, test_dataloader, model, thr_out):
    Nout = n_class
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
    for epoch in range(1000):
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.transpose(1, 2)
            batch = batch.to(torch.float32).to(device)
            target_loss = (torch.tensor(encode_labels(target,Nout,thr_out+0.5))).float().to(device)
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

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.transpose(1, 2)
                batch = batch.to(torch.float32).to(device)
                # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
                model.reset_state()
                out_model, _, rec = model(batch, record=True)
                # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
                # peaks = torch.sum(out, dim=1).to(device)
                # pred = peaks.argmax(1).detach().to(device)
                out = detect_spike(out_model, rec)
                # loss = criterion((out), target_loss)
                pred = out.argmax(1).detach().to(device)
                # test_loss += loss.item() / len(test_dataloader)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(4)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        if (epoch %100 == 0 and epoch != 0) or np.all(test_precision > 0.81) or test_accuracy>0.84:
            model.save(f'/home/liruixin/workspace/gait_classification/models/model_{epoch}_{test_accuracy}.pth')
            print('模型已保存')
            np.save('//home/liruixin/workspace/gait_classification/train_data_record/loss.npy', losslist)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/f1s.npy', f1s)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/precision.npy', precision)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/recall.npy', recall)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/accuracy.npy', accuracy)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/cm.npy', cmlist)
            print('训练已完成，训练参数已保存') 

def snn_train_spike(device, train_dataloader, test_dataloader, model, thr_out):
    Nout = n_class
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    model.to(device)
    print('device:',device)
    opt = optim.Adam(model.parameters().astorch(), lr=0.000172)
    # scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(2000):
        # scheduler.step()
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.transpose(1, 2)
            batch = batch.to(torch.float32).to(device)
            # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out+0.2))).float().to(device)
            target_loss = target.to(device)
            model.reset_state()
            opt.zero_grad()
            out_model, _, rec = model(batch, record=True)
            # peaks = torch.sum(out, dim=1)
            # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
            # out = vmem_with_fire(out_model, rec).to(device)
            # peaks = torch.sum(out, dim=1).to(device)
            out = torch.sum(out_model,dim=1)
            loss = criterion(out, target_loss)
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

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.transpose(1, 2)
                batch = batch.to(torch.float32).to(device)
                # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
                model.reset_state()
                out_model, _, rec = model(batch, record=True)
                # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
                # peaks = torch.sum(out, dim=1).to(device)
                # pred = peaks.argmax(1).detach().to(device)
                # out = detect_spike(out_model, rec).to(device)
                # loss = criterion((out), target_loss)
                out = torch.sum(out_model,dim=1)
                pred = out.argmax(1).detach().to(device)
                # test_loss += loss.item() / len(test_dataloader)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(4)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        if epoch %100 == 0 or np.all(test_precision > 0.85):
            model.save(f'/home/liruixin/workspace/gait_classification/models/model_first_spike_{epoch}_{test_accuracy}.pth')
            print('模型已保存')
            np.save('//home/liruixin/workspace/gait_classification/train_data_record/loss.npy', losslist)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/f1s.npy', f1s)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/precision.npy', precision)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/recall.npy', recall)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/accuracy.npy', accuracy)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/cm.npy', cmlist)
            print('训练已完成，训练参数已保存') 
            
def snn_train_vmem(device, train_dataloader, test_dataloader, model, thr_out):
    Nout = n_class
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    print('device:',device)
    opt = optim.Adam(model.parameters().astorch(), lr=0.00172)
    # scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(2000):
        # scheduler.step()
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.transpose(1, 2)
            batch = batch.to(torch.float32).to(device)
            # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
            model.reset_state()
            opt.zero_grad()
            out_model, _, rec = model(batch, record=True)
            out = torch.sum(rec['4_LIFTorch']['vmem'],dim=1).to(device)
            # peaks = torch.sum(out, dim=1).to(device)
            loss = criterion((out), target)
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

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0

        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.transpose(1, 2)
                batch = batch.to(torch.float32).to(device)
                # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out))).float().to(device)
                model.reset_state()
                out_model, _, rec = model(batch, record=True)
                # peaks = torch.sum(rec['spk_out']['vmem'], dim=1).to(device)
                # peaks = torch.sum(out, dim=1).to(device)
                # pred = peaks.argmax(1).detach().to(device)
                out = torch.sum(rec['4_LIFTorch']['vmem'],dim=1)
                # loss = criterion((out), target_loss)
                pred = out.argmax(1).detach().to(device)
                # data = (rec['spk_out']['vmem']).cpu().detach().numpy().reshape(100,4)
                # fig, ax = plt.subplots()
                # for i in range(4):
                #     ax.plot(data[:,i], label=f"label{i}")
                # ax.legend()
                # plt.title(target)
                # plt.show()
                # test_loss += loss.item() / len(test_dataloader)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()

        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(4)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        if (epoch %100 == 0 and epoch != 0) or np.all(test_precision > 0.81) or test_accuracy>0.86:
            model.save(f'/home/liruixin/workspace/gait_classification/models/model_first_spike_{epoch}_{test_accuracy}.pth')
            print('模型已保存')
            np.save('//home/liruixin/workspace/gait_classification/train_data_record/loss.npy', losslist)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/f1s.npy', f1s)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/precision.npy', precision)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/recall.npy', recall)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/accuracy.npy', accuracy)
            np.save('/home/liruixin/workspace/gait_classification/train_data_record/cm.npy', cmlist)
            print('训练已完成，训练参数已保存') 