from pathlib import Path
import pandas as pd
import numpy as np
import torch
import random
from math import ceil
import torch
import numpy as np
import matplotlib.pyplot as plt
from function import *
from scipy import signal

mul = {}
sigma = {}
Min = {}


def sigma_delta_encoding1(data, num_intervals):
    # 计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份
    thresholds = torch.linspace(data.min(), data.max(), num_intervals+1)[1:-1] # shape (num_intervals-1, )
    # 如果不在(min,max)做等间隔分得阈值，而是固定范围区间为(-2,6)
    # thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]

    # print(thresholds)
    data = np.array(data)

    upper_thresh = []
    lower_thresh = []
    for i in range(len(data)-1):       
        for j in range(num_intervals-1):
            if(data[i]<thresholds[j] and data[i+1]>thresholds[j]):
                num_spike = 1
                for surplus in range(num_intervals-2-j):
                    if data[i+1]>thresholds[j+surplus+1]:
                        num_spike += 1
                upper_thresh.append(num_spike)
                lower_thresh.append(0)
                break
            if (data[i]>thresholds[j] and data[i+1]<thresholds[j]):
                num_spike = 1
                for surplus in range(j-1):
                    if data[i+1]<thresholds[j-(surplus+1)]:
                        num_spike += 1
                upper_thresh.append(0)
                lower_thresh.append(num_spike)
                break
            if j == num_intervals-2:
                upper_thresh.append(0)
                lower_thresh.append(0)
    upper_thresh = np.array(upper_thresh)
    lower_thresh = np.array(lower_thresh)
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    # # 在第一个子图上绘制原始数据
    # ax1.plot(data)
    # ax1.set_title('Data')

    # # 在第二个子图上绘制上阈值线
    # ax2.plot(upper_thresh)
    # ax2.set_title('Upper Threshold')

    # # 在第三个子图上绘制下阈值线
    # ax3.plot(lower_thresh)
    # ax3.set_title('Lower Threshold')

    # # 调整子图之间的间距
    # plt.tight_layout()

    # # 显示图形
    # plt.show()
    # output_matrix = torch.stack([upper_thresh, lower_thresh], dim=0)
    return upper_thresh, lower_thresh

def sigma_delta_encoding2(data, num_intervals):
    # 计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份
    thresholds = np.linspace(data.min(), data.max(), num_intervals) # shape (num_intervals-1, )
    # 如果不在(min,max)做等间隔分得阈值，而是固定范围区间为(-2,6)
    # thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]
    # print(thresholds)
    data = np.array(data)
    M = np.zeros_like(data)
    for i in range(num_intervals-1):
        inds1 = data > thresholds[i]
        inds2 = data < thresholds[i+1]
        M[inds2*inds1] = i
    d_M = M[1:] - M[:-1]
    upper_thresh = np.where(d_M>0,d_M,0)
    lower_thresh = np.where(d_M<0,np.abs(d_M),0)
    return upper_thresh, lower_thresh

def BSA_encode(input, filter, threshold, channels_num=7):
    """
    :param input: 形状为 [23,1024]
    :param filter: 滤波器
    :param threshold: 阈值
    :return:
    """
    data = input.copy()
    output = np.zeros(shape=(data.shape[0], data.shape[1]))
    global mul
    global sigma
    global Min
    for i in range(channels_num):
        mul[i]=np.mean(data[i,:])
        sigma[i]=np.std(data[i,:])
        data[i,:]=(data[i,:]-mul[i])/sigma[i]
    # for i in range(channels_num):
    #     Min[i] = min(data[i, :])
    #     data[i, :] = data[i, :] - Min[i]
    for channel in range(channels_num):
        for i in range(data.shape[1]):
            error1 = 0
            error2 = 0
            for j in range(len(filter)):
                if i + j - 1 <= data.shape[1] - 1:
                    error1 += abs(data[channel][i + j - 1] - filter[j])
                    error2 += abs(data[channel][i + j - 1])
            if error1 <= (error2 - threshold):
                output[channel][i] = 1
                for j in range(len(filter)):
                    if i + j - 1 <= data.shape[1] - 1:
                        data[channel][i + j - 1] -= filter[j]
            else:
                output[channel][i] = 0
    output = np.array(output)
    return output

def BSA_decoding(spikings, filter):
    output = np.zeros(shape=(spikings.shape[0], spikings.shape[1]))
    s = 0
    for channel in range(spikings.shape[0]):
        for t in range(spikings.shape[1]):
            for k in range(len(filter)):
                s += spikings[channel][t - k] * filter[k]
            output[channel][t] = s
            s = 0
    global mul
    global sigma
    global Min
    for channel in range(spikings.shape[0]):
        output[channel,:]=output[channel,:]*(sigma[channel])+mul[channel]
        
    return output