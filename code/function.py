import pandas as pd 
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn import preprocessing
from imu_preprocessing.quantizer import Quantizer
from imu_preprocessing.spike_encoder import ScaleSpikeEncoder, IAFSpikeEncoder
from imu_preprocessing.util.fileIO import IO_binary
from imu_preprocessing.simulators import Simulator_ScaleSpikeEncoder, Simulator_IAFSpikeEncoder
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader
def oversample(dataset,batch_size):
    # 计算每个类别的样本数量
    class_counts = torch.bincount(dataset.targets)
    # 找到样本数量最多的类别，作为基准类别
    max_class_count = torch.max(class_counts)
    # 根据基准类别，计算每个类别需要进行重复采样的数量
    class_weights = max_class_count / class_counts

    # 创建一个权重随机采样器，根据每个样本的类别进行重复采样
    weights = [class_weights[label] for label in dataset.targets]

    sampler = WeightedRandomSampler(weights, len(weights))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


def data_extract_random(path):
    df = pd.read_csv(path)

    df_keen = df.iloc[:, 30:36] 
    df_foot = df.iloc[:, 36:-4]  
    label_column = df.iloc[:, -1]  # 取label列
    # averages = pd.DataFrame()  # 创建空的DataFrame来存储平均值

    # 按照每三列进行迭代
    # for i in range(0, len(df_keen.columns), 3):
    #     # 提取每三列并计算平均值
    #     group = df_keen.iloc[:, i:i+3]
    #     average = group.mean(axis=1)
        
    #     # 添加平均值列到新的DataFrame
    #     column_name = f'Average_{i//3+1}'
    #     averages[column_name] = average
    # df_value = pd.concat([averages, df_foot], axis=1)  # 按列拼接
    df_value = df.iloc[:, [24, 27, 30, 33, 36, 39, 44]]
    df = pd.concat([df_value, label_column], axis=1)  # 按列拼接
    x = []
    y = []
    for i in range(1,5):
        filtered_df = df[(df["label"] == i)]
        filtered_df = ((filtered_df.drop(['label'],axis=1).values))
        x.append(filtered_df)  
        y.append(i-1)  
    return x, y

def IAF_encoder(data,num_channels_input_signal,num_timesteps,iaf_threshold=516):
    
    spike_encoder = IAFSpikeEncoder(iaf_threshold)
    io = IO_binary()
    simulator = Simulator_IAFSpikeEncoder(module=spike_encoder,io=io)
    output_spike_encoder = np.zeros((num_channels_input_signal, num_timesteps), dtype=object)

    # Simulate
    for ch in range(num_channels_input_signal):
        # read a single channel of input signal and process it
        sig_in_ch = data[ch, :].astype(object)
        # obtain the unscaled version of the output filters
        output_spike_encoder[ch, :] = simulator.module.evolve(sig_in_ch)
        
    return output_spike_encoder

def separate_input(arr,len):
    result = []
    for array in arr:
        positive_matrix = np.zeros((1, len))
        negative_matrix = np.zeros((1, len))

        # 将正值和负值分别填充到对应的子矩阵中
        positive_indices = array > 0
        negative_indices = array < 0
        positive_matrix[0, positive_indices] = array[positive_indices]
        negative_matrix[0, negative_indices] = array[negative_indices]
        result.extend(positive_matrix)
        result.extend(np.abs(negative_matrix))
    return np.array(result[0]), np.array(result[1])



    
    
def print_colorful_text(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
    }
    end_color = '\033[0m'
    
    if color in colors:
        print(f"{colors[color]}{text}{end_color}")
    else:
        print(text)
        

def vmem_with_fire(out,rec):

    vmem_with_spike_list = []
    for sample_num,sample in enumerate(out):
        nonzero_rows = torch.nonzero(sample).flatten()
        if len(nonzero_rows) == 0:
            vmem_with_spike = rec['spk_out']['vmem'][sample_num][-1]
        # 计算第一次发射脉冲时，上一个dt对应的最后一层的模电压
        else: vmem_with_spike = rec['spk_out']['vmem'][sample_num][nonzero_rows[0].item()-1]
        vmem_with_spike_list.append(vmem_with_spike)
    data = torch.stack(vmem_with_spike_list, dim=0)
    return data

def vmem_with_fire_xylo(out,rec):
    nonzero_rows = torch.nonzero(out).flatten()
    if len(nonzero_rows) == 0:
        vmem_with_spike = rec['Vmem_out'][-1]
    # 计算第一次发射脉冲时，上一个dt对应的最后一层的模电压
    else: vmem_with_spike = rec['Vmem_out'][nonzero_rows[0].item()-1]

    return torch.tensor(vmem_with_spike)


def encode_labels(labels, num_classes, thr):
    encoded_labels = []
    for label in labels:
        encoded_label = np.zeros(num_classes)
        encoded_label[label] = thr  # Index starts from 0, so subtract 1 from label
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)

def first_no_allzero(tensor):
    non_zero_rows = (tensor != 0).any(dim=1)
    # 找到第一个不全为0的行的索引
    first_non_zero_row_index = (non_zero_rows == 1).nonzero(as_tuple=False)[0, 0]
    # 取出第一个不全为0的行
    result = tensor[first_non_zero_row_index]

    return result

def detect_spike(out,rec):

    vmem_with_spike_list = []

    for sample_num,sample in enumerate(out):
        nonzero_rows = torch.nonzero(sample).flatten()
        if len(nonzero_rows) == 0:#全为0
            vmem_with_spike = rec['spk_out']['vmem'][sample_num][-1]
        # 计算第一次发射脉冲时，上一个dt对应的最后一层的模电压
        else: vmem_with_spike = first_no_allzero(sample)
        vmem_with_spike_list.append(vmem_with_spike)
    data = torch.stack(vmem_with_spike_list, dim=0)
    return data

def detect_spike_xylo(out,rec):
    nonzero_rows = torch.nonzero(out).flatten()
    if len(nonzero_rows) == 0:
        vmem_with_spike = rec['Vmem_out'][-1]
    # 计算第一次发射脉冲时，上一个dt对应的最后一层的模电压
    else: vmem_with_spike = first_no_allzero(out)

    return torch.tensor(vmem_with_spike)