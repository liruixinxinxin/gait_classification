# - Numpy
import numpy as np
import torch
import pickle
import pandas as pd
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
# - Import the computational modules and combinators required for the network
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset,random_split,DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x
from rockpool.nn.networks import SynNet,WaveSenseNet    
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from rockpool.nn.combinators import Sequential, Residual
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential
from rockpool.nn.modules import LIFBitshiftTorch
from function import *
from encode import *
from model import *
from dataset import *
from pathlib import Path
#---------------------#
#    define network   #
#---------------------#
b = 1
c = 14
step = 10
Nhidden = 120
thr = 0.82
thr_out = 1.5
# net = WaveSenseNet(n_channels_in=c, 
#                    n_classes=4, 
#                    dilations=[2, 4, 8, 16],
#                    threshold=thr, 
#                    threshold_out=thr_out)


net = synnet
# netpath = Path('/home/liruixin/workspace/gait_classification/models')
# for i in netpath.rglob('*.pth'):
net.load('models/model_first_spike_1761_0.8650260999254288.pth')
# print(i.parts[-1])
#---------------------#
#      quantize       ‘
#---------------------#
g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# spec['dash_mem_out'] = np.array([0, 1, 0, 0])
# spec['threshold_out'] = np.array([100,100,100,100])
# - Use rockpool.devices.xylo.config_from_specification
# spec['threshold'] = np.full((60,),20,dtype=int).flatten()
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)
print('网络部署完毕')


print('正在加载数据...')
with open('data/dataset/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)
print('正在进行升采样...')
test_dataloader = oversample(test_dataset,batch_size=1)
seed = 42
torch.manual_seed(seed)
# test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(seed))
print('升采样完成')

def model_test(test_dataloader, model):
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    test_preds = []
    test_targets = []
    for batch, target in tqdm(test_dataloader):
        with torch.no_grad():
            batch = ((batch*2).clip(0,15).transpose(1, 2))
            batch = batch.to(torch.int32).squeeze().numpy()
            target = target.type(torch.LongTensor).numpy()
            model.reset_state()
            out_model, _, rec = model(batch, record=True)
            # out = detect_spike_xylo(torch.tensor(out_model), rec)
            # data = (rec['Vmem_out'])
            # if target.item() == 1:
            # fig, ax = plt.subplots()
            # for i in range(4):
            #     ax.plot(data[:, i], label=f"label{i}")
            # # 添加图例
            # ax.legend()
            # # 显示图形
            # plt.title(target)
            # plt.show()
            
            # plt.plot(np.sum(out_model,axis=0))
            # plt.legend()
            # # 显示图形
            # plt.title(target)
            # plt.show()
            
            
            # pred = out_model.argmax(0).detach()
            pred = np.sum(out_model,axis=0).argmax(0)
            test_preds.append(pred)
            test_targets.append(target)

    f1 = f1_score(test_targets, test_preds, average="macro")
    _, test_precision, test_recall, _ = precision_recall_fscore_support(
        test_targets, test_preds, labels=np.arange(4)
    )
    test_accuracy = accuracy_score(test_targets, test_preds)
    cm = confusion_matrix(test_targets, test_preds)
    f1s.append(f1)
    precision.append(test_precision)
    recall.append(test_recall)
    accuracy.append(test_accuracy)
    cmlist.append(cm)
    print(f"F1 Score = {f1}")
    print(f"Val Precision = {test_precision}, Recall = {test_recall}")
    print(f"Val Accuracy = {test_accuracy}")
    print("Confusion Matrix:")
    print(cm)
    
    
model_test(test_dataloader, model=modSim)