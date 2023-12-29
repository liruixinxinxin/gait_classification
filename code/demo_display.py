import tkinter as tk

import numpy as np
import torch
import matplotlib.image as mpimg

from tqdm.auto import tqdm

from tqdm.auto import tqdm 
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#==============================================================#
                        #定义网络和Xylosim#                    
#==============================================================#
# - Numpy
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x
from model import *
net = synnet
net.load('models/model_first_spike_1761_0.8650260999254288.pth')
g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
spec.update(q.global_quantize(**spec))
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)
print('网络部署完毕')
#==============================================================#
#完成#
#==============================================================#

import pandas as pd
import tkinter as tk
from tkinter import filedialog
from encode import *
path = None
array = None
t = 0
fig1 = plt.figure(figsize=(2.5, 1.5))
fig2 = plt.figure(figsize=(2.5, 1.5))
def open_file():
    global path
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        with open(file_path, "r") as file:
            content = file.read()
            text_box.delete("1.0", tk.END)  
            text_box.insert(tk.END, content)

        path = file_path
def plot_graph():
    global path
    global array 
    global fig1
    plt.figure(fig1.number)
    plt.clf()
    df = pd.read_csv(path)
    label_column = df.iloc[:, -1]
    df_value = df.iloc[:, [24, 27, 30, 33, 36, 39, 44]]
    array = df_value[t:t+100].values
    plt.plot(array)
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.title("figure")

    fig = plt.gcf()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()

    canvas.get_tk_widget().grid(row=1, column=1, padx=10, pady=10)
    plot_spiking()

def plot_spiking(): 
    global t
    global fig2
    plt.figure(fig2.number)
    NUM_TAPS = 11
    CUT_OFF_FREQ = 15
    dlti_filter = signal.dlti(signal.firwin(NUM_TAPS, cutoff=CUT_OFF_FREQ, fs=3600), [1] + [0] * 10, 1)
    time, imp = signal.dimpulse(dlti_filter, n=NUM_TAPS)   
    spiking1=BSA_encode(array[:,:],np.squeeze(imp),threshold=0, channels_num=7) #正值部分
    spiking2=BSA_encode(-array[:,:],np.squeeze(imp),threshold=0, channels_num=7) #负值部分
    spiking=np.vstack((spiking1,spiking2)).T 
    num_channels = spiking.shape[0]
    # print(f'adawdawddw:{spiking.shape}')
    fig, axes = plt.subplots(num_channels, 1, figsize=(2, 0.5 * num_channels), sharex=True, sharey=True)

    # 定义一组不同的颜色，可以根据需要增加更多颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'lime', 'teal', 'navy']

    for channel_idx in range(num_channels):
        axes[channel_idx].plot(spiking[channel_idx, :], color=colors[channel_idx])
        axes[channel_idx].set_xticks([])  # 去除 X 轴刻度和标签
        axes[channel_idx].set_yticks([])  # 去除 Y 轴刻度和标签
        axes[channel_idx].axis('off')  # 去除子图的边框和背景


    plt.subplots_adjust(hspace=0)


    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    canvas.get_tk_widget().grid(row=2, column=1, padx=10, pady=10)
    t = t + 10

root = tk.Tk()
root.title("Tkinter Demo")
root.geometry("2400x1200")

# 在左上角生成一个按钮
button_select_file = tk.Button(root, text="选择表格文件", command=open_file)
button_select_file.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
text_box = tk.Text(root, wrap=tk.WORD)
text_box.grid(row=1, column=0, padx=5, pady=5)

# button_plot_graph = tk.Button(root, text="绘制图形", command=plot_graph)
# button_plot_graph.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

button_plot_graph = tk.Button(root, text="开始检测", command=plot_graph)
button_plot_graph.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

root.mainloop()
