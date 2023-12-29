# - Numpy
import numpy as np
import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous

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
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from model import *
#---------------------#
#    define network   #
#---------------------#
b = 1
c = 14
step = 10
Nhidden = 120

net = synnet
net.load('models/model_first_spike_603_0.8767089236887895.pth')
# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks

# - Detect a connected HDK and import the required support package
connected_hdks, support_modules, chip_versions = find_xylo_hdks()

found_xylo = len(connected_hdks) > 0

if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
    
spec = x.mapper(net.as_graph(), weight_dtype = 'float')
from rockpool.transform import quantize_methods as q

# - Quantize the specification
spec.update(q.global_quantize(**spec))
print(spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = dt)
    print(modSamna)