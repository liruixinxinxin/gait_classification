import torch
from torch.utils.data import Dataset, random_split
from scipy import signal
from tqdm.auto import tqdm
from function import *
from encode import *

class BSA_Dataset(Dataset):
    def __init__(self, data, targets,c):
        self.data = []
        self.targets = []
        t = 100
        for label_num, i in (enumerate(data)):
            sub_arrays = [i[k:k+t] for k in range(0, i.shape[0]-t+1, 10)]
            for sub_array in tqdm(sub_arrays):
                one_sample = []
                # 编码
                # sub_array_separate = pn_junction(sub_array, sub_array.shape[1])
                # scaler = preprocessing.StandardScaler().fit(sub_array)                
                # sub_array = (scaler.fit_transform(sub_array))
                sub_array = sub_array.T
                NUM_TAPS = 11
                CUT_OFF_FREQ = 15
                dlti_filter = signal.dlti(signal.firwin(NUM_TAPS, cutoff=CUT_OFF_FREQ, fs=3600), [1] + [0] * 10, 1)
                time, imp = signal.dimpulse(dlti_filter, n=NUM_TAPS)      
                spiking1=BSA_encode(sub_array[:,:],np.squeeze(imp),threshold=0, channels_num=c//2) #正值部分
                spiking2=BSA_encode(-sub_array[:,:],np.squeeze(imp),threshold=0, channels_num=c//2) #负值部分
                spiking=np.vstack((spiking1,spiking2))       
                # decode=BSA_decoding(spiking,np.squeeze(imp)) #解码
                
                # plt.figure(0,figsize=(10,3))
                # plt.plot(range(1,len(sub_array[0,:])+1),sub_array[0,:],color='red')
                # plt.plot(range(1,len(decode[0,:])+1),decode[0,:], color='green')
                # plt.show()         
                
                
                self.data.append(np.array(spiking).reshape(c,-1))
                self.targets.append(targets[label_num])
            pass
        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)
        pass
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class MyDataset(Dataset):
    def __init__(self, data, targets,c):
        self.data = []
        self.targets = []
        t = 100
        for label_num, i in (enumerate(data)):
            sub_arrays = [i[k:k+t] for k in range(0, i.shape[0]-t+1, 10)]
            for sub_array in tqdm(sub_arrays):
                one_sample = []
                # 编码
                # sub_array_separate = pn_junction(sub_array, sub_array.shape[1])
                # scaler = preprocessing.StandardScaler().fit(sub_array)                
                # sub_array = (scaler.fit_transform(sub_array))
                sub_array = sub_array.T
                for channel in sub_array:
                    one_channel = []
                    sub_array_encode_up, sub_array_encode_down = sigma_delta_encoding2(channel, 
                                                                                    num_intervals=17)
                    one_channel.append(sub_array_encode_up.astype(int))
                    one_channel.append(sub_array_encode_down.astype(int))
                    one_channel = np.array(one_channel).reshape(2,-1)
                    one_sample.append(one_channel)
                self.data.append(np.array(one_sample).reshape(c,-1))
                self.targets.append(targets[label_num])
            pass
        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)
        pass
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
    

class Ann_dataset(Dataset):
    def __init__(self, data, targets,c):
        self.data = []
        self.targets = []
        t = 100
        for label_num, i in tqdm(enumerate(data)):
            sub_arrays = [i[k:k+t] for k in range(0, i.shape[0]-t+1, 10)]
            for sub_array in sub_arrays:
                # 编码
                # sub_array_separate = pn_junction(sub_array, sub_array.shape[1])
                #归一化
                # scaler = preprocessing.StandardScaler().fit(sub_array)
                # sub_array = scaler.fit_transform(sub_array)
                self.data.append(sub_array)
                self.targets.append(targets[label_num])
        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)
        pass
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)