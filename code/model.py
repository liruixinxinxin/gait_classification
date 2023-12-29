# 本次训练将数据编码后导入ANN

import torch.nn as nn
from rockpool.nn.networks import SynNet


# 训练网络
class Myann(nn.Module):
    def __init__(self):
        super(Myann, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 1), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(14400, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x.float()))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        # output = self.softmax(x)
        return output


thr_out = 5.0
synnet = SynNet(n_classes=4,
                n_channels=14,
                tau_mem = 0.02,
                threshold_out=thr_out)

              