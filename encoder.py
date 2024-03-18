import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters, to_pool=True, pool_type="strided"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        if to_pool and pool_type=="strided":
            self.pool = nn.Conv2d(num_filters, num_filters, 3, 2, 1)
        elif to_pool and pool_type=="max_pool":
            self.pool = nn.MaxPool2d(2,2)
        self.batch_norm1 = nn.BatchNorm2d(num_filters)
        self.batch_norm2 = nn.BatchNorm2d(num_filters)
        self.to_pool =to_pool
    def forward(self,x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        if self.to_pool:
            x = self.pool(x)                  
        return x