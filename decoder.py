import torch.nn as nn
import torch.nn.functional as F
import torch

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters,upsample_type="transpose"):
        super().__init__()
        if upsample_type=="transpose":
            self.up = nn.ConvTranspose2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif upsample_type=="upsample":
            self.up= nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(num_filters*2, num_filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
    def forward(self,x, prev_layer):
        x = self.up(x)
        x = torch.cat([x, prev_layer], dim=1)
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        return x
