import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class FCDiscriminator(nn.Module):
    def __init__(self, num_channels, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv3d(num_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        # self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv3d(ndf * 2, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        # x = F.gelu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        # x = F.gelu(x)
        # x = self.conv3(x)
        # x = self.leaky_relu(x)
        # x = self.conv4(x)
        # x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


