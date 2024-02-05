import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np


class multi_stream(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(multi_stream, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 2, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):

        x = self.conv(x)
        return x

class concatanated_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(concatanated_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 2, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):

        x = self.conv(x)
        return x

class last_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(last_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 2, padding=0),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 2, padding=0),
        )

    def forward(self, x):

        x = self.conv(x)
        return x

class EPINET(nn.Module):

    def __init__(self, input_ch, filter_num, stream_num):
        super(EPINET, self).__init__()

        filter_concatamated = filter_num * stream_num

        self.multistream1 = multi_stream(in_ch = input_ch, out_ch = filter_num)
        self.multistream2 = multi_stream(in_ch = filter_num, out_ch = filter_num)
        self.multistream3 = multi_stream(in_ch = filter_num, out_ch = filter_num)
        self.concatanated_conv1 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv2 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv3 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv4 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv5 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv6 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.concatanated_conv7 = concatanated_conv(in_ch = filter_concatamated, out_ch = filter_concatamated)
        self.last_conv1 = last_conv(in_ch = filter_concatamated, out_ch = 1)

    def forward(self, x_0d, x_90d, x_45d, x_m45d):
        x_0d = self.multistream1(x_0d)
        x_0d = self.multistream2(x_0d)
        x_0d = self.multistream3(x_0d)

        x_90d = self.multistream1(x_90d)
        x_90d = self.multistream2(x_90d)
        x_90d = self.multistream3(x_90d)

        x_45d = self.multistream1(x_45d)
        x_45d = self.multistream2(x_45d)
        x_45d = self.multistream3(x_45d)

        x_m45d = self.multistream1(x_m45d)
        x_m45d = self.multistream2(x_m45d)
        x_m45d = self.multistream3(x_m45d)

        x = torch.cat((x_90d, x_0d, x_45d, x_m45d), dim=1)

        x = self.concatanated_conv1(x)
        x = self.concatanated_conv2(x)
        x = self.concatanated_conv3(x)
        x = self.concatanated_conv4(x)
        x = self.last_conv1(x)

        return x
