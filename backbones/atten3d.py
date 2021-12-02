import torch
import torch.nn as nn


from torch.autograd import Variable
from config import cfg
from backbones.deform_conv import DeformConv2D


class ChanelWiseFC(nn.Module):  # 逐通道全连接
    def __init__(self, inSize, outSize):
        super(ChanelWiseFC, self).__init__()
        self.fc = nn.Linear(inSize * inSize, outSize)
        self.inSize = inSize
        self.outSize = outSize

    def forward(self, x):
        y = torch.split(x, 1, dim=1)  # 在第一维度(通道)分离
        list = []
        for i in y:
            i = i.view(-1, self.inSize * self.inSize)
            i = self.fc(i)
            list.append(i)

        z = torch.stack(list, dim=1)  # 把前面分裂后的y在第二维度拼接
        return z


class ChanelWiseDeFC(nn.Module):  # 逐通道(反)全连接
    def __init__(self, inSize, outSize):
        super(ChanelWiseDeFC, self).__init__()
        self.fc = nn.Linear(outSize, inSize * inSize)
        self.inSize = inSize
        self.outSize = outSize

    def forward(self, x):
        y = torch.split(x, 1, dim=1)  # 在第一维度(通道)分离
        list = []
        for i in y:
            i = i.view(-1, self.outSize)
            i = self.fc(i)
            list.append(i)
        z = torch.stack(list, dim=1)  # 把前面分裂后的y在第二维度拼接
        z = z.reshape(-1,z.shape[1], self.inSize, self.inSize)  # 这个reshape和批次相关.
        return z


class ChanelWiseConv(nn.Module):  # 逐通道卷积(分组卷积)
    def __init__(self, C, kernel_size, stride, padding):
        super(ChanelWiseConv, self).__init__()
        self.conv_group = nn.Conv2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, groups=C)  # 分组卷积，可以让group设为通道数

    def forward(self, x):
        y = self.conv_group(x)
        return y


class ChanelWiseDeConv(nn.Module):  # 逐通道反卷积(分组反卷积)
    def __init__(self, C, kernel_size, stride, padding, output_padding):
        super(ChanelWiseDeConv, self).__init__()
        self.conv_group = nn.ConvTranspose2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, groups=C, output_padding=output_padding)  # 分组卷积，可以让group设为通道数

    def forward(self, x):
        y = self.conv_group(x)
        return y


class Conv1x1(nn.Module):  # 1*1卷积
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupWiseDeformableConv(nn.Module):  # 这个函数不用了，因为官方已经有分组的了
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(GroupWiseDeformableConv, self).__init__()
        self.k = cfg.num_deformable_groups
        self.conv_deformable = nn.Conv2d(channels // self.k, 18, kernel_size=kernel_size, stride=stride, padding=padding)  # 可变形偏移
        self.conv_offset2d = DeformConv2D(channels // self.k, channels // self.k, kernel_size=kernel_size, padding=padding)  # 这里是已进行偏移后

    def forward(self, x):
        y = torch.split(x, x.shape[1] // self.k, dim=1)  # 在第一维度(通道)分离出k组
        list = []
        for i in y:
            offset = self.conv_deformable(i)  # 学习偏移
            output = self.conv_offset2d(i, offset)  # 这里之后就是那个分组可变形卷积的结果
            list.append(output)
        z = torch.cat(list, dim=1)  # 把前面按通道分裂成k组后的可变性卷积，在通道上拼接，最后形成k组偏移的注意力机制(2*k个偏移量，N是每组的通道数)
        return z


class Atten3D(nn.Module):  # 3D Atten模块,(创新点1)
    def __init__(self, in_planes, inSize, outSize, stage, en_defor=False):  # 输入通道数，fc输入尺寸，fc输出,阶段(对应resnet的stage)
        super(Atten3D, self).__init__()
        self.Conv = Conv1x1(in_planes, in_planes // 16, kernel_size=1, stride=1, padding=0)
        self.ChanelWiseConv = ChanelWiseConv(in_planes // 16, kernel_size=3, stride=2, padding=1)
        self.ChanelWiseFC1 = ChanelWiseFC(inSize, outSize)
        self.ChanelWiseFC2 = ChanelWiseDeFC(inSize, outSize)
        self.ChanelWiseDeConv = ChanelWiseDeConv(in_planes // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.DeConv = Conv1x1(in_planes // 16, in_planes, kernel_size=1, stride=1, padding=0)
        self.stage = stage  # 这个是看stage来设置网络结构的
        self.en_defor = en_defor  # 使用可变形卷积
        # 下面是可变形组卷积的实现(单独测试3D Atten的话，则不引入)
        self.groupWiseDeformableConv = GroupWiseDeformableConv(in_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.Conv(x)  # 1*1卷积在四个阶段都有
        if self.stage != 5:  # 逐通道卷积在stage=5是没有的
            y = self.ChanelWiseConv(y)
        if self.stage==2 or self.stage==3:  # 这里暂定只在stage=2或3的时候用2个逐通道卷积（思路两种可能都有,只有一个可以不写这个if）
            y = self.ChanelWiseConv(y)
        y = self.ChanelWiseFC1(y)  # 每个stage都有
        y = self.ChanelWiseFC2(y)  # 每个stage都有
        if self.stage != 5:  # 逐通道反卷积在stage=5是没有的
            y = self.ChanelWiseDeConv(y)
        if self.stage==2 or self.stage==3:  # 这里暂定只在stage=2或3的时候用2个逐通道反卷积（其实如果只有一个可以不写这个if）
            y = self.ChanelWiseDeConv(y)
        y = self.DeConv(y)  # 每个stage都有
        if self.en_defor==True:  #（创新点3）
            if self.stage==4 or self.stage==5:
                y = self.groupWiseDeformableConv(y)
        return y



