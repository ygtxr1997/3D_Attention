import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
#from .cbam import *
#from .bam import *
from backbones.Atten3D import  *
from torchinfo import summary

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out

#block在Resnet50以及以上只用BottleNeck，layers在ResNet50的时候用[3,4,6,3]指的是残差块
#network_type是看数据集而定的(如果只用ImageNet做分类则可删去，保留后期可能还做其他任务)
class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes):
        self.inplanes = 64#输入通道(3->64)
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#stage1的卷积，输出为112*112*64
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#stage1的最大池化，输出为56*56*64
            self.avgpool = nn.AvgPool2d(7)#均值池化是在最后一层用的
        else:#这里是测试cifar的(可以删除的，但是后期加入其他任务的话，留着拓展网络结构的)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #下面是残差网络的整体设置
        self.upsample0 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3,
                                            output_padding=1)  # 最开始的卷积的对应deconv(这是使用ER LOSS才用的)
        self.bn1 = nn.BatchNorm2d(64)#bn
        self.relu = nn.ReLU(inplace=True)#relu
        self.SCA1=Atten3D(64,14,12,2)#输出为56*56*64(这是针对imageNet的resnet-50的输入，先写死看效果)
        #3D Atten的反卷积究竟用s2/p1/k3还是用原卷积的s2/k1/p0好
        self.upsample1=nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1)#3D Atten的反卷积(这是使用ER LOSS才用的)
        self.layer1 = self._make_layer(block, 64,  layers[0])#输出为56*56*256
        #全部stride改为1，提前下采样，不用在bottleneck里面下采样
        self.down1=nn.Conv2d(256, 256, kernel_size=1, stride=2,bias=False)#输出为28*28*256
        #self.down1=nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=1,bias=False)
        self.SCA2 = Atten3D(256,7,3,3)#输出为28*28*256
        self.upsample2 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=2, padding=0,output_padding=1) # 3D Atten的反卷积，(这是使用ER LOSS才用的)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)#输出为28*28*512
        self.down2 = nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False) #输出为14*14*512
        #self.down2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.SCA3 = Atten3D(512,7,3,4)#输出为14*14*512
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, padding=0,
                                            output_padding=1)  # 3D Atten的反卷积(ER LOSS使用)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)#输出为14*14*1024
        self.down3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=2, bias=False)#输出为7*7*512
        #self.down3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1,bias=False)  # 输出为7*7*512
        self.SCA4 = Atten3D(1024,7,3,5)#输出为7*7*1024
        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, padding=0,
                                            output_padding=1)  # 3D Atten的反卷积(ER LOSS使用)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)#输出为7*7*2048

        self.fc = nn.Linear(512 * block.expansion, num_classes)#输入为2048，输出为1000
        #权重初始化
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        """downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:"""#这个stride其实可以删去，因为思路中单独下采样
        downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),#改成第一个残差块都是用步长为1的分支下采样(在backbone提前下采样了)
                nn.BatchNorm2d(planes * block.expansion),
        )#这里是指每个bottleneck第一个残差块的那个下采样分支，这里要保留，因为它用来区别其他块
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion#64*4=256
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)#7*7,s2,p3
        x = self.bn1(x)#64
        x = self.relu(x)
        x = self.maxpool(x)#测ImageNet有这个，cifar则没有
        atten = self.SCA1(x)#3D atten branch
        """deAtten1 = self.upsample1(atten)
        deAtten1=self.upsample0(deAtten1)#2个上池化回到原图尺寸"""
        x = x * atten#点乘
        x = self.layer1(x)#联系到conv2_x,输入为56*56*64
        x = self.down1(x)

        atten = self.SCA2(x)  # 3D atten branch,stage3
        """deAtten2 = self.upsample2(atten)
        deAtten2 = self.upsample1(deAtten2)
        deAtten2 = self.upsample0(deAtten2)  # 3个上池化回到原图尺寸"""
        x = x * atten  # 点乘
        x = self.layer2(x)
        x = self.down2(x)

        atten = self.SCA3(x)  # 3D atten branch,stage4
        """deAtten3 = self.upsample3(atten)
        deAtten3 = self.upsample2(deAtten3)
        deAtten3 = self.upsample1(deAtten3)
        deAtten3 = self.upsample0(deAtten3)  # 4个上池化回到原图尺寸"""
        x = x * atten  # 点乘
        x = self.layer3(x)
        x = self.down3(x)

        atten = self.SCA4(x)  # 3D atten branch,stage5
        """deAtten4 = self.upsample4(atten)
        deAtten4 = self.upsample3(deAtten4)
        deAtten4 = self.upsample2(deAtten4)
        deAtten4 = self.upsample1(deAtten4)
        deAtten4 = self.upsample0(deAtten4)  # 5个上池化回到原图尺寸"""
        x = x * atten  # 点乘
        x = self.layer4(x)

        """if self.network_type == "ImageNet":#只测ImageNet所以注释了
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)"""
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x#,deAtten1,deAtten2,deAtten3,deAtten4 #这里是求LOSS部分用到

def ResidualNet(network_type, depth, num_classes):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 50:#现在只有这一支有效果
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes)

    return model

model=ResNet(Bottleneck, [3, 4, 6, 3], "ImageNet", 1000)
#model,d1,d2,d3,d4=ResNet(Bottleneck, [3, 4, 6, 3], "ImageNet", 1000)#这里会在多Loss用到
summary(model,input_size=(64,3, 224, 224))
