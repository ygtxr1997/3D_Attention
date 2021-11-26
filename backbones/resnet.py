"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from torchinfo import summary
from backbones.Atten3D import  *
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101']


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100,
                 fp16=False):
        super().__init__()

        self.fp16 = fp16

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1

        # --------------------------stage 2------------------------------
        self.Atten2 = Atten3D(64, 8, 12, 2)  # 输出为32*32*64
        """self.upsample2 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1) # conv1的对应deconv(这是使用ER LOSS才用的),针对ImageNet的"""
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1) #输出为32*32*256
        # 全部stride改为1，提前下采样，不用在bottleneck里面下采样(因为要满足3D Atten的输入尺寸)
        self.down2 = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False)#输出为16*16*256

        #--------------------------stage 3------------------------------
        self.Atten3 = Atten3D(256, 4, 3, 3)  # 输出为16*16*256
        """self.upsample3 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=2, padding=0,
                                            output_padding=1)  # 3D Atten的反卷积，(这是使用ER LOSS才用的)"""
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)#输出为16*16*512
        self.down3 = nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False)# 输出为8*8*512

        # --------------------------stage 4------------------------------
        self.Atten4 = Atten3D(512, 4, 3, 4)  # 输出为8*8*512
        """self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, padding=0,
                                            output_padding=1)  # 3D Atten的反卷积(ER LOSS使用)"""
        self.conv4_x = self._make_layer(block, 256, num_block[2], 1)#输出为8*8*1024
        self.down4 = nn.Conv2d(1024, 1024, kernel_size=1, stride=2, bias=False)# 输出为4*4*1024

        # --------------------------stage 5------------------------------
        self.Atten5 = Atten3D(1024, 4, 3, 5)  # 输出为4*4*1024
        """self.upsample5 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, padding=0,
                                            output_padding=1)  # 3D Atten的反卷积(ER LOSS使用)"""
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)#输出为4*4*2048
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))#输出为1*1*2048
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        # 这里在stage3-stage5的下采样被提前了，所以残差块里面的步长都变为1(因为要适配3D Atten的输入,对照上面的每个self.conv_x，都是stride=1)
        # 咨询了老板，那个提前的下采样(也就是self.down)，可以用3*3/s2/pad1或者用原残差里面的1*1/s2/pad0(说试试效果，我这里全用了后者)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            # ----------------stage1----------------
            output = self.conv1(x)

            #----------------stage2----------------
            atten = self.Atten2(output)  # 3D atten branch
            """
            deAtten2 = self.upsample2(atten)  #deConv for ER LOSS
            """
            output = output * atten  # dot product
            output = self.conv2_x(output)
            output = self.down2(output)

            # ----------------stage3----------------
            atten = self.Atten3(output)
            """
            deAtten3 = self.upsample3(atten)
            deAtten3 = self.upsample2(deAtten3)  # two deConv for ER LOSS
            """
            output = output * atten  # dot product
            output = self.conv3_x(output)
            output = self.down3(output)

            # ----------------stage4----------------
            atten = self.Atten4(output)
            """
            deAtten4 = self.upsample4(atten)
            deAtten4 = self.upsample3(deAtten4)
            deAtten4 = self.upsample2(deAtten4) #three deConv for ER LOSS
            """
            output = output * atten  # dot product
            output = self.conv4_x(output)
            output = self.down4(output)

            # ----------------stage5----------------
            atten = self.Atten5(output)
            """
            deAtten5 = self.upsample5(atten)
            deAtten5 = self.upsample4(deAtten5)
            deAtten5 = self.upsample32=(deAtten5)
            deAtten5 = self.upsample2(deAtten5) #four deConv for ER LOSS
            """
            output = output * atten  # dot product
            output = self.conv5_x(output)

            #--------------pool+fc---------------
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
        output = self.fc(output.float() if self.fp16 else output)

        return output#,deAtten2,deAtten3,deAtten4,deAtten5  ##for the ER LOSS


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):#当前模型只有resnet50，适用于32*32的cifar数据集
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    model = ResNet(BottleNeck, [3, 4, 6, 3], 100)
    summary(model, input_size=(64, 3, 32, 32))
    print('nothing')

