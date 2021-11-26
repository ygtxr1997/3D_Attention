from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):##这里是默认最好的值
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:#返回原图
            return img

        for attempt in range(100):#原文是死循环
            area = img.size()[1] * img.size()[2] #图片的面积
       
            target_area = random.uniform(self.sl, self.sh) * area#裁剪的面积
            aspect_ratio = random.uniform(self.r1, 1/self.r1)#裁剪接受率（对长宽有用）
            # 裁剪后的长宽
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            #如果不越界
            if w < img.size()[2] and h < img.size()[1]:
                #得到左上角点
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3: #这应该是那个现像素均值（文中是随机取像素的吧）
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]#[0,x:x1+h,y1:y1+w]指的是对x-x1+h的元素全部赋值为均值
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

