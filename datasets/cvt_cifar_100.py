import torch
import torchvision

import mxnet as mx

import os
import numpy as np

from PIL import Image

dataset_path = '/home/yuange/dataset/cifar-100'

train_data = torchvision.datasets.CIFAR100(
    root='/home/yuange/dataset/cifar-100',
    train=True,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ToPILImage(),
        ]),
    download=True
)

# test_data = torchvision.datasets.CIFAR100(
#     root='/home/yuange/dataset/cifar-100',
#     train=False,
#     transform=torchvision.transforms.ToPILImage(),
#     download=True
# )

num_sample = train_data.__len__()
print(num_sample)
img = train_data.__getitem__(0)
print(img[0])
img = img[0]
# img[0].save('before.jpg')

idx_path = os.path.join(dataset_path, 'train.idx')
rec_path = os.path.join(dataset_path, 'train.rec')
write_record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

# img = np.asarray(img[0])
img = np.asarray(img) * 255
# print(img.shape)
# import copy
# r = img[0]
# g = img[1]
# b = img[2]
# tmp = copy.deepcopy(r)
# img[0] = b
# img[2] = tmp
img = img.transpose((1, 2, 0))  # H, W, C
img = img[...,[2,1,0]]  # RGB to BGR
header = mx.recordio.IRHeader(flag=0, label=1.0, id=0, id2=0)
s = mx.recordio.pack_img(header, img, quality=95, img_fmt='.jpg')

write_record.write_idx(0, s)
write_record.close()

read_record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

item = read_record.read()
header, s = mx.recordio.unpack(item)
# xxx = mx.image.image()
img = mx.image.imdecode(s).asnumpy()
img = Image.fromarray(img)
img.save('test.jpg')



