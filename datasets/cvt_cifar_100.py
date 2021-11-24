import torch
import torchvision
from torch.utils import data
from tqdm import tqdm

import mxnet as mx

import os
import numpy as np
import numbers

from PIL import Image

dataset_path = '/home/yuange/dataset/cifar-100'

train_data = torchvision.datasets.CIFAR100(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    download=True
)

test_data = torchvision.datasets.CIFAR100(
    root=dataset_path,
    train=False,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    download=True
)


def start_convert(target='train', dataset=train_data, num_classes=100, batch_size=128):

    idx_path = os.path.join(dataset_path, target + '.idx')
    rec_path = os.path.join(dataset_path, target + '.rec')
    write_record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    # The first Header saves 'cnt_samples' and 'cnt_classes'
    first_header = mx.recordio.IRHeader(flag=0, label=[len(dataset) + 1, num_classes],
                                        id=1, id2=0)  # flag will be set as len(label)
    first_s = mx.recordio.pack_img(first_header, np.zeros((32, 32, 3)), quality=100, img_fmt='.png')
    header = first_header
    write_record.write_idx(0, first_s)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_len = len(dataloader)
    for idx, (images, labels) in enumerate(dataloader):

        images = np.asarray(images) * 255
        images = images.transpose((0, 2, 3, 1))  # (B, C, H, W) to (B, H, W, C)
        images = images[..., [2, 1, 0]]  # RGB to BGR

        batch_cnt = images.shape[0]
        for b in range(batch_cnt):
            img = images[b]
            label = labels[b].item()  # Convert Tensor to int
            assert type(label) == int

            header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
            s = mx.recordio.pack_img(header, img, quality=100, img_fmt='.png')

            write_record.write_idx(1 + idx * batch_size + b, s)

        print('[%d/%d]: Converting, target=%s, num_classes=%d' % (idx, total_len, target, num_classes))
    write_record.close()


if __name__ == '__main__':

    """ Start Convert """
    start_convert('train', train_data, 100, 128)
    start_convert('test', test_data, 100, 128)

    """ Check for train """
    print('=====> Checking training dataset')
    check_target = 'train'
    check_idx_path = os.path.join(dataset_path, check_target + '.idx')
    check_rec_path = os.path.join(dataset_path, check_target + '.rec')
    read_record = mx.recordio.MXIndexedRecordIO(check_idx_path, check_rec_path, 'r')

    for idx in range(10):
        item = read_record.read_idx(idx)
        header, s = mx.recordio.unpack(item)

        print('idx=', idx, 'flag=', header.flag, 'label=', header.label, )

        img = mx.image.imdecode(s).asnumpy()
        img = Image.fromarray(img)
        img.save('train' + str(idx) + '.jpg')

    """ Check for test """
    print('=====> Checking testing dataset')
    check_target = 'test'
    check_idx_path = os.path.join(dataset_path, check_target + '.idx')
    check_rec_path = os.path.join(dataset_path, check_target + '.rec')
    read_record = mx.recordio.MXIndexedRecordIO(check_idx_path, check_rec_path, 'r')

    for idx in range(10):
        item = read_record.read_idx(idx)
        header, s = mx.recordio.unpack(item)

        print('idx=', idx, 'flag=', header.flag, 'label=', header.label, )

        img = mx.image.imdecode(s).asnumpy()
        img = Image.fromarray(img)
        img.save('test' + str(idx) + '.jpg')



