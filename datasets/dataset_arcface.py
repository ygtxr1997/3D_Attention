"""
A copy from ArcFace: https://
"""

import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # (112, 112, 3)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class MXCifarTrainDataset(Dataset):
    def __init__(self, root_dir, local_rank,
                 re_p=0.):
        super(MXCifarTrainDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.RandomErasing(p=re_p),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),  # to [-1, 1]
             transforms.RandomErasing()  # Train use random erasing
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))  # cnt_sample, cnt_class
            self.imgidx = np.array(range(1, int(header.label[0])))
            if local_rank == 0:
                print('**** [cifar-train] flag>0, cnt_sample=%d, cnt_class=%d ****' % (
                    int(header.label[0]) - 1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            raise ValueError

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # (32, 32, 3)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class MXCifarTestDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXCifarTestDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),  # to [-1, 1]
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'test.rec')
        path_imgidx = os.path.join(root_dir, 'test.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))  # cnt_sample, cnt_class
            self.imgidx = np.array(range(1, int(header.label[0])))
            if local_rank == 0:
                print('**** [cifar-test] flag>0, cnt_sample=%d, cnt_class=%d ****' % (
                    int(header.label[0]) - 1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            raise ValueError

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # (32, 32, 3)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class MXImageNet1kTrainDataset(Dataset):
    """
    Refer to https://github.com/kuan-wang/pytorch-mobilenet-v3
    """
    def __init__(self, root_dir, local_rank,
                 re_p=0.):
        super(MXImageNet1kTrainDataset, self).__init__()
        self.input_size = 224
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomResizedCrop(self.input_size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomErasing(p=re_p),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),  # to [-1, 1]
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))  # cnt_sample, cnt_class
            self.imgidx = np.array(range(1, int(header.label[0])))
            if local_rank == 0:
                print('**** [imagenet-1k-train] flag>0, cnt_sample=%d, cnt_class=%d ****' % (
                    int(header.label[0]) - 1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            raise ValueError

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # (112, 112, 3)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class MXImageNet1kTestDataset(Dataset):
    """
    Refer to https://github.com/kuan-wang/pytorch-mobilenet-v3
    """
    def __init__(self, root_dir, local_rank,):
        super(MXImageNet1kTestDataset, self).__init__()
        self.input_size = 224
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(int(self.input_size / 0.875)),
             transforms.CenterCrop(self.input_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),  # to [-1, 1]
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'test.rec')
        path_imgidx = os.path.join(root_dir, 'test.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))  # cnt_sample, cnt_class
            self.imgidx = np.array(range(1, int(header.label[0])))
            if local_rank == 0:
                print('**** [imagenet-1k-test] flag>0, cnt_sample=%d, cnt_class=%d ****' % (
                    int(header.label[0]) - 1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            raise ValueError

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()  # (112, 112, 3)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)