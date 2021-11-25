# 3D_Attention
Inhibition-aware (-regularized) 3D attention for robust visual recognition

---
## 0. Environment Preparation

* The version requirements of key frameworks are listed here:
```shell script
python>=3.6
pytorch>=1.6.0
mxnet>=1.6.0
```
* For more detailed requirements, please refer to 'requirements.txt'

---
## 1. Dataset Preparation

### 1.1 CIFAR-100

* Download [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html) dataset.
* Convert to `mxnet.recordio.MXIndexedRecordIO`:
```shell script
cd datasets
python cvt_cifar_100.py
```
* {*Optional*} For faster IO, you can copy your datasets ('train.rec' and 'train.idx') to memory using this:
```shell script
sudo mkdir /tmp/train_tmp
mount -t tmpfs -o size=10G tmpfs /tmp/train_tmp
cp {Your_Datasets} /tmp/train_tmp
```

### 1.2 ImageNet-1k

* TODO

---
## 2. Train

### 2.1 Configure

* Edit `config.py` as you need.
* For `fp16`, please refer to [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html).
* For `rec`, you need to set it as your datasets folder which includes 'xxx.rec' and 'xxx.idx'.
* Some settings are useless. You may ignore them.

### 2.2 Train for CIFAR-100

* This repository adopts DDP training scheme. Please refer to [https://pytorch.org/docs/stable/distributed.html](https://pytorch.org/docs/stable/distributed.html). 
* You can start train on 4 GPUs like this:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --network iresnet18
```
* For more `--network` types, please refer to `backbones/__init__.py`.
* The evaluating process starts every 10 epoch during training. You don't need to run other codes for evaluation.

### 2.3 Train for ImageNet-1k

* TODO

---
## 3. Evaluate

* TODO

