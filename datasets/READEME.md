# Prepare ImageNet-1K

## 1. Download from [http://www.image-net.org/challenges/LSVRC/2012/downloads](http://www.image-net.org/challenges/LSVRC/2012/downloads)

## 2. For training dataset
```shell script
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

## 3. For validating dataset
```shell script
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
or
```shell script
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://files-cdn.cnblogs.com/files/luruiyuan/valprep.sh | bash
```