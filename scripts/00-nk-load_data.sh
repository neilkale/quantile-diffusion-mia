#!/bin/bash

# Create the data/raw folder if it doesn't exist
mkdir -p data/raw/

# Download the CelebA-HQ dataset
if [ ! -d data/raw/celebahq ]; then
    mkdir -p data/raw/celebahq
    gdown 1badu11NqxGf6qM3PTTooQDJvQbejgbTv -O data/raw/celebahq.zip
    unzip -o data/raw/celebahq.zip -d data/raw/celebahq
    rm -f data/raw/celebahq.zip
fi

# Download the ImageNet dataset
if [ ! -d data/raw/imagenet ]; then
    mkdir -p data/raw/imagenet

    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -O data/raw/imagenet/ILSVRC2012_img_train.tar
    mkdir -p data/raw/imagenet/train
    tar -xf data/raw/imagenet/ILSVRC2012_img_train.tar -C data/raw/imagenet/train
    rm -f data/raw/imagenet/ILSVRC2012_img_train.tar

    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar -O data/raw/imagenet/ILSVRC2012_img_test_v10102019.tar
    mkdir -p data/raw/imagenet/test
    tar -xf data/raw/imagenet/ILSVRC2012_img_test_v10102019.tar -C data/raw/imagenet/test
    rm -f data/raw/imagenet/ILSVRC2012_img_test_v10102019.tar

    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -O data/raw/imagenet/ILSVRC2012_img_val.tar
    mkdir -p data/raw/imagenet/val
    tar -xf data/raw/imagenet/ILSVRC2012_img_val.tar -C data/raw/imagenet/val
    rm -f data/raw/imagenet/ILSVRC2012_img_val.tar

    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -O data/raw/imagenet/ILSVRC2012_devkit_t12.tar.gz
    mkdir -p data/raw/imagenet/devkit
    tar -xf data/raw/imagenet/ILSVRC2012_devkit_t12.tar.gz -C data/raw/imagenet/devkit
    rm -f data/raw/imagenet/ILSVRC2012_devkit_t12.tar.gz
fi

# Download the HDA-SynChildFaces dataset
if [ ! -d data/raw/hdasynchildfaces ]; then
    gdown --folder https://drive.google.com/drive/folders/1uoeWxrcNW9A9m9TqYStPW9tJZ1Fld2L3 -O data/raw/hdasynchildfaces
    unzip -o data/raw/hdasynchildfaces/dataset.zip -d data/raw/hdasynchildfaces
    rm -f data/raw/hdasynchildfaces/dataset.zip
fi

# Download the CIFAR-10 dataset
if [ ! -d data/raw/cifar10 ]; then
    mkdir -p data/raw/cifar10
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/raw/cifar10/cifar-10-python.tar.gz
    tar -xf data/raw/cifar10/cifar-10-python.tar.gz -C data/raw/cifar10
    rm -f data/raw/cifar10/cifar-10-python.tar.gz
fi

# Download the CIFAR-100 dataset
if [ ! -d data/raw/cifar100 ]; then
    mkdir -p data/raw/cifar100
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -O data/raw/cifar100/cifar-100-python.tar.gz
    tar -xf data/raw/cifar100/cifar-100-python.tar.gz -C data/raw/cifar100
    rm -f data/raw/cifar100/cifar-100-python.tar.gz
fi