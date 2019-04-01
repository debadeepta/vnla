#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cp -r Matterport3DSimulator/connectivity . 
yes | rm -rf Matterport3DSimulator

wget -O vnla_data.zip https://www.dropbox.com/s/w0n6hzjky5jcea4/vnla_data.zip?dl=1
unzip vnla_data.zip
mv vnla_data/* .
rm -rf vnla_data

mkdir img_features
cd img_features
wget -O ResNet-152-imagenet.zip https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1
unzip ResNet-152-imagenet.zip


