#!/bin/bash

git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cp -r Matterport3DSimulator/connectivity . 
yes | rm -rf Matterport3DSimulator

wget -O vnla_data.zip https://www.dropbox.com/s/06ilmo55w4a5mh8/vnla_data.zip?dl=1
unzip vnla_data.zip

mkdir img_features
cd img_features
wget https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1
unzip ResNet-152-imagenet.zip


