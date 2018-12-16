#!/bin/sh

wget -O vnla_data.zip https://www.dropbox.com/s/06ilmo55w4a5mh8/vnla_data.zip?dl=1
unzip vnla_data.zip

mkdir img_features
cd img_features
wget https://storage.googleapis.com/bringmeaspoon/img_features/ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip


