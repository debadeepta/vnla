#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
split=$2
device=${3:-0}

config_file="configs/verbal_hard_noroom.json"

extra=""

if [ "$exp_name" == "noroom_random" ]
then
   extra="-random_ask 1"
   output_dir="$exp_name"
elif [ "$exp_name" == "noroom_learned" ]
then
  extra=""
  output_dir="$exp_name"
elif [ "$exp_name" == "asknav_learned" ]
then
  extra="-external_main_vocab $PT_DATA_DIR/asknav/train_vocab.txt -data_dir noroom -no_room 1"
  output_dir="main_learned"
else
  echo "Usage: bash eval_noroom.sh [noroom_random|noroom_learned|asknav_learned] [seen|unseen] [gpu_id]"
  echo "Example: bash eval_noroom.sh asknav_learned seen 0"
  exit
fi

model_name="${output_dir}_nav_sample_ask_sample"
extra="$extra -load_path $PT_OUTPUT_DIR/$model_name/${model_name}_val_${split}.ckpt -multi_seed 1 -success_radius 2"

command="python -u train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






