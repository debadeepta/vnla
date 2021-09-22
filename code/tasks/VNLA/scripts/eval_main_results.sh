#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
split=$2
device=${3:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

extra=""
model_name="${output_dir}_nav_sample_ask_sample"

if [ "$exp_name" == "none" ]
then
   extra="-no_ask 1"
elif [ "$exp_name" == "first" ]
then
  extra="-ask_first 1"
elif [ "$exp_name" == "random" ]
then
  extra="-random_ask 1"
elif [ "$exp_name" == "teacher" ]
then
  extra="-teacher_ask 1"
elif [ "$exp_name" == "learned" ]
then
  extra=""
else
  echo "Usage: bash eval_main_results.sh [none|first|random|teacher|learned] [seen|unseen] [gpu_id]"
  echo "Example: bash eval_main_results.sh learned seen 0"
  exit
fi

extra="$extra -load_path $PT_OUTPUT_DIR/$model_name/${model_name}_val_${split}.ckpt -multi_seed 1 -success_radius 2"


command="python -u train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






