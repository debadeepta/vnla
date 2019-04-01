#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
device=${2:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

extra=""

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
  echo "Usage: bash train_main_results.sh [none|first|random|teacher|learned] [gpu_id]"
  echo "Example: bash train_main_results.sh learned 0"
  exit
fi

command="python -u train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






