#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
device=${2:-0}

config_file="configs/verbal_hard_noroom.json"
output_dir="${exp_name}"

command="python -u train.py -config $config_file -exp $output_dir"

if [ "$exp_name" == "noroom_random" ]
then
    command="$command -random_ask 1"
elif [ "$exp_name" == "noroom_learned" ]
then
    command="$command"
else
  echo "Usage: bash train_noroom.sh [noroom_random|noroom_learned] [gpu_id]"
  echo "Example: bash train_noroom.sh noroom_random 0"
  exit
fi

command="$command -device $device"
echo $command
$command






