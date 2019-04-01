#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
device=${2:-0}

config_file="configs/verbal_hard.json"
output_dir="subgoal_effects_$exp_name"

command="python -u train.py -config $config_file -exp $output_dir"

if [ "$exp_name" == "no_subgoal" ]
then
    command="$command -advisor direct -n_iters 70000"
else
  echo "Usage: bash train_subgoal_effects.sh [no_subgoal] [gpu_id]"
  echo "Example: bash train_subgoal_effects.sh no_subgoal 0"
  exit
fi

command="$command -device $device"
echo $command
$command






