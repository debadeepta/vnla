#!/bin/bash

source define_vars.sh

cd ../

exp_name=$1
device=${2:-0}

config_file="configs/verbal_hard.json"
output_dir="subgoal_effects_$exp_name"

command="python train.py -config $config_file -exp $output_dir"

if [ "$exp_name" == "no_subgoal" ]
then
    command="$command -advisor direct"
elif [ "$exp_name" == "subgoal" ]
then
    command="$command -n_iters 175000"
else
  echo "Usage: bash train_subgoal_effects.sh [no_subgoal|subgoal] [gpu_id]"
  echo "Example: bash train_subgoal_effects.sh no_subgoal 0"
  exit
fi

command="$command -device $device"
echo $command
$command






