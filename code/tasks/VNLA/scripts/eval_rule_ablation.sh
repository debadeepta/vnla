#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1

config_file="configs/verbal_hard.json"

if [ "$exp_name" == "rule_a_e" ]
then
   extra="-rule_a_e 1 --deviate_threshold ${2}"
   output_dir="rule_ablation_rule_a_e_${2}"
   split=$3
   device=${4:-0}
elif [ "$exp_name" == "rule_b_d" ]
then
  extra="-rule_b_d 1"
  output_dir="rule_ablation_$exp_name"
  split=$2
  device=${3:-0}
else
  echo "Usage: bash eval_rule_ablation.sh [rule_a_e|rule_b_d] [deviate_threshold] [seen|unseen] [gpu_id]"
  echo "Example: bash eval_rule_ablation.sh rule_a_e 2 seen 0"
  exit
fi

model_name="${output_dir}_nav_sample_ask_teacher"
extra="$extra -load_path $PT_OUTPUT_DIR/$model_name/${model_name}_val_${split}.ckpt -multi_seed 1 -success_radius 2"


command="python -u train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






