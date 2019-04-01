#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1

config_file="configs/verbal_hard.json"

extra=""

if [ "$exp_name" == "rule_a_e" ]
then
   device=${3:-0}
   output_dir="rule_ablation_${exp_name}_${2}"
   extra="-rule_a_e 1 --deviate_threshold $2"
elif [ "$exp_name" == "rule_b_d" ]
then
  device=${2:-0}
  extra="-rule_b_d 1"
  output_dir="rule_ablation_$exp_name"
else
  echo "Usage: bash train_rule_ablation.sh [rule_a_e|rule_b_d] [deviate_threshold] [gpu_id]"
  echo "Example: bash train_main_results.sh rule_a_e 3 0"
  exit
fi

command="python -u train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






