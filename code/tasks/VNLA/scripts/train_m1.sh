#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# To run reinforcement learning for M1, run this after the main_results

source define_vars.sh

cd ../

exp_name="m1-learned"
device=${1:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

# Smaller-scale training
command="time python -u m1_train.py -config $config_file -exp $output_dir -device $device -log_every 100 -save_every 100 -n_iters 15000 -batch_size 50"
echo $command
$command
