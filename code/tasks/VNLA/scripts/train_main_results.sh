#!/bin/bash

source define_vars.sh

cd ../

exp_name=$1

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
  extra="-ask_first 1"
elif [ "$exp_name" == "learned" ]
then
  extra=""
else
  echo "Usage: bash train_main_results.sh [none|first|random|teacher|learned]"
  echo "Example: bash train_main_results.sh learned"
  exit
fi

command="python train.py -config $config_file -exp $output_dir $extra"
echo $command
$command






