#!/bin/bash

source define_vars.sh

cd ../

exp_name=$1

config_file="configs/v3/intent_verbal_hard_cov_v3.json"
exp_name="main_$exp_name"

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
fi

python train.py -config $config_file -exp $exp_name $extra






