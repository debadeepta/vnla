#!/bin/bash

source define_vars.sh

cd ../

exp_name=$1
split=$2
device=${3:-0}

config_file="configs/verbal_hard.json"
output_dir="subgoal_effects"

extra=""

if [ "$exp_name" == "direct_no_subgoal" ]
then
    extra="-advisor direct"
    output_dir="${output_dir}_no_subgoal"
elif [ "$exp_name" == "direct_subgoal" ]
then
    extra="-teacher_interpret 1"
    output_dir="${output_dir}_subgoal"
elif [ "$exp_name" == "indirect_subgoal" ]
then
    extra=""
    output_dir="${output_dir}_subgoal"
else
  echo "Usage: bash eval_subgoal_effects.sh [direct_no_subgoal|direct_subgoal|indirect_subgoal] [seen|unseen] [gpu_id]"
  echo "Example: bash eval_subgoal_effects.sh direct_no_subgoal unseen 0"
  exit
fi

model_name="${output_dir}_nav_sample_ask_teacher"
command="python train.py -config $config_file -exp $output_dir $extra"
command="$command -load_path $PT_OUTPUT_DIR/$model_name/snapshots/${model_name}_val_${split}.ckpt -multi_seed 1 -device $device"
echo $command
$command






