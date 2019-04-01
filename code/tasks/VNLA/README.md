
Scripts for reproducing results in the paper are in `scripts`. All scripts should be executed inside their directory. 

```
$ cd scripts
```

You can run a script without arguments to display its usage. For example,

```
$ bash train_main_results.sh
Usage: bash train_main_results.sh [none|first|random|teacher|learned] [gpu_id]
Example: bash train_main_results.sh learned 0
```

**NOTE**: you may get results slightly different from those reported in the paper because different types of GPU models or CUDA/cuDNN versions may have different implementations. However, this should not alter the experiments' conclusions.

### Main results

This section helps you reproduce **Table 2** in our paper. 

For example, train an agent with a `random` help-requesting policy

```
$ bash train_main_results.sh random
```

Evaluate the agent after it is trained on `test seen`
```
$ bash eval_main_results.sh random seen
```

### Subgoal effects

This section helps you reproduce **Table 3** in our paper. 

You need to train one additional agent: an agent trained without subgoals and with direct advisor. 

```
$ bash train_subgoal_effects.sh no_subgoal
```

Evaluate the agent with a direct advisor on `test unseen` (first row, `test unseen` column of the table):
```
$ bash eval_subgoal_effects.sh direct_no_subgoal unseen
```

### No room types

This section helps you reproduce **Table 4** in our paper. 

You need to train two agents on the `noroom` dataset: one with a `random` help-requesting policy and one with a `learned` help-requesting policy. Evaluating these two agents generating results in the first two rows of the table. 

Train the `random` agent
```
$ bash train_noroom.sh noroom_random
```

and evaluate it on `test seen` of `noroom`

```
$ bash eval_noroom.sh noroom_random seen
```

The third row of the table derives from evaluating the `learned` agent trained on the `asknav` dataset. To train this agent, run
```
$ bash train_main_results.sh learned
```

Evaluate this agent on `test unseen` of `noroom`
```
$ bash eval_noroom.sh asknav_learned unseen
```

### Train with new configuration

1. Set environment variables `PT_DATA_DIR` and `PT_OUTPUT_DIR` to the data directory and the output directory, respectively. See `scripts/define_vars.sh` for more detail. 
2. Create a configuration file in `configs`. See `flags.py` for argument definitions.
2. Run `python train.py -config $CONFIG_FILE_PATH -exp $EXP_NAME`.

Besides the `verbal_hard` advisor, which we use in our paper, we also provide a `verbal_easy`, which does not aggregate repeated actions. 


