
Scripts for reproducing results in the paper are in `scripts`. All scripts should be executed inside the directory. 

```
$ cd scripts
```

You can run a script without arguments to display its usage. For example,

```
$ bash train_main_results.sh
Usage: bash train_main_results.sh [none|first|random|teacher|learned] [gpu_id]
Example: bash train_main_results.sh learned 0
```

**NOTE**: you may not get exactly the numbers reported in the paper because different types of GPU models/CUDA/cuDNN may implement randomization differently. However, the difference should not be significant. 

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

Although the table has three rows, you only need to train two agents: one trained with subgoals (indirect advisor) and another trained without subgoals (direct advisor). 

Train an agent with subgoals:
```
$ bash train_subgoal_effects.sh subgoal
```

Evaluate the agent with a direct advisor on `test unseen` (second row, third column of the table):
```
$ bash eval_subgoal_effects.sh direct_subgoal unseen
```


