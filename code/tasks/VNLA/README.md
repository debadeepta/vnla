
Scripts for reproducing results in the paper are in `scripts`. All scripts should be executed inside this directory. 

```
$ cd scripts
```

You can run a script without arguments to display its usage. For exapmle,

```
$ bash train_main_results.sh
```

### Main results

Train a model 
```
$ bash train_main_results.sh [none|first|random|learned|teacher]
```

Evaluate a model
```
$ bash eval_main_results.sh [none|first|random|learned|teacher] [seen|unseen]
```

### Train models

Training configurations for models in the paper are specified in `task/R2R/configs/v3`. 

Go to `tasks/R2R`
```
$ cd tasks/R2R
```

Train a `LearnToAsk` agent with the `NonVerbal` oracle
```
$ python train.py -config configs/v3/intent_next_optimal_cov_v3.json -exp v3_learn_to_ask 

```

Train a `RandomAsk` agent with the `NonVerbal` oracle
```
$ python train.py -config configs/v3/intent_next_optimal_cov_v3.json -exp v3_random_ask -random_ask 1

```

Similarly, you can train other agents with `-no_ask 1`, `-ask_first 1`, `-oracle_ask 1`.


### Evaluate models

Evaluate a pretrained `RandomAsk` agent with the `NonVerbal` oracle
```
$ python train.py -config configs/v3/intent_next_optimal_cov_v3.json \
> -exp v3_oracle_ask \
> -load_path output/v3_oracle_ask_nav_sample_ask_teacher/snapshots/v3_oracle_ask_nav_sample_ask_teacher_val_seen.ckpt \
> -multi_seed 1 \
> -error 2
```

The agent will be evaluated with multiple random seeds (because there is randomness in computing asking budget).

The `-error` flag controls the radius of the region surrounding the goal viewpoint, where the agent will succeed at its task if it steps inside. 

