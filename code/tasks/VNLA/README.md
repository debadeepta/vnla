
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

### Main results

Train a model 
```
$ bash train_main_results.sh [none|first|random|learned|teacher]
```

Evaluate a model
```
$ bash eval_main_results.sh [none|first|random|learned|teacher] [seen|unseen]
```


