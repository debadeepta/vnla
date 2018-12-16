# Vision-based Indoor Navigation via Language-based Interactions

This repo contains implementation of the paper "Vision-based Indoor Navigation via Language-based Interactions"

### Basic setup

System requirements:
* Python 2.7.15
* PyTorch 0.4.1 

We recommend you install [Anaconda with Python 2.7](https://www.anaconda.com/download/#linux). For the rest of this tutorial, we assume that **you have installed Anaconda with Python 2.7**. 

### Run Matterport3D Simulator

The Matterport3D simulator is the navigation environment of this work. Training and testing models do not requires running the simulator in graphics mode. However, running in graphics mode is still useful for debugging and visualizing the agent behavior. 

Please go to [Peter Anderson Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to run the Matterport3D simulator driver (the `demo` section in the README file). Being able to run the simulator in that repo means you have installed all packages required for running the simulator in this repo. 

If you encounter an error saying "OpenCV does not support OpenGL", you may have to install OpenCV from source with OpenGL support. Follow [this guide](https://www.learnopencv.com/install-opencv3-on-ubuntu/), then link `/build/lib/cv2.so` to you your Anaconda site-packages directory:

```
$ rm $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
$ ln -s $OPENCV_REPO/build/lib/cv2.so $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
```

where `OPENCV_REPO` is where you cloned opencv and `ANACONDA_HOME` is where you installed Anaconda. 

### Clone and build

Clone 

```
$ git clone git@github.com:debadeepta/learningtoask.git
```

Build  

```
$ cd learningtoask/code
$ mkdir build & cd build
$ cmake -DPYTHON_EXECUTABLE:$(which python) ..
$ make 
$ cd ..
```

(if `make` does not work, try `sudo make`)

Link the Matterport3D dataset

```
$ ln -s $MATTERPORT3D_DATA data
```
where `$MATTERPORT3D_DATA/v1/scans` contains data of all scans. 

Test if the Simulator has been correctly built

```
$ python
>>> import sys
>>> sys.path.append('build')
>>> import MatterSim
```

Run the Simulator driver 

```
$ python src/driver/driver.py
```

You can run the driver with more specfic arguments
```
$ python python src/driver/driver.py [house_id] [view_id] [heading]
```

For example:
```
$ python src/driver/driver.py 17DRP5sb8fy 30c97842da204e6290ac32904c924e17 0.349
```

### Data

Download the data [here](https://drive.google.com/file/d/1wGQYqqOXSLXBDY_CjsgYbY7jaaHA74gK/view?usp=sharing), uncompress, and replace the `data` directory in the repo's top-level directory. 

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

