
This directory is extended from [Peter Anderson's Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator)

### System requirements:
* Python 2.7.15
* PyTorch 0.4.1 

We recommend installing via [Anaconda with Python 2.7](https://www.anaconda.com/download/#linux). 
The following instructions assume you **use our recommended method to install Python and PyTorch**. 

#### Install Matterport3D simulator

The Matterport3D simulator is the navigation environment of this work. 
Go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to build the simulator. Training and testing models do not require running the simulator in graphics mode. However, running in graphics mode is still useful for debugging and visualizing the agent behavior. 

To test whether you have sucessfully build the simulator, inside the home directory of the Matterport3DSimulator repo, run
```
$ python
>> import sys
>> sys.path.append('build')
>> import MatterSim
```

**Common errors**
1. `ImportError: No module named MatterSim`: this will surely happen if you use Anaconda to install Python. While compiling the simulator, instead of 
```
$ cmake ..
```
run

```
$ cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) .. 
```
to tell `cmake` where Python is installed. 

2. `OpenCV does not support OpenGL`: you may have to install OpenCV from source with OpenGL support. Follow [this guide](https://www.learnopencv.com/install-opencv3-on-ubuntu/), then link `/build/lib/cv2.so` to you your Anaconda site-packages directory:
```
$ rm $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
$ ln -s $OPENCV_REPO/build/lib/cv2.so $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
```
where `OPENCV_REPO` is where opencv is cloned and `ANACONDA_HOME` is Anaconda's home directory. 

### Build simulator


