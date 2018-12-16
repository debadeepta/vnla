
This directory is extended from [Peter Anderson's Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator)

### System requirements:
* Python 2.7.15
* PyTorch 0.4.1 

We recommend you install these two via [Anaconda with Python 2.7](https://www.anaconda.com/download/#linux). 
The following instructions assume that **you use our recommended method to install Python and PyTorch**. 

The [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator) is the navigation environment of this work. 
First, go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to build the simulator. You can check 

**Graphics mode**: Training and testing models do not require running the simulator in graphics mode. However, running in graphics mode is still useful for debugging and visualizing the agent behavior. 

First, go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to run the Matterport3D simulator driver. Being able to run the simulator in that repo means you have installed all packages required for running the simulator in this repo. 

If you encounter an error saying "OpenCV does not support OpenGL", you may have to install OpenCV from source with OpenGL support. Follow [this guide](https://www.learnopencv.com/install-opencv3-on-ubuntu/), then link `/build/lib/cv2.so` to you your Anaconda site-packages directory:

```
$ rm $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
$ ln -s $OPENCV_REPO/build/lib/cv2.so $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
```

where `OPENCV_REPO` is where opencv is cloned to and `ANACONDA_HOME` is where Anaconda is installed. 

### Build simulator


