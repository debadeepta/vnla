
This directory is extended from [Peter Anderson's Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator)

### System requirements:
* Python 2.7.15
* PyTorch 0.4.1 

We recommend installing via [Anaconda with Python 2.7](https://www.anaconda.com/download/#linux). 
The following instructions assume you **use our recommended method to install Python and PyTorch**. 

### Install Matterport3D simulator

Go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to build the simulator. Training and testing models do not require running the simulator in graphics mode. However, running in graphics mode is still useful for debugging and visualizing the agent behavior. 

To test whether you have sucessfully built the simulator, inside the home directory of the Matterport3DSimulator repo, run
```
$ python
>> import sys
>> sys.path.append('build')
>> import MatterSim
```

**Common errors**
* `ImportError: No module named MatterSim`: this will surely happen if you use Anaconda to install Python. 

While compiling the simulator, instead of 
```
$ cmake ..
```
run

```
$ cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) .. 
```
to tell `cmake` where Python is installed. 

* `OpenCV does not support OpenGL`: you may have to install OpenCV from source with OpenGL support. Follow [this guide](https://www.learnopencv.com/install-opencv3-on-ubuntu/), then link `/build/lib/cv2.so` to you your Anaconda site-packages directory:
```
$ rm $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
$ ln -s $OPENCV_REPO/build/lib/cv2.so $ANACONDA_HOME/lib/python2.7/site-packages/cv2.so
```
where `OPENCV_REPO` is where opencv is cloned and `ANACONDA_HOME` is Anaconda's home directory. 

### Build simulator

Install dependencies
```
$ sudo apt install -y libopencv-dev python-opencv freeglut3 freeglut3-dev libglm-dev libjsoncpp-dev     doxygen libosmesa6-dev libosmesa6 libglew-dev     
```

Then build the simulator
```
$ bash setup.sh
```
Test whether the build was sucessful
```
$ python
>> import sys
>> sys.path.append('build')
>> import MatterSim
```

### Explore the environment

Link to the Matterport3D Dataset, which should be structured as `<Matterdata>/v1/scans/<scanId>/matterport_skybox_images/*.jpg`:
```
$ ln -s <Matterdata> data
```
Then run the driver:
```
$ python src/driver/driver.py [envID] [viewpointID] [init_heading]
```

For example:

```
$ python src/driver/driver.py fzynW3qQPVF 8e8d691920d14c8e8a3a2371edeaa2bd 2.6179938779914944
```




