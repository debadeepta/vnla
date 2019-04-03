
Next: [Run experiments](https://github.com/debadeepta/learningtoask/tree/master/code/tasks/VNLA)

This directory is extended from [Peter Anderson's Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1). Unless specified, all of the code snippets below should be executed in this directory. 

NOTE: the Matterport3D simulator has recently been updated. Our code has not been tested on the new release. Please use v0.1 of the simulator. 

### 1. Install Matterport3D simulator

Go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1), follow the instructions to build the simulator. To test whether you have sucessfully built the simulator, inside the home directory of the Matterport3DSimulator repo, run
```
$ python
>> import sys
>> sys.path.append('build')
>> import MatterSim
```

**Common errors**
* `ImportError: No module named MatterSim`: this will definitely happen if you use Anaconda to install Python. 

While compiling the simulator, instead of 
```
$ cmake ..
```
run

```
$ cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) .. 
```
to tell `cmake` where Python is installed by Anaconda. 

* `OpenCV does not support OpenGL`: you may have to install OpenCV from source with OpenGL support. Follow [this guide](https://www.learnopencv.com/install-opencv3-on-ubuntu/), then link `cv2.so` to you your Anaconda site-packages directory:
```
$ rm $ANACONDA_HOME/lib/$PYTHON_FOLDER/site-packages/cv2.so
$ ln -s $OPENCV_REPO/build/lib/cv2.so $ANACONDA_HOME/lib/$PYTHON_FOLDER/site-packages/cv2.so
```
where `OPENCV_REPO` is where opencv is cloned, `ANACONDA_HOME` is Anaconda's home directory, and `PYTHON_FOLDER` is where Python is installed by Anaconda (e.g., for Python 3.6, it is `python3.6`). 

### 2. Build simulator

Install dependencies
```
$ sudo apt install -y libopencv-dev python-opencv freeglut3 freeglut3-dev libglm-dev libjsoncpp-dev doxygen libosmesa6-dev libosmesa6 libglew-dev     
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

### 3. Explore environments (optional)

*Before following the instructions below, make sure you are able to run the demo in the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) repo.*

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
$ python src/driver/driver.py fzynW3qQPVF 8e8d691920d14c8e8a3a2371edeaa2bd 1.04
```

Use keys A, S, D, W to adjust the camera angle, and use keys 1-9 to go an adjacent viewpoint. 

Next: [Run experiments](https://github.com/debadeepta/learningtoask/tree/master/code/tasks/VNLA)
