
This directory is extended from [Peter Anderson's Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator). Except nothing else is specified, any of the below code snippets should be executed in this directory. 

### 0. System requirements:
* Python 2.7.15
* PyTorch 0.4.1 

We recommend installing via [Anaconda with Python 2.7](https://www.anaconda.com/download/#linux). 
The following instructions assume you **use our recommended method to install Python and PyTorch**. 

### 1. Download Matterport3D dataset

You have to request access to the dataset [here](https://niessner.github.io/Matterport/). 
Training and testing our models do not require running the simulator in graphics mode. So you only need to download the `house_segmentations` of the dataset. Unzip the files so that `<Matterdata>/v1/scans/<scanId>/house_segmentations/panorama_to_region.txt` are present. 

Running in graphics mode is still useful for debugging and visualizing the agent behavior. You need to download the `matterport_skybox_images` portion and unzip the files so that `<Matterdata>/v1/scans/<scanId>/matterport_skybox_images/*.jpg` are present. 

### 1. Install Matterport3D simulator

Go to [Matterport3DSimulator repo](https://github.com/peteanderson80/Matterport3DSimulator), follow the instructions to build the simulator. To test whether you have sucessfully built the simulator, inside the home directory of the Matterport3DSimulator repo, run
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

*Make sure you are able to run the demo in the Matterport3D repo.*

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




