#!/bin/bash

sudo apt-get clean
sudo apt-get update
sudo apt-get install -y libopencv-dev python-opencv freeglut3 freeglut3-dev libglm-dev libjsoncpp-dev doxygen libosmesa6-dev libosmesa6 libglew-dev

cd ../../

if [ ! -f build/MatterSim.so ] || [ ! -f build/libMatterSim.so ]; then 
  rm -rf build
  mkdir build
  cd build
  cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) ..
  make
  cd ..
fi
