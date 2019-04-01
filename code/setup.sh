#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

if [ ! -f build/MatterSim.so ] || [ ! -f build/libMatterSim.so ]; then 
  rm -rf build
  mkdir build
  cd build
  cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) ..
  make
  cd ..
fi
