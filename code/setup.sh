#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

rm -rf build
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) ..
make
cd ..
