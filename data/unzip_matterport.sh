#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

matter_root=$1

if [ ! -d "$matter_root" ]; then
  echo "Usage: bash unzip_matterport.zip matter_root"
  echo "matter_root is the Matterport3D dataset folder where matter_root/v1/scans is located"
  exit 1
fi

cd $matter_root/v1/scans

miss_hs=0
miss_sb=0
for folder in *; do
  filename="$folder/house_segmentations.zip"
  if [ ! -f $filename ]; then
    echo "$filename does not exist!"
    let miss_hs++
  else
    unzip $filename -d .
  fi

  filename="$folder/matterport_skybox_images.zip"
  if [ ! -f $filename ]; then
    echo "$filename does not exist!"
    let miss_sb++
  else
    unzip $filename -d .
  fi
done

echo ""
echo "Done! Missing $miss_hs house_segmentations zip files and $miss_sb matterport_skybox_images zip files"
