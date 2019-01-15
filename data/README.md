### 1. Download VNLA dataset and image features

To download data, run

```
$ bash download.sh
```

The following directories will be created:
* `asknav` and `noroom`: VNLA datasets. Please read our paper for more detail about each dataset.
* `img_features`: precomputed image embeddings computed by ResNet pretrained on ImageNet. 

Next: [Setup simulator](https://github.com/debadeepta/learningtoask/tree/master/code)

### 2. Download Matterport3D dataset

Request access to the dataset [here](https://niessner.github.io/Matterport/). 
Training and testing our models only require downloading the `house_segmentations` portion of the dataset. Unzip the files so that `<some_folder>/v1/scans/<scanId>/house_segmentations/panorama_to_region.txt` are present. 

Running in graphics mode is still useful for debugging and visualizing the agent behavior. You need to download the `matterport_skybox_images` portion and unzip the files so that `<some_folder>/v1/scans/<scanId>/matterport_skybox_images/*.jpg` are present. 
