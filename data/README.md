Next: [Setup simulator](https://github.com/debadeepta/learningtoask/tree/master/code)

### 1. Download VNLA dataset and image features

To download data, run

```
$ bash download.sh
```

**Important**: The ImageNet features are not maintained by us, so if the link would not work in the future, please visit: https://github.com/peteanderson80/Matterport3DSimulator#precomputing-resnet-image-features and click on "ResNet-152-imagenet features [380K/2.9GB]" to get the latest link, or email an author of [this paper](https://arxiv.org/abs/1711.07280).

The following directories and files will be created:
* `asknav` and `noroom`: VNLA datasets. Please read our paper for more detail about each dataset.
* `connectivity`: environment graphs.
* `img_features`: precomputed image embeddings computed by ResNet pretrained on ImageNet. 
* `region_label.txt`: room names. 
* `accepted_objects.txt`: object labels.

### 2. Download Matterport3D dataset

Request access to the dataset [here](https://niessner.github.io/Matterport/). The dataset is for **non-commercial academic purposes** only. Please read and agree to the dataset's [terms and conditions](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf) and **put this link in your project repo as requested by the dataset's creators**.

Training and testing our models only require downloading the `house_segmentations` portion of the dataset. Unzip the files so that `<some_folder>/v1/scans/<scanId>/house_segmentations/panorama_to_region.txt` are present. 

Running in graphics mode is still useful for debugging and visualizing the agent behavior. You need to download the `matterport_skybox_images` portion and unzip the files so that `<some_folder>/v1/scans/<scanId>/matterport_skybox_images/*.jpg` are present. 

We provide the script `unzip_matterport.sh` in this directory to help you unzip the data:
```
$ bash unzip_matterport.sh $matter_root
```

where `$matter_root` is the Matterport3D dataset top folder where `$matter_root/v1/scans/` is located.  



Next: [Setup simulator](https://github.com/debadeepta/learningtoask/tree/master/code)
