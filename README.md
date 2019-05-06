# zoom-learn-zoom
Code and data for CVPR 2019 paper [Zoom to Learn, Learn to Zoom]()
 and [Supplementary material]()

This paper shows that when applying machine learning to digital zoom for photography, it is beneficial to use real, RAW sensor data for training. This code is based on tensorflow. It has been tested on Ubuntu 16.04 LTS.

## ![](./teaser/teaser.png)



## Setup

- Clone/Download this repo
- `$ cd zoom-learn-zoom`
- `$ mkdir VGG_Model`
- Download [VGG-19](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search `imagenet-vgg-verydeep-19` in this page and download `imagenet-vgg-verydeep-19.mat`. We need the pre-trained VGG-19 model for our hypercolumn input and feature loss
- move the downloaded vgg model to folder `VGG_Model`



## SR-Raw Dataset

#### Use SR-Raw

SR-Raw is now available [here]().

#### Try with your own dataset



## Quick test

We put two test raw data inside `test_raw` folder for a quick test.

- Set `inference_root` in  `config/inference.yaml` to the `test_raw` path
- Modify the `save_root`, `restore_ckpt` and `task_folder` in `config/inference.yaml`  to your preferred paths  
- `$ python3 inference.py`



## Citation

If you find this work useful for your research, please cite:

```
@inproceedings{
}
```

## Contact
Please contact me if there is any question (Cecilia Zhang <cecilia77@berkeley.edu>)
