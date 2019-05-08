# zoom-learn-zoom
Code for CVPR 2019 paper: Zoom to Learn, Learn to Zoom

<a href="https://ceciliavision.github.io/project-pages/project-zoom.html" target="_blank">Project Website</a>  |  <a href="" target="_blank">Paper</a>

This paper shows that when applying machine learning to digital zoom for photography, it is beneficial to use real, RAW sensor data for training. This code is based on tensorflow. It has been tested on Ubuntu 16.04 LTS.

## ![](./teaser/teaser.png)


## SR-RAW Dataset

#### Use SR-RAW

SR-RAW training and testing now available <a href="https://drive.google.com/open?id=1UHKEUp77tiCZ9y05JtP6S9Tfo2RftK8m" target="_blank">here</a>.

#### Try with your own data

Our model is trained on raw data in <a href="http://arwviewer.com/" target="_blank">Sony Digital Camera Raw</a>. If you use other types of raw data formats, like [DNG](https://helpx.adobe.com/photoshop/digital-negative.html) used by iPhone (you can use the app [Halide](https://itunes.apple.com/us/app/halide-camera/id885697368?mt=8) to store raw from iPhone), it is necessary to fine tune the model with raw data in that format.


## Quick inference

We put example raw data inside `test_raw` folder for quick inference.

- Clone/Download this repo
- `$ cd zoom-learn-zoom`
- Download our <a href="https://drive.google.com/open?id=1uZdZuLvh_jDo5aO60tLVW4RD7Gf9TP3O" target="_blank"> pre-trained zoom model </a>
- To test on an inference folder, set `inference_root` in  `config/inference.yaml` to the `test_raw` path (or whatever your test folder path is), and set `mode` to `inference`
- To test on a single image, set `mode` to `inference_single` and `inference_path` to the raw data path
- Modify the `restore_ckpt` (the folder where your checkpoints locate), `save_root` and `task_folder` in `config/inference.yaml` to your preferred paths
- Run `$ python3 inference.py`
- Results are in `[save_root]/[task_folder]/[mode]`

## Training from scratch

<em>Coming soon</em>

<!-- - `$ mkdir VGG_Model`
- Download [VGG-19](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search `imagenet-vgg-verydeep-19` in this page and download `imagenet-vgg-verydeep-19.mat`. We need the pre-trained VGG-19 model for our hypercolumn input and feature loss
- move the downloaded vgg model to folder `VGG_Model` -->


## Citation

If you find this work useful for your research, please cite:

```
@inproceedings{zhang2019zoom
  title={Zoom to Learn, Learn to Zoom},
  author={Zhang, Xuaner and Chen, Qifeng and Ng, Ren and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Contact
Please contact me if there is any question (Cecilia Zhang <cecilia77@berkeley.edu>).
