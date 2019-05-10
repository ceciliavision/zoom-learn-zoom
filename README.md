# zoom-learn-zoom
Code for CVPR 2019 paper: Zoom to Learn, Learn to Zoom

<a href="https://ceciliavision.github.io/project-pages/project-zoom.html" target="_blank">Project Website</a>  |  <a href="" target="_blank">Paper</a>

This paper shows that when applying machine learning to digital zoom for photography, it is beneficial to use real, RAW sensor data for training. This code is based on tensorflow (tested on V 1.13.1). It has been tested on Ubuntu 16.04 LTS.

## ![](./teaser/teaser.png)


## SR-RAW Dataset

#### Use SR-RAW

SR-RAW training and testing now available <a href="https://drive.google.com/open?id=1UHKEUp77tiCZ9y05JtP6S9Tfo2RftK8m" target="_blank">here</a>.

To download testing dataset (7 GB), run:
```
bash ./scripts/download.sh 19zlN1fqRRm7E_6i5J3B1OskJocVeuvzG test.zip
unzip test.zip
rm test.zip
```
We used 35 mm images (mostly named '00006' in the sequences) for test.

To download training dataset (58 GB), run:
```
bash ./scripts/download.sh 1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM train.zip
unzip train.zip
rm train.zip
```

#### Try with your own data

Our model is trained on raw data in <a href="http://arwviewer.com/" target="_blank">Sony Digital Camera Raw</a>. If you use other types of raw data formats, like [DNG](https://helpx.adobe.com/photoshop/digital-negative.html) used by iPhone (you can use the app [Halide](https://itunes.apple.com/us/app/halide-camera/id885697368?mt=8) to store raw from iPhone), it is necessary to fine tune the model with raw data in that format.

## Quick inference

In the following, we will download the pre-trained model and example raw data for quick inference.

```
git clone https://github.com/ceciliavision/zoom-learn-zoom.git
cd zoom-learn-zoom
bash ./scripts/download.sh 1iForbFhhWqrq22FA1xIusfUpdi8td4Kq model.zip
unzip model.zip
bash ./scripts/download.sh 1WVSGaKIJVHwphTKhcr9ajolEnBh3aUkR quick_inference.zip
unzip quick_inference.zip
rm *.zip
python3 inference.py
```

Notes about `config/inference.yaml`
- To do inference on a folder, set `mode` to `inference` and set `inference_root` (e.g. `./quick_inference/`)
- To do inference on a single image, set `mode` to `inference_single` and set `inference_path`  (e.g. `./quick_inference/00134.ARW`)
- Set `task_folder` (e.g. `./restore_4x`)
- Results are saved in `./[task_folder]/[mode]`


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
