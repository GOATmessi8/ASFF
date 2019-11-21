#Learning Spatial Fusion for Single-Shot Object Detection

By Songtao Liu, Di Huang, Yunhong Wang

### Introduction
In this work, we propose a novel and data driven strategy for pyramidal feature fusion, referred to as adaptively spatial feature fusion (ASFF). It learns the way to spatially filter conflictive information to suppress the inconsistency, thus improving the scale-invariance of features, and introduces nearly free inference overhead. For more details, please refer to our [arXiv paper]().

<img align="right" src="https://github.com/ruinmessi/RFBNet/blob/master/doc/RFB.png">

&nbsp;
&nbsp;

### COCO 
| System |  *test-dev mAP* | **Time** (V100) | **Time** (2080ti)|
|:-------|:-----:|:-------:|:------:|
| [YOLOv3 608](http://pjreddie.com/darknet/yolo/) | 33.0 | 20ms| 24ms|
| YOLOv3 608+ [BoFs](https://arxiv.org/abs/1902.04103) | 37.0 | 20ms | 24ms|
| YOLOv3* 608(ours baseline) | **38.8** | 20ms | 24ms|
| YOLOv3 608+ ASFF | **40.6** | 22ms | 28ms| 
| YOLOv3 608+ ASFF* | **42.4** | 22ms | 29ms| 
| YOLOv3 800+ ASFF* | **43.9** | 34ms | 40ms| 


### Citing 
Please cite our paper in your publications if it helps your research:

    @article{liu2017RFB,
        title = {Learning Spatial Fusion for Single-Shot Object Detection},
        author = {Songtao Liu, Di Huang and Yunhong Wang},
        booktitle = {arxiv preprint arXiv:1911.07767},
        year = {2019}
    }

### Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Models](#models)

## Installation
- Install [PyTorch-1.3.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. 
    * Note: We currently only support PyTorch-1.0.0+ and Python 3+.
- Compile the DCN layer:
```Shell
./make.sh
```
#Prerequisites
We also use [apex](https://github.com/NVIDIA/apex), opencv, tqdm 

## Datasets
Note: We currently only support [COCO](http://mscoco.org/) and [VOC](http://host.robots.ox.ac.uk/pascal/VOC/).  
To make things easy, we provide simple COCO and VOC dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

Moreover, we also implement the Mix-up strategy in [BoFs](https://arxiv.org/abs/1902.04103) and distributed random resizing in YOLov3.
### COCO Dataset
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ./data/COCO, and a soft-link is recommended. 
```
ln -s /path/to/coco ./data/COCO
```

It should have this basic structure
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2017/
$COCO/images/train2017/
$COCO/images/val2017/
```
The current COCO dataset has released new *train2017* and *val2017* sets, and we defaultly train our model on *train2017* and evaluate on *val2017*. 

## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:    https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
or from our [BaiduYun Driver](https://pan.baidu.com/s/1jIP86jW) 
- MobileNet pre-trained basenet is ported from [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe), which achieves slightly better accuracy rates than the original one reported in the [paper](https://arxiv.org/abs/1704.04861), weight file is available at: https://drive.google.com/open?id=13aZSApybBDjzfGIdqN1INBlPsddxCK14 or [BaiduYun Driver](https://pan.baidu.com/s/1dFKZhdv).

- By default, we assume you have downloaded the file in the `RFBNet/weights` dir:
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RFBNet using the train script simply specify the parameters listed in `train_RFB.py` as a flag or manually change them.
```Shell
python train_RFB.py -d VOC -v RFB_vgg -s 300 
```
- Note:
  * -d: choose datasets, VOC or COCO.
  * -v: choose backbone version, RFB_VGG, RFB_E_VGG or RFB_mobile.
  * -s: image size, 300 or 512.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train_RFB.py` for options)
  * If you want to reproduce the results in the paper, the VOC model should be trained about 240 epoches while the COCO version need 130 epoches.
  
## Evaluation
To evaluate a trained network:

```Shell
python test_RFB.py -d VOC -v RFB_vgg -s 300 --trained_model /path/to/model/weights
```
By default, it will directly output the mAP results on VOC2007 *test* or COCO *minival2014*. For VOC2012 *test* and COCO *test-dev* results, you can manually change the datasets in the `test_RFB.py` file, then save the detection results and submitted to the server. 

## Models

* 07+12 [RFB_Net300](https://drive.google.com/open?id=1apPyT3IkNwKhwuYyp432IJrTd0QHGbIN), [BaiduYun Driver](https://pan.baidu.com/s/1xOp3_FDk49YlJ-6C-xQfHw)
* COCO [RFB_Net512_E](https://drive.google.com/open?id=1pHDc6Xg9im3affOr7xaimXaRNOHtbaPM), [BaiduYun Driver](https://pan.baidu.com/s/1o8dxrom)
* COCO [RFB_Mobile Net300](https://drive.google.com/open?id=1vmbTWWgeMN_qKVWOeDfl1EN9c7yHPmOe), [BaiduYun Driver](https://pan.baidu.com/s/1bp4ik1L)


