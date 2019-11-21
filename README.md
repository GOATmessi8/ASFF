# Learning Spatial Fusion for Single-Shot Object Detection

By Songtao Liu, Di Huang, Yunhong Wang

### Introduction
In this work, we propose a novel and data driven strategy for pyramidal feature fusion, referred to as adaptively spatial feature fusion (ASFF). It learns the way to spatially filter conflictive information to suppress the inconsistency, thus improving the scale-invariance of features, and introduces nearly free inference overhead. For more details, please refer to our [arXiv paper]().

<img align="center" src="https://github.com/ruinmessi/ASFF/blob/master/doc/asff.png">


### COCO 
| System |  *test-dev mAP* | **Time** (V100) | **Time** (2080ti)|
|:-------|:-----:|:-------:|:-------:|
| [YOLOv3 608](http://pjreddie.com/darknet/yolo/) | 33.0 | 20ms| 24ms|
| YOLOv3 608+ [BoFs](https://arxiv.org/abs/1902.04103) | 37.0 | 20ms | 24ms|
| YOLOv3 608(ours baseline) | **38.8** | 20ms | 24ms|
| YOLOv3 608+ ASFF | **40.6** | 22ms | 28ms| 
| YOLOv3 608+ ASFF\* | **42.4** | 22ms | 29ms| 
| YOLOv3 800+ ASFF\* | **43.9** | 34ms | 40ms| 


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
- Compile the DCN layer (ported from [DCNv2 implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0)):
```Shell
./make.sh
```

### Prerequisites
- We also use [apex](https://github.com/NVIDIA/apex), numpy, opencv, tqdm, pyyaml, matplotlib, scikit-image...
    * Note: We use apex for distributed training and synchronized batch normalization. For FP16 training, since the current apex version have some [issues](https://github.com/NVIDIA/apex/issues/318), we use the old version of FP16_Optimizer, and split the code in ./utils/fp_utils.

- We also support tensorboard if you have installed it.   

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

### VOC Dataset
Install the VOC dataset as ./data/VOC. We also recommend a soft-link:
```
ln -s /path/to/VOCdevkit ./data/VOC
```

## Training

- First download the mix-up pretrained [Darknet-53](https://arxiv.org/abs/1902.04103) PyTorch base network weights at: https://drive.google.com/open?id=1phqyYhV1K9KZLQZH1kENTAPprLBmymfP  
  or from our [BaiduYun Driver](https://pan.baidu.com/s/19PaXl6p9vXHG2ZuGqtfLOg) 

- By default, we assume you have downloaded the file in the `ASFF/weights` dir:

- Since random resizing consumes much more GPU memory, we implement FP16 training with an old version of apex. 

- We currently **ONLY** test the code with distributed training on multiple GPUs (10 2080ti or 4 Tesla V100).

- To train YOLOv3 baseline (ours) using the train script simply specify the parameters listed in `main.py` as a flag or manually change them on config/yolov3_baseline.cfg:
```Shell
python -m torch.distributed.launch --nproc_per_node=10 --master_port=${RANDOM+10000} main.py \
--cfg config/yolov3_baseline.cfg -d COCO --tfboard --distributed --ngpu 10 \
--checkpoint weights/darknet53_feature_mx.pth --start_epoch 0 --half --log_dir log/COCO -s 608 
```
- Note:
  * --cfg: config files.
  * --tfboard: use tensorboard.
  * --distributed: distributed training (we only test the code with distributed training)
  * -d: choose datasets, COCO or VOC.
  * --ngpu: number of GPUs.
  * -c, --checkpoint: pretrained weights or resume weights. You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `main.py` for options)
 
  * --start_epoch: used for resume training.
  * --half: FP16 training.
  * --log_dir: log dir for tensorboard.
  * -s: evaluation image size, from 320 to 608 as in YOLOv3.

- To train YOLOv3 with ASFF or ASFF\*, you only need add some addional flags:
```Shell
python -m torch.distributed.launch --nproc_per_node=10 --master_port=${RANDOM+10000} main.py \
--cfg config/yolov3_baseline.cfg -d COCO --tfboard --distributed --ngpu 10 \
--checkpoint weights/darknet53_feature_mx.pth --start_epoch 0 --half --asff --rfb --dropblock \
--log_dir log/COCO_ASFF -s 608 
```
- Note:
  * --asff: add ASFF module on YOLOv3.
  * --rfb: use [RFB](https://github.com/ruinmessi/RFBNet) moduel on ASFF.
  * --dropblock: use [DropBlock](https://arxiv.org/abs/1810.12890).
  
## Evaluation
To evaluate a trained network is simple, just add the --test flag on the training script above:

```Shell
python -m torch.distributed.launch --nproc_per_node=10 --master_port=${RANDOM+10000} main.py --cfg config/yolov3_baseline.cfg -d COCO --t    fboard --distributed --ngpu 10 --checkpoint /path/to/you/weights --start_epoch 0 --half --asff --rfb --log_dir log/COCO_ASFF -s 608 --test
```
- Note:
  * --vis: Visualization of ASFF.
  * --testset: evaluate on COCO *test-dev*.
  * -s: evaluation image size.

By default, it will directly output the mAP results on COCO *val2017* or VOC *test 2007*. 

## Models

* yolov3_baseline (ours) [weights](https://drive.google.com/open?id=1RbjUQbNxl4cEbk-6jFkFnOHRukJY5EQk) [baiduYun](https://pan.baidu.com/s/131JhlaOBbeL9l4tqiJO9yA)

* yolov3_asff [weights](https://drive.google.com/open?id=1Dyf8ZEga_VT2O3_c5nrFJA5uON1aSJK-) [baiduYun](https://pan.baidu.com/s/1a-eQZ0kDpsnUooD4RtRdxg)

* yolov3_asff\* (320-608) [weights](https://drive.google.com/open?id=1N668Za8OBbJbUStYde0ml9SZdM7tabXy) [baiduYun](https://pan.baidu.com/s/1d9hOQBj20HCy51qWbonxMQ)

* yolov3_asff\* (480-800) [weights](https://drive.google.com/open?id=18N4_nNVqYbjawerEHQnwJGPcRvcLOe06) [baiduYun](https://pan.baidu.com/s/1HERhiP4vmUekxxm5KQrX8g)


