from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.voc_evaluator import VOCEvaluator
from utils import distributed_util
from utils.distributed_util import reduce_loss_dict
from dataset.cocodataset import *
from dataset.vocdataset import *
from dataset.data_augment import TrainTransform
from dataset.dataloading import *

import os
import sys
import argparse
import yaml
import random
import math
import cv2
cv2.setNumThreads(0)

import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.distributed as dist
import time

import apex

######## unlimit the resource in some dockers or cloud machines ####### 
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_baseline.cfg',
                        help='config file. see readme')
    parser.add_argument('-d', '--dataset', type=str,
                        default='COCO', help='COCO or VOC dataset')
    parser.add_argument('--n_cpu', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--distributed', dest='distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int,
                            default=0, help='local_rank')
    parser.add_argument('--ngpu', type=int, default=10,
                        help='number of gpu')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('-s', '--test_size', type=int, default=416)
    parser.add_argument('--testset', dest='testset', action='store_true', default=False,
                        help='test set evaluation')
    parser.add_argument('--half', dest='half', action='store_true', default=False,
                        help='FP16 training')
    parser.add_argument('--rfb', dest='rfb', action='store_true', default=False,
                        help='Use rfb block')
    parser.add_argument('--asff', dest='asff', action='store_true', default=False,
                        help='Use ASFF module for yolov3')
    parser.add_argument('--vis', dest='vis', action='store_true', default=False,
                        help='visualize fusion weight and detection results')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    return parser.parse_args()

def eval():
    """
    YOLOv3 evaler. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")


    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)

    backbone=cfg['MODEL']['BACKBONE']
    test_size = (args.test_size,args.test_size)

    if args.dataset == 'COCO':
        evaluator = COCOAPIEvaluator(
                    data_dir='data/COCO/',
                    img_size=test_size,
                    confthre=0.001,
                    nmsthre=0.65,
                    testset=args.testset,
                    vis=args.vis)

        num_class=80

    elif args.dataset == 'VOC':
        '''
        # COCO style evaluation, you have to convert xml annotation files into a json file.
        evaluator = COCOAPIEvaluator(
                    data_dir='data/VOC/',
                    img_size=test_size,
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    testset=args.testset,
                    voc = True)
        '''
        evaluator = VOCEvaluator(
                    data_dir='data/VOC/',
                    img_size=test_size,
                    confthre=0.001,
                    nmsthre=0.65,
                    vis=args.vis)
        num_class=20
    # Initiate model
    if args.asff:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
            print("For mobilenet, we currently don't support dropblock, rfb and FeatureAdaption")
        else:
            from models.yolov3_asff import YOLOv3
        print('Training YOLOv3 with ASFF!')
        model = YOLOv3(num_classes = num_class, rfb=args.rfb, vis=args.vis, asff=args.asff)
    else:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
        else:
            from models.yolov3_baseline import YOLOv3
        print('Training YOLOv3 strong baseline!')
        if args.vis:
            print('Visualization is not supported for YOLOv3 baseline model')
            args.vis = False
        model = YOLOv3(num_classes = num_class, rfb=args.rfb)

    save_to_disk = (not args.distributed) or distributed_util.get_rank() == 0

    if args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        cpu_device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=cpu_device)
        #model.load_state_dict(ckpt,strict=False)
        model.load_state_dict(ckpt)
    if cuda:
        print("using cuda")
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        model = model.to(device)

    if args.half:
        model = model.half()

    if args.ngpu > 1:
        if args.distributed:
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
            #model = apex.parallel.DistributedDataParallel(model)
        else:
            model = nn.DataParallel(model) 

    dtype = torch.float16 if args.half else torch.float32

    if args.distributed:
        distributed_util.synchronize()

    ap50_95, ap50 = evaluator.evaluate(model, args.half, args.distributed)

    if args.distributed:
        distributed_util.synchronize()
    sys.exit(0) 


if __name__ == '__main__':
    eval()
