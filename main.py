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
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.distributed as dist
import torch.optim as optim
import time

import apex
from utils.fp16_utils import FP16_Optimizer

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
    parser.add_argument('--start_epoch', type=int,
                            default=0, help='start epoch')
    parser.add_argument('--eval_interval', type=int,
                            default=10, help='interval epoch between evaluations')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--save_dir', type=str,
                        default='save',
                        help='directory where model are saved')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='test model')
    parser.add_argument('-s', '--test_size', type=int, default=416)
    parser.add_argument('--testset', dest='testset', action='store_true', default=False,
                        help='test set evaluation')
    parser.add_argument('--half', dest='half', action='store_true', default=False,
                        help='FP16 training')
    parser.add_argument('--rfb', dest='rfb', action='store_true', default=False,
                        help='Use rfb block')
    parser.add_argument('--asff', dest='asff', action='store_true', default=False,
                        help='Use ASFF module for yolov3')
    parser.add_argument('--dropblock', dest='dropblock', action='store_true', default=False,
                        help='Use dropblock')
    parser.add_argument('--nowd', dest='no_wd', action='store_true', default=False,
                        help='no weight decay for bias')
    parser.add_argument('--vis', dest='vis', action='store_true', default=False,
                        help='visualize fusion weight and detection results')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--tfboard', action='store_true', help='tensorboard path for logging', default=False)
    parser.add_argument('--log_dir', type=str,
                        default='log/',
                        help='directory where tf log are saved')
    return parser.parse_args()

def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    save_prefix = 'yolov3'

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)

    backbone = cfg['MODEL']['BACKBONE']
    lr = cfg['TRAIN']['LR']
    epochs = cfg['TRAIN']['MAXEPOCH']
    cos = cfg['TRAIN']['COS']
    sybn = cfg['TRAIN']['SYBN']
    mixup = cfg['TRAIN']['MIX']
    no_mixup_epochs= cfg['TRAIN']['NO_MIXUP_EPOCHS']
    label_smooth = cfg['TRAIN']['LABAL_SMOOTH']
    momentum = cfg['TRAIN']['MOMENTUM']
    burn_in = cfg['TRAIN']['BURN_IN']
    batch_size = cfg['TRAIN']['BATCHSIZE']
    decay = cfg['TRAIN']['DECAY']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['TRAIN']['RANDRESIZE']
    input_size = (cfg['TRAIN']['IMGSIZE'],cfg['TRAIN']['IMGSIZE'])
    test_size = (args.test_size,args.test_size)
    steps = (180, 240) # for no cos lr shedule training


    # Learning rate setup
    base_lr = lr

    if args.dataset == 'COCO':
        dataset = COCODataset(
                  data_dir='data/COCO/',
                  img_size=input_size,
                  preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),max_labels=50),
                  debug=args.debug)
        num_class = 80
    elif args.dataset == 'VOC':
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(root='data/VOC',
                 image_sets = train_sets,
                 input_dim = input_size,
                 preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),max_labels=30))
        num_class = 20
    else:
        print('Only COCO and VOC datasets are supported!')
        return

    save_prefix += ('_'+args.dataset)

    if label_smooth:
        save_prefix += '_label_smooth'

    # Initiate model
    if args.asff:
        save_prefix += '_asff'
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
            save_prefix += '_mobilev2'
            print("For mobilenet, we currently don't support dropblock, rfb and FeatureAdaption")
        else:
            from models.yolov3_asff import YOLOv3
        print('Training YOLOv3 with ASFF!')
        model = YOLOv3(num_classes = num_class, ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=args.rfb, vis=args.vis, asff=args.asff)
    else:
        save_prefix += '_baseline'
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
            save_prefix += '_mobilev2'
        else:
            from models.yolov3_baseline import YOLOv3
        print('Training YOLOv3 strong baseline!')
        if args.vis:
            print('Visualization is not supported for YOLOv3 baseline model')
            args.vis = False
        model = YOLOv3(num_classes = num_class, ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=args.rfb)


    save_to_disk = (not args.distributed) or distributed_util.get_rank() == 0

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.Conv2d):
                if backbone == 'mobile':
                    init.kaiming_normal_(m.weight, mode='fan_in')
                else:
                    init.kaiming_normal_(m.weight, a=0.1, mode='fan_in')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.zeros_(m.bias)
                m.state_dict()[key][...] = 0

    model.apply(init_yolo)

    if sybn:
        model = apex.parallel.convert_syncbn_model(model)

    if args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        cpu_device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=cpu_device)
        model.load_state_dict(ckpt,strict=False)
        #model.load_state_dict(ckpt)
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

    if args.tfboard and save_to_disk:
        print("using tfboard")
        from torch.utils.tensorboard import SummaryWriter
        tblogger = SummaryWriter(args.log_dir)

    model.train()
    if mixup:
        from dataset.mixupdetection import MixupDetection
        dataset = MixupDetection(dataset,
                  preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),max_labels=50),
                  )
        dataset.set_mixup(np.random.beta, 1.5,1.5)

        save_prefix += '_mixup'

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size,drop_last=False,input_dimension=input_size)
    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=args.n_cpu, pin_memory=True)

    dataiterator = iter(dataloader)

    if args.dataset == 'COCO':
        evaluator = COCOAPIEvaluator(
                    data_dir='data/COCO/',
                    img_size=test_size,
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    testset=args.testset,
                    vis=args.vis)

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
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    vis=args.vis)


    dtype = torch.float16 if args.half else torch.float32

    # optimizer setup
    # set weight decay only on conv.weight
    if args.no_wd:
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'conv.weight' in key:
                params += [{'params':value, 'weight_decay':decay }]
            else:
                params += [{'params':value, 'weight_decay':0.0}]

        save_prefix += '_no_wd'
    else:
        params = model.parameters()

    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay)

    if args.half:
        optimizer = FP16_Optimizer(optimizer,verbose=False)

    if cos:
        save_prefix += '_cos'

    tmp_lr = base_lr

    def set_lr(tmp_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr

    # start training loop
    start = time.time()
    epoch = args.start_epoch
    epoch_size = len(dataset) // (batch_size*args.ngpu)
    while epoch < epochs+1:
        if args.distributed:
            batch_sampler.sampler.set_epoch(epoch)

        if epoch > epochs-no_mixup_epochs+1:
            args.eval_interval = 1
            if mixup:
                print('Disable mix up now!')
                mixup=False
                dataset.set_mixup(None)
                if args.distributed:
                    sampler = torch.utils.data.DistributedSampler(dataset)
                else:
                    sampler = torch.utils.data.RandomSampler(dataset)
                batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size,drop_last=False,input_dimension=input_size)
                dataloader = DataLoader(
                        dataset, batch_sampler=batch_sampler, num_workers=args.n_cpu, pin_memory=True)

        #### DropBlock Shedule #####
        Drop_layer = [16, 24, 33]
        if args.asff:
            Drop_layer = [16, 22, 29]
        if (epoch == 5 or (epoch == args.start_epoch and args.start_epoch > 5)) and (args.dropblock) and backbone!='mobile':
            block_size = [1, 3, 5]
            keep_p = [0.9, 0.9, 0.9]
            for i in range(len(Drop_layer)):
                model.module.module_list[Drop_layer[i]].reset(block_size[i], keep_p[i])

        if (epoch == 80 or (epoch == args.start_epoch and args.start_epoch > 80) ) and (args.dropblock) and backbone!='mobile':
            block_size = [3, 5, 7]
            keep_p = [0.9, 0.9, 0.9]
            for i in range(len(Drop_layer)):
                model.module.module_list[Drop_layer[i]].reset(block_size[i], keep_p[i])

        if (epoch == 150 or (epoch == args.start_epoch and args.start_epoch > 150)) and (args.dropblock) and backbone!='mobile':
            block_size = [7, 7, 7]
            keep_p = [0.9, 0.9, 0.9]
            for i in range(len(Drop_layer)):
                model.module.module_list[Drop_layer[i]].reset(block_size[i], keep_p[i])


        for iter_i,  (imgs, targets,img_info,idx) in enumerate(dataloader):
            #evaluation
            if ((epoch % args.eval_interval == 0)and epoch > args.start_epoch and iter_i == 0) or args.test:
                if not args.test and save_to_disk:
                    torch.save(model.module.state_dict(), os.path.join(args.save_dir,
                            save_prefix+'_'+repr(epoch)+'.pth'))

                if args.distributed:
                    distributed_util.synchronize()
                ap50_95, ap50 = evaluator.evaluate(model, args.half,args.distributed)
                if args.distributed:
                    distributed_util.synchronize()
                if args.test:
                    sys.exit(0) 
                model.train()
                if args.tfboard and save_to_disk:
                    tblogger.add_scalar('val/COCOAP50', ap50, epoch)
                    tblogger.add_scalar('val/COCOAP50_95', ap50_95, epoch)

        # learning rate scheduling (cos or step)
            if epoch < burn_in:
                tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (burn_in*epoch_size), 4)
                set_lr(tmp_lr)
            elif cos:
                if epoch <= epochs-no_mixup_epochs and epoch > 20:
                    min_lr = 0.00001
                    tmp_lr = min_lr + 0.5*(base_lr-min_lr)*(1+math.cos(math.pi*(epoch-20)*1./\
                        (epochs-no_mixup_epochs-20)))
                elif epoch > epochs-no_mixup_epochs:
                    tmp_lr = 0.00001
                set_lr(tmp_lr)

            elif epoch == burn_in:
                tmp_lr = base_lr
                set_lr(tmp_lr)
            elif epoch in steps and iter_i == 0:
                tmp_lr = tmp_lr * 0.1
                set_lr(tmp_lr)


            optimizer.zero_grad()

            imgs = Variable(imgs.to(device).to(dtype))
            targets = Variable(targets.to(device).to(dtype), requires_grad=False)
            loss_dict = model(imgs, targets, epoch)
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            loss = sum(loss for loss in loss_dict['losses'])
            if args.half:
                optimizer.backward(loss)
            else:
                loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()


            if iter_i % 10 == 0 and save_to_disk:
            # logging
                end = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: anchor %.2f, iou %.2f, l1 %.2f, conf %.2f, cls %.2f, imgsize %d, time: %.2f]'
                % (epoch, epochs, iter_i, epoch_size, tmp_lr,
                 sum(anchor_loss for anchor_loss in loss_dict_reduced['anchor_losses']).item(),
                 sum(iou_loss for iou_loss in loss_dict_reduced['iou_losses']).item(),
                 sum(l1_loss for l1_loss in loss_dict_reduced['l1_losses']).item(),
                 sum(conf_loss for conf_loss in loss_dict_reduced['conf_losses']).item(),
                 sum(cls_loss for cls_loss in loss_dict_reduced['cls_losses']).item(),
                 input_size[0], end-start),
                flush=True)

                start = time.time()
                if args.tfboard and save_to_disk:
                    tblogger.add_scalar('train/total_loss',
                            sum(loss for loss in loss_dict_reduced['losses']).item(),
                            epoch*epoch_size+iter_i)

            # random resizing
            if random_resize and iter_i %10 == 0 and iter_i > 0:
                tensor = torch.LongTensor(1).to(device)
                if args.distributed:
                    distributed_util.synchronize()

                if save_to_disk:
                    if epoch > epochs-10:
                        size = 416 if args.dataset=='VOC' else 608
                    else:
                        size = random.randint(*(10,19))
                        size = int(32 * size)
                    tensor.fill_(size)

                if args.distributed:
                    distributed_util.synchronize()
                    dist.broadcast(tensor, 0)

                input_size = dataloader.change_input_dim(multiple=tensor.item(), random_range=None)

                if args.distributed:
                    distributed_util.synchronize()

        epoch +=1
    if not args.test and save_to_disk:
        torch.save(model.module.state_dict(), os.path.join(args.save_dir,
            "yolov3_"+args.dataset+'_Final.pth'))
    
    if args.distributed:
        distributed_util.synchronize()
    ap50_95, ap50 = evaluator.evaluate(model, args.half)

    if args.tfboard and save_to_disk:
        tblogger.close()


if __name__ == '__main__':
    main()
