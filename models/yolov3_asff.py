import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import *
from .yolov3_head import YOLOv3Head

from collections import defaultdict

def build_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb):
    """
    Build yolov3 layer modules.
    Args:
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """
    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))           #0
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))          #1
    mlist.append(resblock(ch=64))                                           #2
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))         #3
    mlist.append(resblock(ch=128, nblocks=2))                               #4
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))        #5
    mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here     #6
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))        #7
    mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here     #8
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))       #9
    mlist.append(resblock(ch=1024, nblocks=4))                              #10

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=1, shortcut=False))              #11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #12
    #SPP Layer
    mlist.append(SPPLayer())                                                #13

    mlist.append(add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1))       #14
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))       #15
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #16
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #17

    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #18
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #19
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))        #20
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))        #21
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #22
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))               #23
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #24
    # 2nd yolo branch

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #25
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #26
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))        #27
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #28
    mlist.append(DropBlock(block_size=1, keep_prob=1))                    #29
    mlist.append(resblock(ch=256, nblocks=1, shortcut=False))               #30
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #31
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #32

    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, num_classes = 80, ignore_thre=0.7, label_smooth = False, rfb=False, vis=False, asff=False):
        """
        Initialization of YOLOv3 class.
        Args:
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()
        self.module_list = build_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb)


        self.level_0_fusion = ASFF(level=0,rfb=rfb,vis=vis)

        self.level_0_header = YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=32, in_ch=1024,
                              ignore_thre=ignore_thre,label_smooth = label_smooth, rfb=rfb)

        self.level_1_fusion = ASFF(level=1,rfb=rfb,vis=vis)

        self.level_1_header = YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=16, in_ch=512,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb)

        self.level_2_fusion = ASFF(level=2,rfb=rfb,vis=vis)

        self.level_2_header = YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=8, in_ch=256,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb)
        self.vis=vis

    def forward(self, x, targets=None, epoch=0):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """

        train = targets is not None
        output = []
        anchor_losses= []
        iou_losses = []
        l1_losses = []
        conf_losses = []
        cls_losses = []
        route_layers = []
        if self.vis:
            fuse_wegihts = []
            fuse_fs = []

        for i, module in enumerate(self.module_list):

            # yolo layers
            x = module(x)

            # route layers
            if i in [6, 8, 17, 24, 32]:
                route_layers.append(x)
            if i == 19:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 26:
                x = torch.cat((x, route_layers[0]), 1)
        

        for l in range(3):
            fusion = getattr(self, 'level_{}_fusion'.format(l))
            header = getattr(self, 'level_{}_header'.format(l))

            if self.vis:
                fused, weight, fuse_f = fusion(route_layers[2],route_layers[3],route_layers[4])
                fuse_wegihts.append(weight)
                fuse_fs.append(fuse_f)
            else:
                fused = fusion(route_layers[2],route_layers[3],route_layers[4])

            if train:
                x, anchor_loss, iou_loss, l1_loss, conf_loss, cls_loss = header(fused, targets)
                anchor_losses.append(anchor_loss)
                iou_losses.append(iou_loss)
                l1_losses.append(l1_loss)
                conf_losses.append(conf_loss)
                cls_losses.append(cls_loss)
            else:
                x = header(fused)

            output.append(x)

        if train:
            losses = torch.stack(output, 0).unsqueeze(0).sum(1,keepdim=True)
            anchor_losses = torch.stack(anchor_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            iou_losses = torch.stack(iou_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            l1_losses = torch.stack(l1_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            conf_losses = torch.stack(conf_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            cls_losses = torch.stack(cls_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            loss_dict = dict(
                    losses = losses,
                    anchor_losses = anchor_losses,
                    iou_losses = iou_losses,
                    l1_losses = l1_losses,
                    conf_losses = conf_losses,
                    cls_losses = cls_losses,
            )
            return loss_dict
        else:
            if self.vis:
                return torch.cat(output, 1), fuse_wegihts, fuse_fs
            else:
                return torch.cat(output, 1)

