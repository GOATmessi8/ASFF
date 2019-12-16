from torch import nn
from .network_blocks import *
from .yolov3_head import YOLOv3Head


def create_yolov3_mobilenet_v2(num_classes, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
    """
    MobileNet V2 main class

    Args:
        num_classes (int): Number of classes
        width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
        inverted_residual_setting: Network structure
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        Set to 1 to turn off rounding
    """
    block = InvertedResidual
    input_channel = 32
    last_channel = 1280

    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    # only check the first element, assuming user knows t,c,n,s are required
    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
        raise ValueError("inverted_residual_setting should be non-empty "
                         "or a 4-element list, got {}".format(inverted_residual_setting))

    # building first layer
    input_channel = make_divisible(input_channel * width_mult, round_nearest)
    last_channel = make_divisible(last_channel * max(1.0, width_mult), round_nearest)
    mlist = nn.ModuleList()
    mlist.append(ConvBNReLU(3, input_channel, stride=2))
    # building inverted residual blocks
    for t, c, n, s in inverted_residual_setting:
        output_channel =make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            mlist.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
    # building last several layers
    mlist.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))   #18

    # YOLOv3
    mlist.append(ressepblock(last_channel, 1024, in_ch=512, shortcut=False))               #19
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1,leaky=False))      #20
    # SPP Layer
    mlist.append(SPPLayer())                                               #21

    mlist.append(add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1, leaky=False))      #22
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1,leaky=False))   #23
    mlist.append(DropBlock(block_size=1, keep_prob=1))                     #24
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1, leaky=False))      #25 (17)

    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1, leaky=False))        #26
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #27
    mlist.append(add_conv(in_ch=352, out_ch=256, ksize=1, stride=1,leaky=False))        #28
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1,leaky=False))     #29
    mlist.append(DropBlock(block_size=1, keep_prob=1))                      #30
    mlist.append(ressepblock(512, 512, in_ch=256,shortcut=False))        #31
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1,leaky=False))        #32
    # 2nd yolo branch

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1,leaky=False))        #33
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #34
    mlist.append(add_conv(in_ch=160, out_ch=128, ksize=1, stride=1,leaky=False))        #35
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1,leaky=False))     #36
    mlist.append(DropBlock(block_size=1, keep_prob=1))                      #37
    mlist.append(ressepblock(256, 256, in_ch=128,shortcut=False))        #38
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1,leaky=False))        #39

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
        self.module_list = create_yolov3_mobilenet_v2(num_classes)

        if asff:
            self.level_0_conv =ASFFmobile(level=0,rfb=rfb,vis=vis)
        else:
            self.level_0_conv =add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1,leaky=False)  

        self.level_0_header = YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=32, in_ch=1024,
                              ignore_thre=ignore_thre,label_smooth = label_smooth, rfb=rfb, sep=True)

        if asff:
            self.level_1_conv =ASFFmobile(level=1,rfb=rfb,vis=vis)
        else:
            self.level_1_conv =add_conv(in_ch=256, out_ch=512, ksize=3, stride=1,leaky=False)  

        self.level_1_header = YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=16, in_ch=512,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb, sep=True)

        if asff:
            self.level_2_conv =ASFFmobile(level=2,rfb=rfb,vis=vis)
        else:
            self.level_2_conv =add_conv(in_ch=128, out_ch=256, ksize=3, stride=1,leaky=False)  

        self.level_2_header = YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=8, in_ch=256,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb, sep=True)
        self.asff = asff

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

        for i, module in enumerate(self.module_list):

            # yolo layers
            x = module(x)

            # route layers
            if i in [6, 13, 25, 32, 39]:
                route_layers.append(x)
            if i == 27:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 34:
                x = torch.cat((x, route_layers[0]), 1)
        

        for l in range(3):
            conver = getattr(self, 'level_{}_conv'.format(l))
            header = getattr(self, 'level_{}_header'.format(l))
            if self.asff:
                f_conv= conver(route_layers[2],route_layers[3],route_layers[4])
            else:
                f_conv = conver(route_layers[l+2])
            if train:
                x, anchor_loss, iou_loss, l1_loss, conf_loss, cls_loss = header(f_conv, targets)
                anchor_losses.append(anchor_loss)
                iou_losses.append(iou_loss)
                l1_losses.append(l1_loss)
                conf_losses.append(conf_loss)
                cls_losses.append(cls_loss)
            else:
                x = header(f_conv)

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
            return torch.cat(output, 1)
