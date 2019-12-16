import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bboxes_iou
import numpy as np
from .utils_loss import *
from .network_blocks import *

class YOLOv3Head(nn.Module):
    def __init__(self, anch_mask, n_classes, stride, in_ch=1024, ignore_thre=0.7, label_smooth = False, rfb=False, sep=False):
        super(YOLOv3Head, self).__init__()
        self.anchors = [
            (10, 13), (16, 30), (33, 23), 
            (30, 61), (62, 45), (42, 119),
            (116, 90), (156, 198), (121, 240) ]
        if sep:
            self.anchors = [
                (10, 13), (16, 30), (33, 23), 
                (30, 61), (62, 45), (42, 119),
                (116, 90), (156, 198), (373, 326)]

        self.anch_mask = anch_mask
        self.n_anchors = 4
        self.n_classes = n_classes
        self.guide_wh = nn.Conv2d(in_channels=in_ch,
                              out_channels=2*self.n_anchors, kernel_size=1, stride=1, padding=0)
        self.Feature_adaption=FeatureAdaption(in_ch, in_ch, self.n_anchors, rfb, sep)

        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors*(self.n_classes+5), kernel_size=1, stride=1, padding=0)
        self.ignore_thre = ignore_thre
        self.l1_loss = nn.L1Loss(reduction='none')
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.iou_loss = IOUloss(reduction='none')
        self.iou_wh_loss = IOUWH_loss(reduction='none')
        self.stride = stride
        self._label_smooth = label_smooth

        self.all_anchors_grid = self.anchors
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """

        wh_pred = self.guide_wh(xin) #Anchor guiding

        if xin.type() == 'torch.cuda.HalfTensor': #As DCN only support FP32 now, change the feature to float.
            wh_pred = wh_pred.float()
            if labels is not None:
                labels = labels.float()
            self.Feature_adaption = self.Feature_adaption.float()
            self.conv = self.conv.float()
            xin = xin.float()

        feature_adapted = self.Feature_adaption(xin, wh_pred)

        output = self.conv(feature_adapted)
        wh_pred = torch.exp(wh_pred)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        image_size = fsize * self.stride
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        wh_pred = wh_pred.view(batchsize, self.n_anchors, 2 , fsize, fsize)
        wh_pred = wh_pred.permute(0, 1, 3, 4, 2).contiguous()

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0,1,3,4,2).contiguous()

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors-1, 1, 1)), [batchsize, self.n_anchors-1, fsize, fsize]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors-1, 1, 1)), [batchsize, self.n_anchors-1, fsize, fsize]))

        default_center = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(dtype)

        pred_anchors = torch.cat((default_center, wh_pred), dim=-1).contiguous()

        anchors_based = pred_anchors[:, :self.n_anchors-1, :, :, :]   #anchor branch
        anchors_free = pred_anchors[:, self.n_anchors-1, :, :, :]     #anchor free branch
        anchors_based[...,2] *= w_anchors
        anchors_based[...,3] *= h_anchors
        anchors_free[...,2] *= self.stride*4
        anchors_free[...,3] *= self.stride*4
        pred_anchors[...,:2] = pred_anchors[...,:2].detach()

        if not self.training:

            pred = output.clone()
            pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                    pred[...,np.r_[:2, 4:n_ch]])
            pred[...,0] += x_shift
            pred[...,1] += y_shift
            pred[...,:2] *= self.stride
            pred[...,2] = torch.exp(pred[...,2])*(pred_anchors[...,2])
            pred[...,3] = torch.exp(pred[...,3])*(pred_anchors[...,3])
            refined_pred = pred.view(batchsize, -1, n_ch)
            return refined_pred.data

        #training for anchor prediction
        if self.training:

            target = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, n_ch).type(dtype)
            l1_target = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 4).type(dtype)
            tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 4).type(dtype)
            obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).type(dtype)

            cls_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, self.n_classes).type(dtype)
            coord_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize).type(dtype)
            anchor_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize).type(dtype)

            labels = labels.data
            mixup = labels.shape[2]>5
            if mixup:
                label_cut = labels[...,:5]
            else:
                label_cut = labels
            nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

            truth_x_all = labels[:, :, 1] * 1.
            truth_y_all = labels[:, :, 2] * 1.
            truth_w_all = labels[:, :, 3] * 1.
            truth_h_all = labels[:, :, 4] * 1.
            truth_i_all = (truth_x_all/image_size*fsize).to(torch.int16).cpu().numpy()
            truth_j_all = (truth_y_all/image_size*fsize).to(torch.int16).cpu().numpy()

            pred = output.clone()
            pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                    pred[...,np.r_[:2, 4:n_ch]])
            pred[...,0] += x_shift
            pred[...,1] += y_shift
            pred[...,2] = torch.exp(pred[...,2])*(pred_anchors[...,2])
            pred[...,3] = torch.exp(pred[...,3])*(pred_anchors[...,3])
            pred[...,:2] *= self.stride

            pred_boxes = pred[...,:4].data
            for b in range(batchsize):
                n = int(nlabel[b])
                if n == 0:
                    continue

                truth_box = dtype(np.zeros((n, 4)))
                truth_box[:n, 2] = truth_w_all[b, :n]
                truth_box[:n, 3] = truth_h_all[b, :n]
                truth_i = truth_i_all[b, :n]
                truth_j = truth_j_all[b, :n]

                # calculate iou between truth and reference anchors
                anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors, xyxy=False)
                best_n_all = np.argmax(anchor_ious_all, axis=1)
                best_anchor_iou = anchor_ious_all[np.arange(anchor_ious_all.shape[0]),best_n_all]
                best_n = best_n_all % 3
                best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                    best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

                truth_box[:n, 0] = truth_x_all[b, :n]
                truth_box[:n, 1] = truth_y_all[b, :n]
                pred_box = pred_boxes[b]
                pred_ious = bboxes_iou(pred_box.view(-1,4),
                        truth_box, xyxy=False)
                pred_best_iou, _= pred_ious.max(dim=1)
                pred_best_iou = (pred_best_iou > self.ignore_thre)
                pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
                obj_mask[b]= ~pred_best_iou
                truth_box[:n, 0] = 0
                truth_box[:n, 1] = 0

                if sum(best_n_mask) == 0:
                    continue
                for ti in range(best_n.shape[0]):
                    if best_n_mask[ti] == 1:
                        i, j = truth_i[ti], truth_j[ti]
                        a = best_n[ti]
                        free_iou = bboxes_iou(truth_box[ti].cpu().view(-1,4),
                                pred_anchors[b, self.n_anchors-1, j, i, :4].data.cpu().view(-1,4),xyxy=False)  #iou of pred anchor 

                        #choose the best anchor
                        if free_iou > best_anchor_iou[ti]:
                            aa = self.n_anchors-1
                        else:
                            aa = a

                        cls_mask[b, aa, j, i, :] = 1
                        coord_mask[b, aa, j, i] = 1

                        anchor_mask[b, self.n_anchors-1, j, i] = 1
                        anchor_mask[b, a, j, i] = 1

                        obj_mask[b, aa, j, i]= 1 if not mixup else labels[b, ti, 5]

                        target[b, a, j, i, 0] = truth_x_all[b, ti]
                        target[b, a, j, i, 1] = truth_y_all[b, ti]
                        target[b, a, j, i, 2] = truth_w_all[b, ti]
                        target[b, a, j, i, 3] = truth_h_all[b, ti]

                        target[b, self.n_anchors-1, j, i, 0] = truth_x_all[b, ti]
                        target[b, self.n_anchors-1, j, i, 1] = truth_y_all[b, ti]
                        target[b, self.n_anchors-1, j, i, 2] = truth_w_all[b, ti]
                        target[b, self.n_anchors-1, j, i, 3] = truth_h_all[b, ti]

                        l1_target[b, aa, j, i, 0] = truth_x_all[b, ti]/image_size *fsize - i*1.0
                        l1_target[b, aa, j, i, 1] = truth_y_all[b, ti]/image_size *fsize - j*1.0
                        l1_target[b, aa, j, i, 2] = torch.log(truth_w_all[b, ti]/\
                            (pred_anchors[b, aa, j, i, 2])+ 1e-12)
                        l1_target[b, aa, j, i, 3] = torch.log(truth_h_all[b, ti]/\
                            (pred_anchors[b, aa, j, i, 3]) + 1e-12)
                        target[b, aa, j, i, 4] = 1
                        if self._label_smooth:
                            smooth_delta = 1
                            smooth_weight = 1. / self.n_classes
                            target[b, aa, j, i, 5:]= smooth_weight* smooth_delta

                            target[b, aa, j, i, 5 + labels[b, ti,
                                0].to(torch.int16)] = 1 - smooth_delta*smooth_weight
                        else:
                            target[b,aa, j, i, 5 + labels[b, ti,
                                0].to(torch.int16)] = 1

                        tgt_scale[b, aa,j, i, :] = 2.0 - truth_w_all[b, ti]*truth_h_all[b, ti] / image_size/image_size


            # Anchor loss
            anchorcoord_mask = anchor_mask>0
            loss_anchor = self.iou_wh_loss(pred_anchors[...,:4][anchorcoord_mask], target[...,:4][anchorcoord_mask]).sum()/batchsize

            #Prediction loss
            coord_mask = coord_mask>0
            loss_iou = (tgt_scale[coord_mask][...,0]*\
                    self.iou_loss(pred[..., :4][coord_mask], target[..., :4][coord_mask])).sum() / batchsize
            tgt_scale = tgt_scale[...,:2]
            loss_xy = (tgt_scale*self.bcewithlog_loss(output[...,:2], l1_target[...,:2])).sum() / batchsize
            loss_wh = (tgt_scale*self.l1_loss(output[...,2:4], l1_target[...,2:4])).sum() / batchsize
            loss_l1 = loss_xy + loss_wh
            loss_obj = (obj_mask*(self.bcewithlog_loss(output[..., 4], target[..., 4]))).sum() / batchsize
            loss_cls = (cls_mask*(self.bcewithlog_loss(output[..., 5:], target[..., 5:]))).sum()/ batchsize

            loss = loss_anchor + loss_iou + loss_l1+ loss_obj + loss_cls

            return loss, loss_anchor, loss_iou, loss_l1, loss_obj, loss_cls

