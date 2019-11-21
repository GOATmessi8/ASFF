import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IOUWH_loss(nn.Module): #used for anchor guiding
    def __init__(self, reduction='none'):
        super(IOUWH_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        target[:,:2] = 0
        tl = torch.max((target[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))

        br = torch.min((target[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        U = area_p+area_g-area_i+ 1e-16
        iou= area_i / U

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class IOUloss(nn.Module):
    def __init__(self, reduction='none'):
        super(IOUloss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        tl = torch.max((pred[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))
        br = torch.min((pred[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        iou= (area_i) / (area_p+area_g-area_i+ 1e-16)

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
