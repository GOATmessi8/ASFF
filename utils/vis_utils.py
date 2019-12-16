# -*- coding: utf-8 -*-

import numpy as np
import os 
import matplotlib

matplotlib.use('AGG')

import matplotlib.pyplot as plt
import torch
import cv2
import math
from skimage import transform

def make_vis(dataset, index, img, fuse_weights, fused_fs):
    save_dir = 'vis_output/{}/{}'.format(dataset,index)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(fuse_weights)):
        weights = fuse_weights[i].float().cpu().squeeze().numpy()
        max_v = weights.max()
        min_v = weights.min()
        for j in range(3):
            v = weights[j,:,:]
            save_name = os.path.join(save_dir, 'level_{}_weight_{}.png'.format(i+1,j+1))
            add_heat(img, v, max_v, min_v, save=save_name)

        fused_f = fused_fs[i].float().cpu().squeeze().numpy()
        max_f = fused_f.max()
        min_f = fused_f.min()
        save_f_name = os.path.join(save_dir, 'fused_feature_level_{}.png'.format(i+1))
        add_heat(img, fused_f, max_f, min_f, save=save_f_name)

def make_pred_vis(dataset,index, img, class_names, bboxes, cls, scores):
    save_preddir = 'vis_output/{}/pred/'.format(dataset)
    os.makedirs(save_preddir, exist_ok=True)

    save_pred_name = os.path.join(save_preddir,'{}.png'.format(index))

    bboxes = bboxes.numpy()
    scores = scores.numpy()
    cls_ids = cls.numpy()

    im = vis(img, bboxes, scores, cls_ids, class_names)
    
    cv2.imwrite(save_pred_name, im)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, color=None):

    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        cls_conf = scores[i]
        if cls_conf < conf:
            continue
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[0]+box[2])
        y2 = int(box[1]+box[3])


        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if class_names is not None:
            cls_conf = scores[i]
            cls_id = int(cls_ids[i])
            class_name = class_names[cls_id]
            classes = len(class_names)
            offset = cls_id * 123456 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, '%s: %.2f'%(class_name,cls_conf), (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    return img

def add_heat(image, heat_map, max_v, min_v, alpha=0.4, save=None, cmap='jet', axis='off'):
    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = max_v
    min_value = min_v
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)




