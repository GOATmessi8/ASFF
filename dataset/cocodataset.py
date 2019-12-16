import os
import numpy as np

import torch
from .dataloading import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *

COCO_CLASSES=(
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='data/COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=(416,416), preproc=None, debug=False, voc=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            debug (bool): if True, only one data id is selected from the dataset
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.voc = voc
        if voc:
            self.coco = COCO(self.data_dir+'VOC2007/Annotations/'+self.json_file)
        else:
            self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c['name'] for c in cats])
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        id_ = self.ids[index]

        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann['width']
        height = im_ann['height']
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.data_dir, 'images', self.name,
                                #'COCO_'+self.name+'_'+'{:012}'.format(id_) + '.jpg')
                                '{:012}'.format(id_) + '.jpg')

        if self.voc:
            file_name = im_ann['file_name']
            img_file = os.path.join(self.data_dir, 'VOC2007', 'JPEGImages',
                                file_name)

        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'images', 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)
        assert img is not None

        #img, info_img = preprocess(img, self.input_dim[0])

        # load labels
        valid_objs = []
        for obj in annotations:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj['category_id'])
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        img_info = (width, height)

        return img, res, img_info, id_

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        img, res, img_info, id_ = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, res, self.input_dim)


        return img, target, img_info, id_
