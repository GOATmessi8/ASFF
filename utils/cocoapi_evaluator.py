import json
import tempfile
import sys
from tqdm import tqdm

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from dataset.cocodataset import *
from dataset.data_augment import ValTransform
from utils.utils import *
from utils import distributed_util
from utils.vis_utils import make_vis, make_pred_vis
import time
import apex

DEBUG =False

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = distributed_util.scatter_gather(predictions_per_gpu)
    if not distributed_util.is_main_process():
        return
    # merge the list of dicts
    predictions = []
    for p in all_predictions:
        for a in p:
            predictions.append(a)

    return predictions

class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, img_size, confthre, nmsthre, testset=False, voc=False, vis=False):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        json_f = 'instances_val2017.json'
        name='val2017'
        if testset:
            json_f = 'image_info_test-dev2017.json'
            name='test2017'
        if voc:
            json_f = 'pascal_test2007.json'

        self.testset= testset
        self.dataset = COCODataset(data_dir=data_dir,
                                   img_size=img_size,
                                   json_file=json_f,
                                   preproc = ValTransform(rgb_means=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
                                   name=name,
                                   voc = voc)

        self.num_images = len(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.voc = voc
        self.vis = vis

    def evaluate(self, model, half=False, distributed=False):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        if isinstance(model, apex.parallel.DistributedDataParallel):
            model = model.module
            distributed=True

        model=model.eval()
        cuda = torch.cuda.is_available()
        if half:
            Tensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
        else:
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ids = []
        data_dict = []
        img_num = 0

        indices = list(range(self.num_images))
        if distributed:
            dis_indices = indices[distributed_util.get_rank()::distributed_util.get_world_size()]
        else:
            dis_indices = indices
        progress_bar = tqdm if distributed_util.is_main_process() else iter
        num_classes = 80 if not self.voc else 20

        inference_time=0
        nms_time=0
        n_samples=len(dis_indices)-10

        for k, i in enumerate(progress_bar(dis_indices)):
            img, _, info_img, id_ = self.dataset[i]  # load a batch
            info_img = [float(info) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(Tensor).unsqueeze(0))
                if k > 9:
                    start=time.time()

                if self.vis:
                    outputs,fuse_weights,fused_f = model(img)
                else:
                    outputs = model(img)

                if k > 9:
                    infer_end=time.time()
                    inference_time += (infer_end-start)

                outputs = postprocess(
                    outputs, num_classes, self.confthre, self.nmsthre)

                if k > 9:
                    nms_end=time.time()
                    nms_time +=(nms_end-infer_end)

                if outputs[0] is None:
                    continue
                outputs = outputs[0].cpu().data

            bboxes = outputs[:, 0:4]
            bboxes[:, 0::2] *= info_img[0] / self.img_size[0]
            bboxes[:, 1::2] *= info_img[1] / self.img_size[1]
            bboxes[:, 2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:, 3] = bboxes[:,3] - bboxes[:,1]
            cls = outputs[:, 6]
            scores = outputs[:, 4]* outputs[:,5]
            for ind in range(bboxes.shape[0]):
                label = self.dataset.class_ids[int(cls[ind])]
                A = {"image_id": id_, "category_id": label, "bbox": bboxes[ind].numpy().tolist(),
                 "score": scores[ind].numpy().item(), "segmentation": []} # COCO json format
                data_dict.append(A)
            
            if self.vis:
                o_img,_,_,_  = self.dataset.pull_item(i)
                make_vis('COCO', i, o_img, fuse_weights, fused_f)
                class_names = self.dataset._classes
                make_pred_vis('COCO', i, o_img, class_names, bboxes, cls, scores)

            if DEBUG and distributed_util.is_main_process():
                o_img,_  = self.dataset.pull_item(i)
                class_names = self.dataset._classes
                make_pred_vis('COCO', i, o_img, class_names, bboxes, cls, scores)

        if distributed:
            distributed_util.synchronize()
            data_dict = _accumulate_predictions_from_multiple_gpus(data_dict)
            inference_time = torch.FloatTensor(1).type(Tensor).fill_(inference_time)
            nms_time = torch.FloatTensor(1).type(Tensor).fill_(nms_time)
            n_samples = torch.LongTensor(1).type(Tensor).fill_(n_samples)
            distributed_util.synchronize()
            torch.distributed.reduce(inference_time, dst=0)
            torch.distributed.reduce(nms_time, dst=0)
            torch.distributed.reduce(n_samples, dst=0)
            inference_time = inference_time.item()
            nms_time = nms_time.item()
            n_samples = n_samples.item()

        if not distributed_util.is_main_process():
            return 0, 0


        print('Main process Evaluating...')

        annType = ['segm', 'bbox', 'keypoints']
        a_infer_time = 1000*inference_time / (n_samples)
        a_nms_time= 1000*nms_time / (n_samples)

        print('Average forward time: %.2f ms, Average NMS time: %.2f ms, Average inference time: %.2f ms' %(a_infer_time, \
                a_nms_time, (a_infer_time+a_nms_time)))

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('yolov3_2017.json', 'w'))
                cocoDt = cocoGt.loadRes('yolov3_2017.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return cocoEval.stats[0], cocoEval.stats[1]
        else:
            return 0, 0

