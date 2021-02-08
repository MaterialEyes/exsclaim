## Acquired from https://github.com/pytorch/vision/tree/master/references/detection
import math
import sys
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from . import utils
from .coco_eval import CocoEvaluator
from . import process
import pathlib

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, lr_scheduler=None, model_name="unnamed_model"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", model_name=model_name)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    #if epoch == -5:
    #    warmup_factor = 1. / 1000
    #    warmup_iters = min(1000, len(data_loader) - 1)

    #    lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(loss_value)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def run_nms_on_outputs(outputs):
    nms_outputs = []
    for image_outputs in outputs:
        scale_bar_info = []
        for i, box in enumerate(image_outputs["boxes"]):
            confidence = image_outputs["scores"][i]
            if True:
                #print("confidence is over 0.5")
                x1, y1, x2, y2 = box
                label = image_outputs['labels'][i]
                scale_bar_info.append([x1, y1, x2, y2, confidence, label])
        scale_bar_info = process.non_max_suppression_malisiewicz(np.asarray(scale_bar_info), 0.4)
        boxes = torch.empty((0, 4))
        labels = []
        scores = []
        boxes_temp = []
        for scale_object in scale_bar_info:
            x1, y1, x2, y2, confidence, label = scale_object
            boxes_temp.append([x1, y1, x2, y2])
            labels.append(label)
            scores.append(confidence)
        boxes_temp = torch.tensor(boxes_temp)
        boxes = torch.cat((boxes, boxes_temp))
        image_dict = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores)
        }
        nms_outputs.append(image_dict)
    return nms_outputs

@torch.no_grad()
def evaluate(model, data_loader, device, model_name="unnamed_model"):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", model_name=model_name)
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    nms_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(cpu_device) for img in images)

        #torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images) 
        nms_outputs = run_nms_on_outputs(outputs)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        nms_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in nms_outputs]
        
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        nms_res = {target["image_id"].item(): output for target, output in zip(targets, nms_outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        nms_evaluator.update(nms_res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    current_file = pathlib.Path(__file__).resolve(strict=True)
    save_file = current_file.parent / 'results' / '{}.txt'.format(model_name)
    with open(save_file, "a") as f:
        f.write("Averaged stats:{}\n".format(metric_logger))

    coco_evaluator.synchronize_between_processes()
    nms_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize(model_name=model_name)
    torch.set_num_threads(n_threads)
    nms_evaluator.accumulate()
    nms_evaluator.summarize(model_name=model_name, nms=True)

    return coco_evaluator
