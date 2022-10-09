import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn
import sklearn.metrics

# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    boxes, scores = [], []

    # remove bounding_boxes with scores less than threshold
    valid_indices = torch.where(confidence_score > threshold)[0]
    bounding_boxes = bounding_boxes[valid_indices]
    confidence_score = confidence_score[valid_indices]

    indices = confidence_score.argsort(descending=True)
    confidence_score = confidence_score[indices]
    bounding_boxes = bounding_boxes[indices]


    while len(bounding_boxes) > 0:
        # print("hereeeee")
        # select the first bounding box and filter the boxes out of the remaining boxes with iou > 0.3 with the given box
        boxes.append(bounding_boxes[0])
        scores.append(confidence_score[0])
        
        bounding_boxes = bounding_boxes[1:]
        confidence_score = confidence_score[1:]

        ious = iou(boxes[-1].unsqueeze(0), bounding_boxes)
        # print(f'ious {ious}')
        valid_indices = torch.where(ious<=0.3)[0]
        if len(valid_indices) == 0:
            break
        else:
            bounding_boxes = bounding_boxes[valid_indices,:]
            confidence_score = confidence_score[valid_indices]
    # if len(boxes)>0:
    #     print("PRINTING BOXESSS:::::::::::::::::::::", str(len(boxes)))
    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue

    box1: N1*4
    box2: N2*4

    compute the iou for all the pairs of gt and rois with significant confidence scores. iou = (intersection area)/(union area)
    """
    N1, N2 = box1.shape[0], box2.shape[0]
    
    # compute area (coord[2]-coord[0])*(coord[3]-coord[1])
    area_box1 = (box1[:,2] - box1[:,0])*(box1[:,3] - box1[:,1]) # (N1,)
    area_box2 = (box2[:,2] - box2[:,0])*(box2[:,3] - box2[:,1]) # (N2,)

    area_box1 = area_box1.unsqueeze(1) # (N1,1)
    area_box2 = area_box2.unsqueeze(0) # (1,N2)

    union_area = area_box1 + area_box2 # (N1,N2)

    box1 = torch.repeat_interleave(box1.unsqueeze(1), repeats=N2, dim = 1) # N1 x N2 x 4
    box2 = torch.repeat_interleave(box2.unsqueeze(0), repeats=N1, dim = 0) # N1 x N2 x 4

    # N1 x N2 [finding the top left and bottom right co-ordinates of the intersection]
    xmin = torch.max(box1[:,:,0], box2[:,:,0])
    ymin = torch.max(box1[:,:,1], box2[:,:,1])
    xmax = torch.min(box1[:,:,2], box2[:,:,2])
    ymax = torch.min(box1[:,:,3], box2[:,:,3])

    # intersection area
    # (min(box1[:,:,2], box2[:,:,2]) - max(box1[:,:,0], box2[:,:,0]))*(min(box1[:,:,3], box2[:,:,3]) - max(box1[:,:,1], box2[:,:,1]))
    width = xmax - xmin
    width[width < 0] = 0
    height = ymax - ymin
    height[height < 0] = 0 
    intersection_area = width*height
    union_area = union_area - intersection_area
    return intersection_area/union_area # N1 x N2


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list