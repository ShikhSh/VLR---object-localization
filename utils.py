import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def nms(bounding_boxes, confidence_score, threshold=0.05, iou_threshold = 0.3):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    -pick up the best score bounding box
    -check if its score > threshold, if not then we can break since the best one is below threshold so rest one will be below it for sure
    -now that can be taken into the best ones to return
    -now calculate the iou score with the rest of the boxes
    -for the indices where the iou score < threshold, keep them
    -

    return: list of bounding boxes and scores
    """
    print(confidence_score.shape)
    conf_sc, indices = torch.sort(confidence_score, descending=True)
    print("PRINTING BOXESSS:::::::::::::::::::::")
    # print(bounding_boxes.shape)
    print(conf_sc)
    bounding_boxes = torch.squeeze(bounding_boxes)
    b_boxes = bounding_boxes[indices]
    boxes, scores = torch.Tensor([]), []

    while len(conf_sc)>0:
        best_bbox = torch.unsqueeze(b_boxes[0], dim = 0)
        best_score = conf_sc[0]
        # print(best_score)
        if best_score < threshold:
            # print("not me")
            # break
            continue
        print("I Survived")
        boxes = torch.cat((boxes, best_bbox))
        # boxes.append(best_bbox)
        scores.append(best_score)
        # print(best_bbox)
        # print(b_boxes)
        iou_score = iou(b_boxes, best_bbox)#torch.unsqueeze(iou(b_boxes, best_bbox))
        # print(iou_score)
        iou_score = torch.squeeze(iou_score, dim = 1)
        # print("after")
        # print(iou_score)
        indices_to_keep = torch.where(iou_score<iou_threshold)
        # print(indices_to_keep)
        # print("PRINTING BOXESSS:::::::::::::::::::::")
        # print(b_boxes.shape)
        b_boxes = b_boxes[indices_to_keep]
        conf_sc = conf_sc[indices_to_keep]


        # # the 0th element will be removed since the overlap will be greater than the threshold
        # # b_boxes = b_boxes[1:]
        # # conf_sc = conf_sc[1:]
        # i = 0
        # while i < len(conf_sc):
        #     current_bbox = b_boxes[i]
        #     # current_score = conf_sc[i]
        #     iou_score = iou(best_bbox, current_bbox)

        #     # need to remove the box since the overlap is higher than the threshold
        #     if iou_score > iou_threshold:
        #         b_boxes = torch.cat((b_boxes[0:i], b_boxes[i+1:len(b_boxes)]))
        #         conf_sc = torch.cat((conf_sc[0:i], conf_sc[i+1:len(conf_sc)]))

    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)

    assumes xmin, ymin, xmax, ymax format for each box

    returns IoU vallue
    """
    n1 = box1.shape[0]
    n2 = box2.shape[0]
    ones_matrix = torch.ones((n1,n2))

    x_min1 = torch.reshape(box1[:,0],(-1,1))*ones_matrix
    y_min1 = torch.reshape(box1[:,1],(-1,1))*ones_matrix
    x_max1 = torch.reshape(box1[:,2],(-1,1))*ones_matrix
    y_max1 = torch.reshape(box1[:,3],(-1,1))*ones_matrix
    # print(box1[:,0].shape)
    # print(x_min1.shape)

    x_min2 = ones_matrix*torch.reshape(box2[:,0],(1,-1))
    y_min2 = ones_matrix*torch.reshape(box2[:,1],(1,-1))
    x_max2 = ones_matrix*torch.reshape(box2[:,2],(1,-1))
    y_max2 = ones_matrix*torch.reshape(box2[:,3],(1,-1))
    # print(x_min2.shape)

    x_start = torch.maximum(x_min1,x_min2)
    x_end = torch.minimum(x_max1,x_max2)
    width = x_end - x_start
    width[width<0] = 0
    # print(width.shape)
    
    y_start = torch.maximum(y_min1,y_min2)
    y_end = torch.minimum(y_max1,y_max2)
    height = y_end - y_start
    height[height<0] = 0
    # print(height.shape)

    intersection_area = height*width
    # print("Intersection")
    # print(intersection_area.shape)

    area_b1 = (x_max1-x_min1)*(y_max1-y_min1)#((box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1]))*ones_matrix
    area_b2 = (x_max2-x_min2)*(y_max2-y_min2)#ones_matrix*torch.reshape(((box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])),(1,-1))

    union_area = area_b1+area_b2-intersection_area

    iou = 1.0*intersection_area/union_area
    # print(iou.shape)
    return iou


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

def get_unsupervised_box_data(bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)
    return list of boxes as expected by the wandb bbox plotter
    """
    # print(bbox_coordinates)
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
        } for i in range(len(bbox_coordinates))
        ]

    return box_list