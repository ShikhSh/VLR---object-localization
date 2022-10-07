from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from symbol import parameters
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn2 import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou, get_box_data
from PIL import Image, ImageDraw
import time
from tqdm import tqdm
from sklearn.metrics import auc
import torchvision
# import mapcalc
from collections import defaultdict

img_size = 512
# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    help='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    help='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float,
    help='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    help='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int,
    help='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    help='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    help='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_map(num_gt_boxes, boxes_match_score):
    """
    Calculate the mAP for classification.

    Args:
    - num_gt_boxes: Number of gt boxes per class
    - boxes_match_score: (list containing the whether there is a matching
    gt box with iou>=0.5 for a given prediction or not, required to determine if a given prediction is a TP or FP)

    Steps for computing AP
    1. For each class
        - check if there are any gt boxes belonging to that class/ any predicted box (with score>thresh)
        - Sort the boxes based on the prediction score (boxes with iou >= 0.5 and class score >=0.05), done so that we could compute precision and recall
        - Computing area under precison vs recall curve i.e. append 0.0 in the precision and recall lists. Iterate over all the boxes and determine if its a TP (matched a GT box) 
        or FP (didnt match a gt box)
        - at the end, append 0 precision and 1 recall, to account for the other end of precision-recall curve
        - compute AP for each class = Sum((recall[i] - recalls[i-1])*precisions[i])

    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    AP = [0 for i in range(20)]

    for class_num in range(20):
        # check if there are any gt boxes belonging to that class/ any predicted box (with score>thresh)
        if len(boxes_match_score[class_num]) == 0 or num_gt_boxes[class_num] == 0:
            AP[class_num] = 0
            continue
        
        curr_scores = boxes_match_score[class_num] # boxes with iou >= 0.5 and class score >=0.05
        curr_scores.sort(key = lambda x: x['score'], reverse= True) # Sort the boxes based on the prediction score

        tp, fp = 0.0, 0.0 
        precisions = [0.0]
        recalls= [0.0]
        for idx in range(len(curr_scores)):
            match = curr_scores[idx]['match']
            # Computing area under precison vs recall curve i.e. append 0.0 in the precision and recall lists. Iterate over all the boxes and determine if its a TP (matched a GT box) or FP (didnt match a gt box)
            if match:
                tp += 1.0
            else:
                fp += 1.0
            precision = tp/(tp + fp)
            recall = tp/num_gt_boxes[class_num]
            precisions.append(precision)
            recalls.append(recall)
        precisions.append(0.0)
        recalls.append(1.0) # at the end, append 0 precision and 1 recall, to account for the other end of precision-recall curve
        ap = sum([(recalls[i] - recalls[i-1])*np.max(precisions[i:]) for i in range(1, len(precisions))]) # compute AP for the class

        AP[class_num] = ap

        # print(f'precision {precisions} recalls {recalls}')
    print(f'AP {AP}')
    return np.array(AP)


def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold

    Steps followed:
    1. For each data point 
        - do a fwd pass to determine the class scores for each roi.
        - for all classes
            - Determine the number of gt boxes for the given data point belonging to a particular class [required for computing recall]
            - perform NMS on rois (only on the boxes with pred score for the given class > thresh)
            - for all the boxes obtained determine if there is a matching gt (iou>=0.5) or not
    2. Compute AP: using num_gt_boxes (keeps track of number of gt boxes for a given class) for each class and boxes_match_score (list containing the whether there is a matching
    gt box with iou>=0.5 for a given prediction or not, required to determine if a given prediction is a TP or FP)


    """
    num_data_points = 0
    num_steps = len(val_loader)
    progress_bar = tqdm(range(num_steps))
    boxes_match_score = [[] for i in range(20)]
    num_gt_boxes = [0 for i in range(20)]

    with torch.no_grad():
        for iter, data in enumerate(val_loader):
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = torch.stack([torch.as_tensor(x) for x in data['rois']], dim=0)
            gt_boxes = torch.stack([torch.as_tensor(x) for x in data['gt_boxes']], dim=0).squeeze(dim=0)
            gt_class_list = torch.stack([torch.as_tensor(x) for x in data['gt_classes']], dim=0).squeeze(dim=0)

            # TODO (Q2.3): perform forward pass, compute cls_probs
            cls_scores = model(image.cuda(), rois*img_size, target.cuda())


            # TODO (Q2.3): Iterate over each class (follow comments)
            for class_num in range(20):
                # get valid rois and cls_scores
                scores = cls_scores[:, class_num]
                boxes = rois.squeeze()[:,:]

                # finding the number of gt boxes for the current class (useful for computing recall)
                curr_gt_boxes = None
                if target[0, class_num] == 1:
                    # print(f'gt_class_list {gt_class_list} {torch.where(gt_class_list == class_num)[0]} {class_num}')
                    curr_gt_boxes = gt_boxes[torch.where(gt_class_list == class_num)[0], :]
                    num_gt_boxes[class_num] += curr_gt_boxes.shape[0] 
                
                # perform NMS on boxes and scores, boxes are reverse sorted based on the scores [To get rid of predicted boxes with iou > threshold]
                boxes, scores = nms(boxes, scores, threshold=thresh)
                if len(boxes) == 0:
                    continue

                boxes = torch.stack(boxes, dim=0)
                scores = torch.stack(scores, dim=0)

                # get the iou for the predictions with gt
                ious = None
                if curr_gt_boxes != None:
                    ious = iou(boxes, curr_gt_boxes)

                # determine for all the rois if a given roi has a matching gt box or not (required to determine if a given prediction is a TP or FP)
                for idx in range(len(boxes)):
                    match_found = False
                    if ious is not None:
                        iou_idx_max = ious[idx].argmax()
                        if ious[idx, iou_idx_max] >= 0.5:
                            # found some gt box that matched the prediction, mark it as used since you cannot use it again
                            ious[:, iou_idx_max] = 0
                            match_found = True
                    boxes_match_score[class_num].append({'match':match_found, 'score':scores[idx].item()})
                


            # TODO (Q2.3): visualize bounding box predictions when required
                progress_bar.update(1)
    
    AP = calculate_map(num_gt_boxes, boxes_match_score)

    return AP


def box_plots(val_dataset, indices, model, epoch, thresh = 0.05):
    class_id_to_label = dict(enumerate(model.classes))
    for idx in indices:
        data = VOCDataset.collate_fn([val_dataset[idx]])

        image = data['image']
        target = data['label']
        wgt = data['wgt']
        rois = torch.stack([torch.as_tensor(x) for x in data['rois']], dim=0)
        gt_boxes = torch.stack([torch.as_tensor(x) for x in data['gt_boxes']], dim=0)
        gt_class_list = torch.stack([torch.as_tensor(x) for x in data['gt_classes']], dim=0)

        cls_scores = model(image.cuda(), rois*img_size, target.cuda())

        predicted_boxes = []
        predicted_classes = []

        for class_num in range(20):
            # get valid rois and cls_scores
            scores = cls_scores[:, class_num]
            boxes = rois.squeeze()[:,:]

            boxes, scores = nms(boxes, scores, threshold=thresh)
            if len(boxes) == 0:
                continue

            boxes = torch.stack(boxes, dim=0)
            scores = torch.stack(scores, dim=0)

            for i in range(0, len(boxes)):
                predicted_boxes.append(boxes[i].cpu().numpy().tolist())
                predicted_classes.append(class_num)
            
        if len(predicted_boxes)==0:
            img = wandb.Image(tensor_to_PIL(image.squeeze(0)))
            logs = {"epoch":epoch, f"val/bbox_{idx}": img}
            wandb.log(logs)
        else:
            print(f'image {image.shape} predicted_boxes {predicted_boxes} predicted_classes {predicted_classes}')
            img = wandb.Image(tensor_to_PIL(image.squeeze(0)), boxes={
                "predictions": {
                    "box_data": get_box_data(predicted_classes, predicted_boxes),
                    "class_labels": class_id_to_label,
                },
            })
            wandb.log({f"val/bbox_{idx}": img, "epoch": epoch})

        

def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None, scheduler=None, val_dataset=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    
    for epoch in range(args.epochs):
        # box_plots(val_dataset=val_dataset, indices=[0,1,2,3], model=model, epoch=epoch)
        num_steps = len(train_loader)
        progress_bar = tqdm(range(num_steps))
        losses = AverageMeter()
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = torch.stack([torch.as_tensor(x) for x in data['rois']], dim=0)
            gt_boxes = torch.stack([torch.as_tensor(x) for x in data['gt_boxes']], dim=0)
            gt_class_list = torch.stack([torch.as_tensor(x) for x in data['gt_classes']], dim=0)

            # print(f'wgt {wgt}')
            # blah
            

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            # print(f'rois {rois.shape}')
            out = model(image.cuda(), rois*img_size, target.cuda())

            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1
            losses.update(loss.item(), image.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if step_cnt % 500 == 0 and step_cnt !=0:
                if args.use_wandb:
                    logs = {'iter': step_cnt, 'train/loss': losses.avg, 'train/Loss': train_loss/step_cnt}
                    wandb.log(logs)


            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", np.mean(ap))
                model.train()

                if args.use_wandb:
                    logs = {'iter': step_cnt, 'val/mAP': np.mean(ap)}
                    for i, class_ in enumerate(model.classes):
                        logs[f'val/aP_{class_}'] = ap[i]
                    wandb.log(logs)

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            progress_bar.set_postfix({'train/loss': train_loss/step_cnt})
            progress_bar.update(1)
            scheduler.step()
    # TODO (Q2.4): Plot class-wise APs
        # generating plots vs epoch
        if args.use_wandb:
            ap = test_model(model, val_loader)
            logs = {'epoch': epoch, 'epoch/val_mAP': np.mean(ap), 'epoch/train_loss': train_loss/step_cnt, 'epoch/Train_Loss': losses.avg}
            for i, class_ in enumerate(model.classes):
                logs[f'epoch/val/aP/{class_}'] = ap[i]
            my_table = wandb.Table(columns=["class", "AP"], data=[[cl, ap] for (cl, ap) in zip(list(model.classes), ap)])
            logs['classApScores'] = my_table
            wandb.log(logs)
            print(f'Class_wise AP {ap}')


        with open(os.path.join('task_2/wts', f'model_{epoch}.pth'), 'wb') as f:
            torch.save(model.state_dict(), f)

        if args.use_wandb:
            box_plots(val_dataset=val_dataset, indices=[0,1,2,3], model=model, epoch=epoch)


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    data_directory = '../../VOCdevkit/VOC2007/'
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    if args.use_wandb:
        wandb.init(project="vlr-hw1")
    train_dataset = VOCDataset('trainval', image_size=img_size,data_dir=data_directory)
    val_dataset = VOCDataset('test', image_size=img_size,data_dir=data_directory)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=VOCDataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=VOCDataset.collate_fn)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in net.features.parameters():
        param.requires_grad = False

    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_decay_steps], gamma=args.lr_decay)
    # Training
    train_model(net, train_loader, val_loader, optimizer, args, scheduler, val_dataset)

    # with open(os.path.join('task_2/wts', 'model.pth'), 'wb') as f:
    #     torch.save(net.state_dict(), f)
    

if __name__ == '__main__':
    main()