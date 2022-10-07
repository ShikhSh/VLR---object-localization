from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import torchvision.models as models
import pickle as pkl

from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, iou, tensor_to_PIL
from PIL import Image, ImageDraw
import sklearn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")

data_directory = '../VOCdevkit/VOC2007/'
USE_WANDB = True
# hyper-parameters
# ------------
parser = argparse.ArgumentParser(help='PyTorch ImageNet Training')
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
    default=True,
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

def freeze_alexnet_weigths(model):
    # Freezing the model for the convolution layers:
    for i in [0,3,6,8,10]:
        model.features[i].weight.requires_grad = False

def set_up_wandb():
    if USE_WANDB:
        wandb.login(key="f123ce836f30a91233b673ad557cf57dfe08ef9d")
        run = wandb.init(
            name = "vlr_hw1_trial",
            reinit=True,
            project="vlr_hw1"
        )

def load_pretained_weights(model):
    alex_net_pretrained = models.alexnet(pretrained = True)
    for i in [0, 3, 6, 8, 10]:
        model.features[i].load_state_dict(alex_net_pretrained.features[i].state_dict())
    return model

def calculate_map(track_tp, track_fp, n_class_gt):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    
    # for each class:
    # extract gt_boxes for that class using gt_class
    # start with max score output bounding box
    # find corresponding gt box with highest iou score (match the sequence of x's and y's for bb and gt_boxes)
    #   remove that gt box, since it cannot be used any more
    #   if iou score > threshold, there is a great overlap between the 2
    #       tp++
    #   else
    #       fp++
    # continue iterating over the other bb for the same class
    # thus for each class, we get one tp and one fp score
    # get the tp and fp scores for all the classes
    # calculate the precision and recall
    # now calc area under precision-recall curve using sklearn
    # this is ap(Avg Precision), do that for all, get map
    track_tp, track_fp, n_class_gt = np.array(track_tp), np.array(track_fp), np.array(n_class_gt)
    recall = 1.0*track_tp/n_class_gt
    precision = 1.0*track_tp/(track_tp+track_fp)
    map = sklearn.metrics.auc(recall, precision)
    return map




def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    with torch.no_grad():
        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            image = image.to(device)
            target = target.to(device)
            # wgt = wgt.to(device)
            rois = rois.to(device)

            # TODO (Q2.3): perform forward pass, compute cls_probs
            imoutput = model(image, rois, target)

            # TODO (Q2.3): Iterate over each class (follow comments)
            # for each class
            # extract the bounding boxes for that image from highest scoring to lowest scoring
            # match the ones
            class_aps = []
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                tp = 0
                fp = 0
                track_tp = []
                track_fp = []
                class_gt_indices = torch.where(gt_class_list == class_num)
                class_gt_boxes = gt_boxes[class_gt_indices]
                n_class_gt = len(class_gt_boxes)

                # use NMS to get boxes and scores
                boxes, scores = nms(rois, imoutput[:, class_num])
                if len(boxes) == 0:
                    # we need not keep a count of false negatives otherwise this would have come here
                    class_aps.append(0)
                    continue
                
                if len(class_gt_boxes) == 0:
                    # there are no gt boxes for this, thus we need not do anything about it
                    # fp += len(boxes)
                    class_aps.append(0)
                    continue

                # now calculate the iou for all the boxes and 
                iou_values = iou(boxes, class_gt_boxes)
                
                for idx in range(len(boxes)):
                    # find the best gt_box for an iou
                    max_ios_pos = iou_values[idx].argmax()
                    # check if that value is greater than the threshold
                    if iou_values[idx, max_ios_pos] >= thresh:
                        iou_values[:, max_ios_pos] = -1 #since it should not be used again
                        tp+=1
                    else:
                        fp+=1
                    track_tp.append(tp)
                    track_fp.append(fp)
                
                # TODO (Q2.3): visualize bounding box predictions when required
                map_ = calculate_map(track_tp, track_fp, n_class_gt)
                class_aps.append(map_)

                print("MAP: ", str(map_))


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            
            # TODO Convert inputs to cuda if training on GPU
            image = image.to(device)
            target = target.to(device)
            # wgt = wgt.to(device)
            rois = rois.to(device)
            # gt_boxes = gt_boxes.to(device)
            # gt_class_list = gt_class_list.to(device)

            # take care that proposal values should be in pixels
            # multiply image size
            image_size = image.shape[0]
            rois = rois*image_size

            # TODO (Q2.2): perform forward pass
            imoutput = model(image, rois, target)

            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout

    # TODO (Q2.4): Plot class-wise APs


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    set_up_wandb()
    train_dataset = VOCDataset(split='trainval',image_size = 512 , data_dir=data_directory)
    val_dataset = VOCDataset(split='test',image_size = 512 , data_dir=data_directory)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize with AlexNet weights
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    net = load_pretained_weights(net)
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
    # do it in state dict
    freeze_alexnet_weigths(net)
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    params = list(net.classifier.parameters()) + list(net.score_fc.parameters()) + list(net.bbox_fc.parameters())
    optimizer = torch.optim.SGD(params, lr = args.lr, momentum = args.momentum, weight_decay = args.weightDecay, nesterov = True)

    # Training
    train_model(net, train_loader, val_loader, optimizer, args)

if __name__ == '__main__':
    main()


# Caveats
# for MAP -> pick up GT for rois vs ROIS for GT
