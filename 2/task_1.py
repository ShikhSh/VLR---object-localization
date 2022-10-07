import argparse
from math import gamma
import os
from sched import scheduler
import shutil
import time

import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') # binary cross entropy 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) # LR schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset('trainval', image_size=512) # creating the dataset instance
    val_dataset = VOCDataset('test', image_size=512)
    # train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True,
        collate_fn=VOCDataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=VOCDataset.collate_fn)

    if USE_WANDB == True:
        wandb.init(project="vlr-hw1")

    # plot_boxes([100,200,300,400,225,325,425], val_dataset, model, 28, "val")
    # blah

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

            plot_boxes([0,1], train_dataset, model, epoch)
            plot_boxes([0,1], val_dataset, model, epoch, "val")

        scheduler.step()


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    """
    - Method that runs a pass over the training dataset, compute the mAP and recall metrics on the fly.
    - Compute the training loss and do a backward pass to update the params to minimize the loss. 
    - Push training loss, mAP, recall, step to wandb
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        input = data['image'].cuda()
        target = torch.as_tensor(data['label']).cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model(input)

        # TODO (Q1.1): Perform any necessary operations on the output
        imoutput = torch.squeeze(imoutput) # (bs, num_classes)
        # TODO (Q1.1): Compute loss using ``criterion``
        # print(f'target {target.shape} imoutput {imoutput.shape}')
        loss = criterion(imoutput, target).sum()

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target.data, data['wgt'])
        m2 = metric2(imoutput.data, target.data)

        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)
        # TODO (Q1.1): compute gradient and perform optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if USE_WANDB:
            # push the relevant metrics to wandB
            wandb.log({
                "train/loss": losses.avg,
                "train/mAP": avg_m1.avg,
                "train/recall": avg_m2.avg,
                "train/step": epoch*len(train_loader) + i,
            })

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals


def validate(val_loader, model, criterion, epoch=0):
    """
    - Method that runs a pass over the val dataset, compute the mAP and recall metrics on the fly.
    - Compute the val loss. 
    - Push val loss, mAP, recall, step to wandb
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = torch.as_tensor(data['label']).cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model(data['image'].cuda())

        # TODO (Q1.1): Perform any necessary functions on the output
        # imoutput = nn.AdaptiveMaxPool2d((1,1))(imoutput)
        imoutput = torch.squeeze(imoutput)
        # print(f'imoutput {imoutput.shape} target {target.shape}')
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(imoutput, target).sum()

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target.data, data['wgt'])
        m2 = metric2(imoutput.data, target.data)

        losses.update(loss.item(), data['image'].size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if USE_WANDB:
            wandb.log({
                "val/loss": losses.avg,
                "val/mAP": avg_m1.avg,
                "val/recall": avg_m2.avg,
                "val/step": epoch*len(val_loader) + i,
            })

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize things as mentioned in handout
        # TODO (Q1.3): Visualize at appropriate intervals
        # check plot_boxes method
    
    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg

def plot_boxes(indices, dataset, model, epoch, split="train"):
    # TODO (Q1.3): Visualize things as mentioned in handout
    """
    Args:
    - indices: indices to be visualized
    - dataset: dataset instance
    - model: model instance
    - epoch: current epoch
    Method that visualizes the feature map plots for a given data split. 

    """
    data = [dataset.__getitem__(idx) for idx in indices]
    datapoints = VOCDataset.collate_fn(data)
    feature_maps = model(datapoints['image'], box_plots=True)
    for i in range(len(indices)):
        gt_label = data[i]['gt_classes'][0]
        orig_img = data[i]['image']
        feature_map = feature_maps[i, gt_label,:,:]
        print(f'feature_map {feature_map.shape} orig_img {orig_img.shape[1:]} gt_label {gt_label}')
        feature_map = feature_map.unsqueeze(0).unsqueeze(0)
        feature_map = F.interpolate(feature_map, size=orig_img.shape[1:], mode="bilinear").squeeze() # perform bilinear interpolation of the feature map and resize it to input image size
        # feature_map = (feature_map-torch.min(feature_map))/(torch.max(feature_map) - torch.min(feature_map))
        feature_map = torch.sigmoid(feature_map)
        feature_map = feature_map.squeeze().detach().cpu().numpy()
        cmap = plt.get_cmap('jet')
        feature_map = cmap(feature_map)
        img = wandb.Image(feature_map)
        orig_img = wandb.Image(tensor_to_PIL(orig_img))
        
        wandb.log({f"{split}/Feature_map_{indices[i]}": img, "epoch": epoch, f"{split}/orig_img_{indices[i]}": orig_img})

    return 0

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


# def metric1(output, target):
#     # TODO (Q1.5): compute metric1
#     # print(f'output {output.shape} target {target.shape}')
#     output = torch.sigmoid(output).cpu().numpy()
#     target = target.cpu().numpy()
#     # print(target)
#     # print(output)
#     return compute_ap(gt=target, pred=output)


def metric2(output, target, threshold=0.5):
    # TODO (Q1.5): compute metric2
    output = torch.sigmoid(output)
    output = torch.where(output>threshold, 1, 0)
    # s = torch.where(torch.sum(target, dim=0)!=0)[0]
    # output = output[:,s]
    # target = target[:,s]
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    return sklearn.metrics.recall_score(y_true=target.astype('float32'), y_pred=output.astype('float32'), average="samples", zero_division=0)

def metric1(output, target, wts):
    # TODO (Q1.5): compute metric1
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    # print(f"{output=}")
    # print(f"{target=}")
    wts = torch.stack(wts, dim=0).mean(dim=1).cpu().numpy()
    mAP = sklearn.metrics.average_precision_score(target, output, average="samples", sample_weight=wts)
    # print(mAP)
    return mAP


# def metric2(output, target):
#     # TODO (Q1.5): compute metric2
#     # assumption taken: threshold = 0.5
#     output = torch.sigmoid(output).cpu().numpy()
#     output[output<0.2] = 0.
#     output[output!=0] = 1.
#     target = target.cpu().numpy()
#     recall = sklearn.metrics.recall_score(target, output, average="macro", zero_division=0)
#     return recall

if __name__ == '__main__':
    main()
