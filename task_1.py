import argparse
import os
import shutil
import time
from tkinter import image_names

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

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")

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
    '--weightDecay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=1,
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

data_directory = '../VOCdevkit/VOC2007/'

def set_up_wandb():
    if USE_WANDB:
        wandb.login(key="f123ce836f30a91233b673ad557cf57dfe08ef9d")
        run = wandb.init(
            name = "vlr_hw1_trial", ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit=True, ### Allows reinitalizing runs when you re-run this cell
            project="vlr_hw1"#, ### Project should be created in your wandb account 
        #     config=config ### Wandb Config for your run
        )

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    set_up_wandb()

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
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weightDecay, nesterov = True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)


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

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset(split='trainval',image_size = 512 , data_dir=data_directory)
    val_dataset = VOCDataset(split='test',image_size = 512 , data_dir=data_directory)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn = collate_fn,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn = collate_fn,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('Epoch')
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


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
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
        # print(data[0][0])
        # print(data[1][0])
        # print(data[2][0])
        # print(data[3][0])
        # print(data[4][0])
        # print(data[5][0])
        images = data[0]
        target = data[1]
        
        data_time.update(time.time() - end)
        images = images.to(device)
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = target.to(device)

        # TODO (Q1.1): Get output from model
        imoutput = model(images)

        # TODO (Q1.1): Perform any necessary operations on the output
        # max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=2)
        # print(imoutput.shape)
        # F.max_pool2d(imoutput, kernel_size = (29,29))
        imoutput = torch.max(imoutput, dim = 2)[0].max(2)[0]

        # TODO (Q1.1): Compute loss using ``criterion``
        print("SHAPES::::::::")
        print(imoutput.shape)
        print(target.shape)
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        # m1 = metric1(imoutput.data, target)
        # m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), images.size(0))
        # avg_m1.update(m1)
        # avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        wandb.log({'epoch': epoch, 'train/loss': loss})
        print("EPOCH:", epoch, " Loss:", loss)
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
        #           'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
        #               epoch,
        #               i,
        #               len(train_loader),
        #               batch_time=batch_time,
        #               data_time=data_time,
        #               loss=losses,
        #               avg_m1=0,#avg_m1,
        #               avg_m2=0#avg_m2
        #               ))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals

        # End of train()


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):
        # print(data)
        images = data[0]
        target = data[1]
        images = images.to(device)
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = target.to(device)

        # TODO (Q1.1): Get output from model
        imoutput = model(images)

        # TODO (Q1.1): Perform any necessary functions on the output
        imoutput = torch.sum(imoutput, axis = 1)
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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


    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


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


def metric1(output, target):
    # TODO (Q1.5): compute metric1

    return [0]


def metric2(output, target):
    # TODO (Q1.5): compute metric2

    return [0]


if __name__ == '__main__':
    main()