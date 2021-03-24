__version__ = '0.3.17'

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import time
import random
import pprint
import visdom
import logging
import datetime
import numpy as np
from easydict import EasyDict as edict

import matplotlib
import matplotlib.pyplot as plt

from datasets.base_dataset import ImageDataset
from utils.util import cfg, AverageMeter, accuracy
from modules.loggers.base_logger import get_logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

cudnn.benchmark = True
timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
opt.MODEL.PRETRAINED = True
opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224  # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = '2B'
opt.EXPERIMENT.TASK = 'finetune'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(
    opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TRAIN = edict()
opt.TRAIN.BATCH_SIZE = 32
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 8
opt.TRAIN.PRINT_FREQ = 20
opt.TRAIN.SEED = None
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.LR_GAMMA = 0.5
opt.TRAIN.LR_MILESTONES = [5, 7, 9, 10, 11, 12]
opt.TRAIN.EPOCHS = 12
opt.TRAIN.VAL_SUFFIX = '7c'
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.RESUME = None

opt.DATASET = 'recognition'

opt.VISDOM = edict()
opt.VISDOM.PORT = 8097
opt.VISDOM.ENV = '[' + opt.DATASET + ']' + opt.EXPERIMENT.CODENAME


if opt.TRAIN.SEED is None:
    opt.TRAIN.SEED = int(time.time())
msg = 'Use time as random seed: {}'.format(opt.TRAIN.SEED)
print(msg)
# logger.info(msg)
transforms.__package__
random.seed(opt.TRAIN.SEED)
torch.manual_seed(opt.TRAIN.SEED)
torch.cuda.manual_seed(opt.TRAIN.SEED)

if not osp.exists(opt.EXPERIMENT.DIR):
    os.makedirs(opt.EXPERIMENT.DIR)


logger = get_logger(opt.LOG.LOG_FILE)
logger.info('\n\nOptions:')
logger.info(pprint.pformat(opt))

DATA_INFO = cfg.DATASETS[opt.DATASET.upper()]

# Data-loader of training set
transform_train = transforms.Compose([
    transforms.Resize((opt.MODEL.IMAGE_SIZE)),  # Smaller edge
    transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Data-loader of testing set
transform_val = transforms.Compose([
    transforms.Resize((opt.MODEL.IMAGE_SIZE)),
    transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageDataset(DATA_INFO.TRAIN_DIR, transform_train)
val_dataset = ImageDataset(DATA_INFO.TRAIN_DIR, transform_val)

assert(len(train_dataset.classes) == DATA_INFO.NUM_CLASSES)
logger.info('{} images are found for train_val'.format(
    len(train_dataset.imgs)))

train_imgs = [(img, target) for (img, target)
              in train_dataset.imgs if not img[-5] in opt.TRAIN.VAL_SUFFIX]
logger.info('{} images are used to train'.format(len(train_imgs)))
val_imgs = [(img, target) for (img, target)
            in train_dataset.imgs if img[-5] in opt.TRAIN.VAL_SUFFIX]
logger.info('{} images are used to val'.format(len(val_imgs)))


train_dataset.samples = train_dataset.imgs = train_imgs
val_dataset.samples = val_dataset.imgs = val_imgs


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=opt.TRAIN.SHUFFLE, num_workers=opt.TRAIN.WORKERS)

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

# create model
if opt.MODEL.PRETRAINED:
    logger.info("=> using pre-trained model '{}'".format(opt.MODEL.ARCH))
    model = models.__dict__[opt.MODEL.ARCH](pretrained=True)
else:
    raise NotImplementedError
#    logger.info("=> creating model '{}'".format(args.arch))
#    model = models.__dict__[opt.MODEL.ARCH]()

if opt.MODEL.ARCH.startswith('resnet'):
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    #model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
else:
    raise NotImplementedError
    model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.module.parameters(), opt.TRAIN.LEARNING_RATE)
lr_scheduler = MultiStepLR(
    optimizer, opt.TRAIN.LR_MILESTONES, gamma=opt.TRAIN.LR_GAMMA, last_epoch=-1)

if opt.TRAIN.RESUME is None:
    last_epoch = 0
    logger.info("Training will start from Epoch {}".format(last_epoch + 1))

else:
    last_checkpoint = torch.load(opt.TRAIN.RESUME)
    assert(last_checkpoint['arch'] == opt.MODEL.ARCH)
    model.module.load_state_dict(last_checkpoint['state_dict'])
    optimizer.load_state_dict(last_checkpoint['optimizer'])
    logger.info("Checkpoint '{}' was loaded.".format(opt.TRAIN.RESUME))

    last_epoch = last_checkpoint['epoch']
    logger.info("Training will be resumed from Epoch {}".format(
        last_checkpoint['epoch']))

vis = visdom.Visdom(port=opt.VISDOM.PORT)
vis.close()
vis.text('HELLO', win=0, env=opt.VISDOM.ENV)

train_losses = []
train_top1s = []
test_losses = []
test_top1s = []


def visualize():
    X = np.array(range(len(train_losses))) + 1 + last_epoch
    vis.line(
        X=np.column_stack((X, X)),
        Y=np.column_stack((train_losses, test_losses)),
        win=1,
        env=opt.VISDOM.ENV,
        opts={
            'title': 'loss over time',
            'xlabel': 'epoch',
            'ylabel': 'loss',
            'legend': ['train', 'test']
        }
    )

    vis.line(
        X=np.column_stack((X, X)),
        Y=np.column_stack((train_top1s, test_top1s)),
        win=2,
        env=opt.VISDOM.ENV,
        opts={
            'title': 'accuracy over time',
            'xlabel': 'epoch',
            'ylabel': 'accuracy (%)',
            'legend': ['train', 'test']
        }
    )


def save_checkpoint(state, filename='checkpoint.pk'):
    torch.save(state, osp.join(opt.EXPERIMENT.DIR, filename))
    logger.info('A snapshot was saved to {}.'.format(filename))


def train(train_loader, model, criterion, optimizer, epoch):
    logger.info('Epoch {}'.format(epoch))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info('[{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5))

    train_losses.append(losses.avg)
    train_top1s.append(top1.avg)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    test_losses.append(losses.avg)
    test_top1s.append(top1.avg)

    return top1.avg


criterion = nn.CrossEntropyLoss()
best_prec1 = 0
best_epoch = 0

for epoch in range(last_epoch + 1, opt.TRAIN.EPOCHS + 1):
    logger.info('-' * 50)
    lr_scheduler.step(epoch)
    logger.info('lr: {}'.format(lr_scheduler.get_lr()))
    train(train_loader, model, criterion, optimizer, epoch)
    prec1 = validate(test_loader, model, criterion)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        best_epoch = epoch

    if epoch % opt.TRAIN.SAVE_FREQ == 0:
        save_checkpoint({
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'prec1': prec1,
            'optimizer': optimizer.state_dict(),
        }, '{}_[{}]_{:.02f}.pk'.format(opt.MODEL.ARCH, epoch, prec1))

    if is_best:
        save_checkpoint({
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'prec1': prec1,
            'optimizer': optimizer.state_dict(),
        }, 'best_model.pk')

        vis.text('Best accuracy: {} (Epoch {})'.format(
            prec1, epoch), win=0, env=opt.VISDOM.ENV)
    visualize()


logger.info('Best accuracy for single crop: {:.02f}%'.format(best_prec1))
#best_checkpoint_path = osp.join(opt.EXPERIMENT.DIR, 'best_model.pk')
#logger.info("Loading parameters from the best checkpoint '{}',".format(best_checkpoint_path))
#checkpoint = torch.load(best_checkpoint_path)
#logger.info("which has a single crop accuracy {:.02f}%.".format(checkpoint['prec1']))
# model.load_state_dict(checkpoint['state_dict'])

best_epoch = np.argmin(test_losses)
best_loss = test_losses[best_epoch]
plt.figure(0)
x = np.arange(last_epoch + 1, opt.TRAIN.EPOCHS + 1)
plt.plot(x, train_losses, '-+')
plt.plot(x, test_losses, '-+')
plt.scatter(best_epoch + 1, best_loss, c='C1', marker='^', s=80)
plt.ylim(ymin=0, ymax=5)
plt.grid(linestyle=':')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Loss over epoch')
plt.savefig(osp.join(opt.EXPERIMENT.DIR, 'loss_curves.png'))


best_epoch = np.argmax(test_top1s)
best_top1 = test_top1s[best_epoch]
plt.figure(1)
plt.plot(x, train_top1s, '-+')
plt.plot(x, test_top1s, '-+')
plt.scatter(best_epoch + 1, best_top1, c='C1', marker='^', s=80)
plt.ylim(ymin=0, ymax=100)
plt.grid(linestyle=':')
plt.xlabel('epoch')
plt.ylabel('accuracy (%)')
plt.title('accuracy over epoch')
plt.savefig(osp.join(opt.EXPERIMENT.DIR, 'accuracy_curves.png'))
