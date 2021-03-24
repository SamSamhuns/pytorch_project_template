__version__ = '0.3.17'

import os
import os.path as osp

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#import argparse
import visdom
import logging
import numpy as np
import random
import time
import datetime
import pprint
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from world import cfg, create_logger, AverageMeter, accuracy


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# In[ ]:
cudnn.benchmark = True

# In[ ]:
timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
opt.MODEL.PRETRAINED = True
opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224  # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = '2B'
opt.EXPERIMENT.TASK = 'test'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(
    opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TEST = edict()
opt.TEST.CHECKPOINT = '/home/guang/Projects/Kaggle/landmark-recognition-challenge/experiments/2B/resnet50_[8]_96.07.pk'
opt.TEST.WORKERS = 8
opt.TEST.BATCH_SIZE = 32
opt.TEST.OUTPUT = osp.join(opt.EXPERIMENT.DIR, 'pred.npz')

opt.DATASET = 'recognition'

opt.VISDOM = edict()
opt.VISDOM.PORT = 8097
opt.VISDOM.ENV = '[' + opt.DATASET + ']' + opt.EXPERIMENT.CODENAME


# In[ ]:


if not osp.exists(opt.EXPERIMENT.DIR):
    os.makedirs(opt.EXPERIMENT.DIR)


# In[ ]:

logger = create_logger(opt.LOG.LOG_FILE)
logger.info('\n\nOptions:')
logger.info(pprint.pformat(opt))


# In[ ]:
DATA_INFO = cfg.DATASETS[opt.DATASET.upper()]

# Data-loader of testing set
transform_test = transforms.Compose([
    transforms.Resize((opt.MODEL.IMAGE_SIZE)),
    transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# In[ ]:

#train_dataset = datasets.ImageFolder(DATA_INFO.TRAIN_DIR, transform_test)
test_dataset = datasets.ImageFolder(DATA_INFO.TEST_DIR, transform_test)
logger.info('{} images are found for test'.format(len(test_dataset.imgs)))

test_list = pd.read_csv(osp.join(DATA_INFO.ROOT_DIR, 'test.csv'))
test_list = test_list['id']
logger.info('{} images are expected for test'.format(len(test_list)))
# In[ ]:

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.TEST.BATCH_SIZE, shuffle=False, num_workers=opt.TEST.WORKERS)

# In[ ]:

# create model
if opt.MODEL.PRETRAINED:
    logger.info("=> using pre-trained model '{}'".format(opt.MODEL.ARCH))
    model = models.__dict__[opt.MODEL.ARCH](pretrained=True)
else:
    raise NotImplementedError
#    logger.info("=> creating model '{}'".format(args.arch))
#    model = models.__dict__[opt.MODEL.ARCH]()

# In[ ]:

if opt.MODEL.ARCH.startswith('resnet'):
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    #model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
else:
    raise NotImplementedError
    model = torch.nn.DataParallel(model).cuda()


# In[ ]:

last_checkpoint = torch.load(opt.TEST.CHECKPOINT)
assert(last_checkpoint['arch'] == opt.MODEL.ARCH)
model.module.load_state_dict(last_checkpoint['state_dict'])
# optimizer.load_state_dict(last_checkpoint['optimizer'])
logger.info("Checkpoint '{}' was loaded.".format(opt.TEST.CHECKPOINT))

last_epoch = last_checkpoint['epoch']
#logger.info("Training will be resumed from Epoch {}".format(last_checkpoint['epoch']))


# In[ ]:

vis = visdom.Visdom(port=opt.VISDOM.PORT)
vis.close()
vis.text('HELLO', win=0, env=opt.VISDOM.ENV)

# In[ ]:

softmax = torch.nn.Softmax(dim=1).cuda()

pred_indices = []
pred_scores = []
pred_confs = []

model.eval()

for i, (input, target) in enumerate(tqdm(test_loader)):
    target = target.cuda(async=True)
    input_var = Variable(input, volatile=True)

    # compute output
    output = model(input_var)
    top_scores, top_indices = torch.topk(output, k=20)
    top_indices = top_indices.data.cpu().numpy()
    top_scores = top_scores.data.cpu().numpy()

    confs = softmax(output)
    top_confs, _ = torch.topk(confs, k=20)
    top_confs = top_confs.data.cpu().numpy()

    pred_indices.append(top_indices)
    pred_scores.append(top_scores)
    pred_confs.append(top_confs)

pred_indices = np.concatenate(pred_indices)
pred_scores = np.concatenate(pred_scores)
pred_confs = np.concatenate(pred_confs)

# In[ ]:
images = [osp.basename(image) for image, _ in test_dataset.imgs]

np.savez(opt.TEST.OUTPUT, pred_indices=pred_indices, pred_scores=pred_scores,
         pred_confs=pred_confs, images=images, checkpoint=opt.TEST.CHECKPOINT)
logger.info("Results were saved to '{}'.".format(opt.TEST.OUTPUT))
