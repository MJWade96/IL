#!/home/wangchuang/anaconda3/envs/pytorchgpupy3.7/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
#from torchnet.meter import AverageValueMeter
import torch.backends.cudnn as cudnn

from model import *

parser = {
    'data_dir': './selfdrivingcar-data/',
    'nb_epoch': 50,
    'test_size': 0.1,
    'learning_rate': 0.0001,
    'samples_per_epoch': 64,
    'batch_size': 36,
    'cuda': True,
    'seed': 7
}
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


