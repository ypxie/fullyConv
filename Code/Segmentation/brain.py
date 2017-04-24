import sys, os
import numpy as np
projroot   = os.path.join('..','..')
coderoot   = os.path.join(projroot, 'Code')
sys.path.insert(0, os.path.join(coderoot))
from segmentation import train_worker

home = os.path.expanduser('~')
trainingDataroot = os.path.join(home,'DataSet', 'FY_TMI', 'train')
modelroot = os.path.join(projroot, 'Data','Model')


import argparse
parser = argparse.ArgumentParser(description='Nature cell detection training')

parser.add_argument('--reuse_weigths', action='store_false', default=True,
                    help='continue from last checkout point')

parser.add_argument('--show_progress', action='store_false', default=show_progress,
                    help='show the training process using images')

parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--maxepoch', type=int, default=640, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for training')

parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--patchsize', type=int, default=200, metavar='S', help='training patch size')
parser.add_argument('--showfre', type=int, default=100, metavar='S', help='freq of batch to show testing images.')

parser.add_argument('--savefre', type=int, default=10, metavar='S', help='freq of batch to save the model.')

parser.add_argument('--refershfreq', type=int, default=2, metavar='S', help='refesh the training data')
parser.add_argument('--chunknum', type=int, default=384, metavar='S', help='number of image in each chunk')

parser.add_argument('--label_channels', type=int, default=2, metavar='S', help='the channel of label')
parser.add_argument('--img_channels', type=int, default=3, metavar='S', help='the channel of image')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--use_validation', action='store_false', default=False,
                    help='If use validation or not.')
parser.add_argument('--valid_freq', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--use_weighted', action='store_false', default=True,
                    help='If use validation or not.')
parser.add_argument('--use_reinforce', action='store_false', default=False,
                            help='If use validation or not.')

# the following is for the data augmentation
parser.add_argument('--number_repeat', type=int, default=2, metavar='N',
                    help='number of repeats to loop over each data patch.')

train_worker(trainingDataroot = trainingDataroot, trainingset= 'brain',
             show_progress=True, modelroot=modelroot, modelsubfolder = 'multiout')