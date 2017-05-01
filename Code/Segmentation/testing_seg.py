import os, sys
import argparse

projroot   = os.path.join('..','..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))

testingimageroot = os.path.join(home,'DataSet', 'FY_TMI', 'test')
modelroot = os.path.join(projroot, 'Data','Model')

from collections import namedtuple
import numpy as np
import argparse

import torch
from torch_fcn.Models import MultiContex_seg as build_model
from torch_fcn.proj_utils.testingclass import runtestImg

parser = argparse.ArgumentParser(description = 'Cell detection testing args.')

parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--show_progress', action='store_false', default=True,
                    help='show the training process using images')

parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--showmap', action='store_false', default= False, help='show the probability map.')
parser.add_argument('--showseed', action='store_false', default=False, help='show the seed marked on the image.')
parser.add_argument('--showseg', action='store_false', default= False, help='show the segmentation marked on the iamge.')
parser.add_argument('--resizeratio', default=1, help='If you want to resize the original test image.')

parser.add_argument('--printImg', action='store_false', default=True, help='If you want to print the resulting images.')

parser.add_argument('--lenpool', default=[7,9,11, 13], help='pool of length to get local maxima.')
#parser.add_argument('--thresh_pool', default = np.arange(0.05, 0.4, 0.05), help='pool of length to get local maxima.')
parser.add_argument('--thresh_pool', default = [0.25], help='pool of length to get local maxima.')

parser.add_argument('--seg_thresh_pool', default = np.arange(0.05, 0.6, 0.1), help='pool of length to get local maxima.')

parser.add_argument('--img_channels', type=int, default=3, metavar='N', help='Input image channel.')
parser.add_argument('--multi_context', action='store_false', default= True,
                            help='If use a multi context information or not.')

args = parser.parse_args()

test_tuple = namedtuple('test', 'testingset ImgExt trainingset det_model_folder weights_name Probrefresh Seedrefresh')

testing = True
device_id = 2
det_model = build_model(multi_context = args.multi_context)

modeltype = 'multiout' if args.multi_context else 'multiout_no_multicont'

if args.cuda:
    det_model.cuda(device_id)
 
if __name__ == "__main__":
    img_channels = 3
    testingpool = [ 
                    test_tuple('breast', ['.tif'], 'breast', modeltype ,'weights.pth',  True, True)
                  ]
    testingParam = {}
    testingParam['windowsize'] = 3000
    testingParam['batch_size'] = args.batch_size
    testingParam['fixed_window'] = False
    testingParam['board'] = 30
    testingParam['step_size'] = None

    for this_test in testingpool:
        testingset = this_test.testingset
        ImgExt  = this_test.ImgExt
        trainingset = this_test.trainingset
        det_model_folder = this_test.det_model_folder
        weights_name = this_test.weights_name
        Probrefresh = this_test.Probrefresh
        Seedrefresh = this_test.Seedrefresh

        weights_name_noext, _ = os.path.splitext(weights_name)
       
        resultmask   = det_model_folder  + '_' + weights_name_noext
        testingimagefolder = os.path.join(testingimageroot, testingset)
        savefolder = os.path.join(testingimagefolder, resultmask)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        
        det_modelfolder = os.path.join(modelroot,trainingset, det_model_folder)
        #seg_modelfolder = os.path.join(modelroot,trainingset, seg_model_folder)

        det_weightspath = os.path.join(det_modelfolder,weights_name)    
        #seg_weightspath = os.path.join(seg_modelfolder,weights_name)
        
        ModelDict = {}

        weights_dict = torch.load(det_weightspath)
        print(det_weightspath)
        det_model.load_state_dict(weights_dict['weights'])# 12)

        classparams = {}
        classparams['ImgDir'] = testingimagefolder
        classparams['savefolder'] = savefolder
        classparams['resultmask'] = resultmask
        classparams['ImgExt'] =  ImgExt

        classparams['resizeratio'] = args.resizeratio

        classparams['model'] =     det_model
        classparams['Probrefresh']  =  Probrefresh
        classparams['Seedrefresh']  =  Seedrefresh

        classparams['thresh_pool']  =  args.thresh_pool
        classparams['seg_thresh_pool']  =  args.seg_thresh_pool

        classparams['lenpool']  =  args.lenpool
        classparams['showmap']  =  args.showmap
        classparams['showseed'] =  args.showseed
        classparams['showseg'] =   args.showseg
        classparams['batch_size']  =  args.batch_size
        
        tester = runtestImg(classparams)
        if testing:
           tester.folder_seg(**testingParam)

        if args.printImg:
            tester.printContours(threshhold= args.thresh_pool[0],  seg_thresh= args.seg_thresh_pool[3], min_len=args.lenpool[-2])
