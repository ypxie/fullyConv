import sys
import os
import argparse

projroot   = os.path.join('..','..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))

modelroot = os.path.join(projroot, 'Data','Model')
trainingset = 'Com_Det' #'breast'
#trainingimagefolder = os.path.join(home,'DataSet', 'FY_TMI', 'train', trainingset)
trainingimagefolder = os.path.join(home,'DataSet', trainingset)

from time import time
import numpy as np

from torch_fcn.proj_utils.Extractor import FcnExtractor
from torch_fcn.proj_utils.loss_fun import weighted_loss
from torch_fcn.Models import MultiContex as build_model
from torch_fcn.proj_utils.train_eng import train_blocks
import torch.optim as optim

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Nature cell detection training')

    parser.add_argument('--reuse_weigths', action='store_false', default=True,
                        help='continue from last checkout point')

    parser.add_argument('--show_progress', action='store_false', default=True,
                        help='show the training process using images')

    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxepoch', type=int, default=1280, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for training')
    
    parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
    parser.add_argument('--patchsize', type=int, default=200, metavar='S', help='training patch size')
    parser.add_argument('--showfre', type=int, default=20, metavar='S', help='freq of batch to show testing images.')

    parser.add_argument('--savefre', type=int, default=5, metavar='S', help='freq of batch to save the model.')

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

    #the following is for the data augmentation
    parser.add_argument('--number_repeat', type=int, default=2, metavar='N',
                        help='number of repeats to loop over each data patch.')                   
    args = parser.parse_args()

    creteria = weighted_loss(base = 'mse')
    #loss = weighted_loss(base = 'mse')
    #use_weighted = 1    #means you have a weight mask in the last dimension of mask
    #loss = 'mse'
    #use_weighted = 0    #means you have a weight mask in the last dimension of mask
    #opt = adam(lr=lr, clipnorm=100)
    #opt = SGD(lr=lr, momentum=0.9, decay=1e-8, nesterov=True,clipvalue=0.1)

    strumodel = build_model()
    if args.cuda:
        strumodel.cuda(2)

    #opt = optim.Adadelta(strumodel.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    opt = optim.SGD(strumodel.parameters(), lr=args.lr, momentum=0.9, nesterov = True, weight_decay=args.weight_decay)

    modelsubfolder = 'multicontex'
    
    modelfolder = os.path.join(modelroot, trainingset,modelsubfolder)

    classparams = {}
    classparams['patchsize']   = args.patchsize
    classparams['labelpatchsize']   = args.patchsize
    classparams['channel'] = args.img_channels
    classparams['label_channel'] = args.label_channels  # one for mask and last dimension is for weight mask
    classparams['ImageGenerator_Identifier'] =  'process_mask_with_weight' #'process_contour_cellcounting' # 'process_cell_mask'
    # the following is for Image generator parameters
    decayparam = {}
    decayparam['alpha'] = 3
    decayparam['r'] = 15
    decayparam['scale'] = 5
    decayparam['decalratio'] = 0.1
    decayparam['smallweight'] = 0.05
    decayparam['use_gauss'] = 0
    
    weight_params = {}
    weight_params['beta'] = 1.0/decayparam['scale']
    weight_params['alpha'] = (1.0* 5)/decayparam['scale']

    classparams['decayparam'] = decayparam
    classparams['w'] = 10 # weight wrt distance
    classparams['wc'] = 1 # weight wrt to class
    classparams['dilate'] = 8

    classparams['datadir'] = trainingimagefolder
    classparams['volume'] = 2
    classparams['dataExt'] = ['.tif', '.png']            # the data ext
    classparams['labelExt']    = ['.mat']             # the label ext
    classparams['contourname']   = 'Contours'         # the contour name in the cell array
    classparams['labelSuffix']  = ["",'_withcontour', '_gt','_seg'] # the suffix of label
    classparams['maxsamples']  = 1280000
    classparams['usecontour']  = 1 # this is just used to fill the cotour to get filled_img, 
    classparams['pickratio']   = 0.05  # 1 means take all the pixel

    classparams['maximg'] = 50
    classparams['mask_thresh'] = 50
    classparams['mask_prob'] =0.1
    classparams['maxpatch'] = 10
    classparams['random_pick'] =  True # if you wanna random pick
    
    StruExtractor = FcnExtractor(classparams)
    
    validation_params = classparams.copy()
    #validation_params['datadir'] = validation_folder
    validation_params['get_validation'] = True
    validation_params['maxpatch'] = 1
    validation_params['maximg'] =  50
    validation_params['patchsize'] = args.patchsize
    validation_params['labelpatchsize']= args.patchsize
    
    
    best_score = 10000
    tolerance = 10000
    worseratio = 2
    train_param = {
    'name':modelsubfolder,
    'strumodel':strumodel,
    'optimizer': opt,
    'creteria': creteria,
    'modelfolder': modelfolder,
    'classparams':classparams,
    'validation_params':validation_params,
    'weight_params' : weight_params,
    'StruExtractor': StruExtractor,
    
    'best_score': best_score,
    'tolerance':tolerance,
    'worseratio': worseratio,
    }
    train_blocks(train_param, args)  
