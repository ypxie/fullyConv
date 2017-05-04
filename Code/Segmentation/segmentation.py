import sys, os
import numpy as np
projroot   = os.path.join('..','..')
coderoot   = os.path.join(projroot, 'Code')
sys.path.insert(0, os.path.join(coderoot))

import argparse
from time import time
import numpy as np

from torch_fcn.proj_utils.Extractor import FcnExtractor
from torch_fcn.proj_utils.loss_fun import weighted_loss,dice
from torch_fcn.Models import MultiContex_seg as build_model
from torch_fcn.proj_utils.train_eng import train_blocks_double
import torch.optim as optim

def train_worker(trainingDataroot, trainingset, modelroot='.', device=0, multi_context=True,
                 show_progress=False, modelsubfolder = 'multicontex', parser=None):
    if parser is None:
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
        parser.add_argument('--showfre', type=int, default=200, metavar='S',
                            help='freq of batch to show testing images.')

        parser.add_argument('--savefre', type=int, default=200, metavar='S', help='freq of batch to save the model.')

        parser.add_argument('--refershfreq', type=int, default=2, metavar='S', help='refesh the training data')
        parser.add_argument('--chunknum', type=int, default=384, metavar='S', help='number of image in each chunk')

        parser.add_argument('--label_channels', type=int, default=2, metavar='S', help='the channel of label')
        parser.add_argument('--img_channels', type=int, default=3, metavar='S', help='the channel of image')

        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

        parser.add_argument('--use_validation', action='store_false', default=False,
                            help='If use validation or not.')
        parser.add_argument('--validfreq', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--use_weighted', action='store_false', default=True,
                            help='If use validation or not.')
        parser.add_argument('--use_reinforce', action='store_false', default=False,
                            help='If use validation or not.')

        parser.add_argument('--multi_context', action='store_false', default=multi_context,
                            help='If use a multi context information or not.')

        # the following is for the data augmentation
        parser.add_argument('--number_repeat', type=int, default=2, metavar='N',
                            help='number of repeats to loop over each data patch.')

    args = parser.parse_args()

    trainingimagefolder = os.path.join(trainingDataroot, trainingset)
    modelfolder = os.path.join(modelroot, trainingset,modelsubfolder)

    name = '_'.join([trainingset,modelsubfolder])

    det_creteria = weighted_loss(base = 'mse')
    seg_creteria = dice
    print("The multi_context setting is ", args.multi_context)
    strumodel = build_model(multi_context = args.multi_context)
    if args.cuda:
        strumodel.cuda(device)

    #opt = optim.Adadelta(strumodel.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    #opt = optim.SGD(strumodel.parameters(), lr=args.lr, momentum=0.9, nesterov = True, weight_decay=args.weight_decay)
    opt = optim.Adamax(strumodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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

    classparams['double_output'] = True
    classparams['decayparam'] = decayparam
    classparams['w'] = 10 # weight wrt distance
    classparams['wc'] = 1 # weight wrt to class
    classparams['dilate'] = 8

    classparams['datadir'] = trainingimagefolder
    classparams['volume'] = 2
    classparams['dataExt'] = ['.tif', '.png', '.jpg']            # the data ext
    classparams['labelExt']    = ['.mat']             # the label ext
    classparams['contourname']   = 'Contours'         # the contour name in the cell array
    classparams['labelSuffix']  = ["",'_withcontour', '_gt','_seg'] # the suffix of label
    classparams['maxsamples']  = 1280000
    classparams['usecontour']  = 1 # this is just used to fill the cotour to get filled_img, 
    classparams['pickratio']   = 0.2  # 1 means take all the pixel

    classparams['maximg'] = 30
    classparams['mask_thresh'] = 50
    classparams['mask_prob'] =0.1
    classparams['maxpatch'] = 30
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
    'name':name,
    'strumodel':strumodel,
    'optimizer': opt,
    'creteria': [det_creteria, seg_creteria],
    'modelfolder': modelfolder,
    'classparams':classparams,
    'validation_params':validation_params,
    'weight_params' : weight_params,
    'StruExtractor': StruExtractor,
    
    'best_score': best_score,
    'tolerance':  tolerance,
    'worseratio': worseratio,
    }
    train_blocks_double(train_param, args)  
    print('Just finish the training of {}'.format(trainingset))

if __name__ == '__main__':
    
    home = os.path.expanduser('~')
    trainingDataroot = os.path.join(home,'DataSet', 'FY_TMI', 'train')
    modelroot = os.path.join(projroot, 'Data','Model')

    train_worker(trainingDataroot = trainingDataroot, trainingset= 'breast',device=0,  multi_context = True,
                 show_progress=True, modelroot=modelroot, modelsubfolder = 'multiout')
    #train_worker(trainingDataroot=trainingDataroot, trainingset='breast', multi_context=False,
    #             show_progress=True, modelroot=modelroot, modelsubfolder='multiout_no_multicont')
