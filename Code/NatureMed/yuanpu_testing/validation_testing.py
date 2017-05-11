import os, sys
import argparse

projroot   = os.path.join('..','..','..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))

#testingimageroot = '/media/yuanpuxie/lab_drive/patches'
#testingimageroot = '/media/yuanpuxie/Seagate Backup Plus Drive/TCGA_data/CropImage/CropImage'

#testingimageroot = '/media/yuanpuxie/LinlinGao-BME360'
#testingimageroot = '/media/yuanpuxie/FujunLiu/muscle-classification/nuclei_annotation'

from collections import namedtuple
import numpy as np
import argparse

import torch
from torch_fcn.Models import MultiContex as build_model
from torch_fcn.proj_utils.testingclass import runtestImg
from torch_fcn.proj_utils.evaluation import eval_folder

from torch_fcn.proj_utils.local_utils import Indexflow
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description = 'Cell detection testing args.')

parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--show_progress', action='store_false', default=True,
                    help='show the training process using images')

parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--showmap', action='store_false', default=False, help='show the probability map.')
parser.add_argument('--showseed', action='store_false', default=False, help='show the seed marked on the image.')
parser.add_argument('--showseg', action='store_false', default=False, help='show the segmentation marked on the iamge.')
parser.add_argument('--resizeratio', default=1, help='If you want to resize the original test image.')

parser.add_argument('--printImg', action='store_false',   default=True, help='If you want to print the resulting images.')
parser.add_argument('--runTest',  action='store_false',   default=True, help='If you want to test the image for reslst.')
parser.add_argument('--runEval',  action='store_false',   default=True, help='If you want to run the evaluation code.')
parser.add_argument('--indvidual',  action='store_false', default=True, help='If you want to run individual model.')

parser.add_argument('--lenpool', default=[5,7,9,11,13, 15,17], help='pool of length to get local maxima.')
parser.add_argument('--thresh_pool', default = np.arange(0.0, 1, 0.05), help='pool of length to get local maxima.')
#parser.add_argument('--thresh_pool', default = [0.25, 0.3], help='pool of length to get local maxima.')
#parser.add_argument('--seg_thresh_pool', default = [0.3], help='pool of length to get local maxima.')
parser.add_argument('--img_channels', type=int, default=3, metavar='N', help='Input image channel.')

args = parser.parse_args()
det_model = build_model()
if args.cuda:
    det_model = det_model.cuda()

def get_resultmask(trainingset, model_folder, weightname):
    return  trainingset + '_' + model_folder + '_' + weightname

def test_worker(testingpool, det_model, testingimageroot):
    testingParam = {}
    testingParam['windowsize'] = 3000
    testingParam['batch_size'] = args.batch_size
    testingParam['fixed_window'] = False
    testingParam['board'] = 30
    testingParam['step_size'] = None

    for this_test in testingpool:
        testingset = this_test.testingset
        ImgExt = this_test.ImgExt
        trainingset = this_test.trainingset
        det_model_folder = this_test.det_model_folder
        weights_name = this_test.weights_name
        Probrefresh = this_test.Probrefresh
        Seedrefresh = this_test.Seedrefresh
        modelroot   = this_test.modelroot

        weights_name_noext, _ = os.path.splitext(weights_name)
        resultmask = get_resultmask(this_test.trainingset, det_model_folder, weights_name_noext)
        testingimagefolder = os.path.join(testingimageroot, testingset)
        savefolder = os.path.join(testingimagefolder, resultmask)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        det_modelfolder = os.path.join(modelroot, trainingset, det_model_folder)
        det_weightspath = os.path.join(det_modelfolder, weights_name)
        ModelDict = {}
        print(det_weightspath)
        weights_dict = torch.load(det_weightspath,map_location=lambda storage, loc: storage)
        det_model.load_state_dict(weights_dict)  # 12)

        ModelDict['model'] = det_model

        classparams = {}
        classparams['ImgDir'] = testingimagefolder
        classparams['savefolder'] = savefolder
        classparams['resultmask'] = resultmask
        classparams['ImgExt'] = ImgExt

        classparams['resizeratio'] = args.resizeratio

        classparams['model'] = ModelDict['model']
        classparams['Probrefresh'] = Probrefresh
        classparams['Seedrefresh'] = Seedrefresh

        classparams['thresh_pool'] = args.thresh_pool
        # classparams['seg_thresh_pool']  =  args.seg_thresh_pool
        classparams['lenpool'] = args.lenpool
        classparams['showmap'] = args.showmap
        classparams['showseed'] = args.showseed
        classparams['showseg'] = args.showseg
        classparams['batch_size'] = args.batch_size

        tester = runtestImg(classparams)
        tester.folder_det(**testingParam)

        if args.printImg:
            tester.printCoords(threshhold=args.thresh_pool[0], step=1, min_len=args.lenpool[-1])

if __name__ == "__main__":
    dataroot = os.path.join(projroot, 'Data')
    modelroot = os.path.join(dataroot, 'NatureModel', 'YuanpuModel')

    testingimageroot = os.path.join(home, 'Dropbox', 'DataSet', 'NatureData','YuanpuData', 'ValidationData')
    test_tuple = namedtuple('test',
                            'testingset ImgExt trainingset modelroot det_model_folder weights_name Probrefresh Seedrefresh')

    testing_folders = np.array([
                             ('AdrenalGland'),
                             ('Bladder'),
                             ('Breast'),
                             ('Colorectal'),
                             ('Eye'),
                             ('Kidney'),
                             ('Lung'),
                             ('Ovary'),
                             ('Pleura'),
                             ('Skin'),
                             ('Stomach'),
                             ('Thymus'),
                             ('Uterus'),
                             ('BileDuct'),
                             ('Brain'),
                             ('Cervix'),
                             ('Esophagus'),
                             ('HeadNeck'),
                             ('Liver'),
                             ('LymphNodes'),
                             ('Pancreas'),
                             ('Prostate'),
                             ('SoftTissue'),
                             ('Testis'),
                             ('Thyroid')
                        ])
    template_pool = [None, ['.tif', '.png', '.jpg'], 'All', modelroot,'multicontex' ,'best_weights.pth',  True, True]
    template_tuple =  test_tuple(*template_pool)

    testing_pool = []
    
    for this_foldername in testing_folders:
        this_pool = template_pool[:]
        this_pool[0] = this_foldername
        if args.indvidual:
            this_pool[2] = this_foldername
        
        testing_pool.append(test_tuple(*this_pool))
    if args.runTest:
        test_worker(testing_pool, det_model, testingimageroot)

    if args.runEval:
        saveroot = os.path.join(home, 'Dropbox', 'DataSet', 'NatureData','YuanpuData','Experiments','evaluation_validation')
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
        
        for idx, this_foldername in enumerate(testing_folders):
            savefolder = os.path.join(saveroot, this_foldername)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            this_tuple = testing_pool[idx]
            weights_name = this_tuple.weights_name
            det_model_folder = this_tuple.det_model_folder

            weights_name_noext, _ = os.path.splitext(weights_name)
            resultmask = get_resultmask(this_tuple.trainingset, det_model_folder, weights_name_noext)
            this_folder = os.path.join(testingimageroot, this_foldername)
            resfolder   = os.path.join(this_folder, resultmask)
            eval_folder(imgfolder= this_folder, resfolder= resfolder,savefolder= savefolder,
                        radius=16, resultmask = resultmask,thresh_pool=args.thresh_pool,
                        len_pool= args.lenpool, imgExt=['.tif', '.jpg','.png'],contourname='Contours')
    #testingpool = [
                    #test_tuple('LymphNodes', ['.tif'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True),
                    #test_tuple('Thymus', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True),
                    #test_tuple('Thyroid', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True),
                    #test_tuple('Uterus', ['.tif'], 'Com_Det',  'multicontex' ,'weights.pth',  True, True)
                    #test_tuple('Brain', ['.jpg'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True),   
                    #test_tuple('test-small', ['.png'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True),   
                    #test_tuple('Liver', ['.tif'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True),
                    #test_tuple('Lung', ['.tif'], 'Com_Det', 'multicontex' ,'weights.pth',  True, True)
                  #]
