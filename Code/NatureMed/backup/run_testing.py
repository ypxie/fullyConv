from __future__ import absolute_import
import os
import sys
os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=True'

cloudRoot  = os.path.join('..','..','..','..')
projroot = os.path.join('..','..')
homeroot = os.path.join('..','..','..','..', '..')

testingimageroot = os.path.join(homeroot,'DataSet', 'Nature_Med')

kerasversion = 'keras-1'
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras'))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras','layers'))
sys.path.insert(0, '..')
sys.path.insert(0, os.path.join(projroot,'code', 'proj_utils') )

dataroot = os.path.join(projroot, 'Data')
modelroot = os.path.join(projroot, 'Data', 'Model')


import numpy as np
import deepdish as dd
from testingclass import runtestImg
import warnings
warnings.filterwarnings("ignore")
from loss_fun import weighted_loss
from proj_utils.keras_utils import elu
from keras.optimizers import Adadelta, SGD, RMSprop, adam

#------------------------old--------------------------------model
det_dataset = 'Com_Det'    
det_modelname = 'deep_det_fcn'
weights_name = 'weights.h5'
from kerasOneModel import buildCellModel as buildmodel

sameModel = True
strumodel = None
if __name__ == "__main__":
    #Probrefresh = 0
    Seedrefresh = 1
    thresh_pool = np.arange(0.0,0.7,0.05)
    lenpool = [5, 6, 7, 8,9,10] #[3,5,7] #for cell detection

    steppool = [1] #range(3,49,2);
    showmap = 1
    showseed = 1
    batchsize = 1280
    resizeratio = 1
    printImg = 0
    img_channels = 3
    testingParam = {}
    testingParam['windowsize'] = 500
    testingParam['batch_size'] = 2
    testingParam['fixed_window'] = False
    testingParam['board'] = 30
    testingParam['step_size'] = None

                    # testingset, ImgExt, trainingset, modelsubfolder, testtype , test, evaluate ,Probrefresh
    testingpool = [ #('BoneMarrow' , [ '.png'], trainingset,  modelsubfolder, 'fcn',     True, False, True),
                    ('Brain' , ['.png'], det_dataset,  det_modelname, 'fcn',True, False, True),
                    ('Breast', ['.jpg'], det_dataset,  det_modelname, 'fcn',True, False, True),
                    ('Lung/Lung Adenocarcinoma' , ['.jpg'], det_dataset,  det_modelname, 'fcn',True, False, True),
                    ('Lung/Lung Squamous Cell Carcinoma' , ['.jpg'], det_dataset,  det_modelname, 'fcn',True, False, True),
                    ('NET/TMA1' , ['.jpg'], det_dataset,  det_modelname, 'fcn',True, False, True),
                    ('NET/TMA2' , ['.jpg'], det_dataset,  det_modelname, 'fcn',True, False, True),
                  ]

    for tetstingset in testingpool:
        testtype = 'fcn'
        testingset, ImgExt, trainingset, modelsubfolder, testtype , test, evaluate ,Probrefresh = tetstingset

        weights_name_noext, _ = os.path.splitext(weights_name)
        metric_name  = modelsubfolder + 'metric' + '_' + weights_name_noext
        resultmask   = modelsubfolder  + '_' + weights_name_noext

        testingimagefolder = os.path.join(testingimageroot, testingset)

        #testingimagefolder = os.path.join(dataroot,'TrainingData', testingset)
        savefolder = os.path.join(testingimagefolder, resultmask)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        if sameModel != True or strumodel is None:
            modelfolder = os.path.join(modelroot,trainingset, modelsubfolder)
            meandevpath = os.path.join(modelfolder, 'meanstd.h5')
            arcpath = os.path.join(modelfolder, 'arc.json')
            weightspath = os.path.join(modelfolder,weights_name)
            paramspath = os.path.join(modelfolder, 'params.h5')
            ModelDict = {}
            meandev = dd.io.load(meandevpath)
            trainparam = dd.io.load(paramspath)
            #trainparam = None
            if test:
                #strumodel = buildmodel(img_channels = 3)
                strumodel = buildmodel(img_channels, last_activ='relu')
                #strumodel = buildmodel(img_channels=3, activ=activ, last_activ='relu',pooling=pooling, nf=nf,make_predict = True)
                strumodel.load_weights(weightspath)
            else:
                strumodel = {}

        ModelDict['params'] = trainparam
        ModelDict['thismean'] = meandev['thismean']
        ModelDict['thisdev'] = meandev['thisdev']
        ModelDict['model'] = strumodel

        classparams = {}
        classparams['ImgDir'] = testingimagefolder
        classparams['savefolder'] = savefolder
        classparams['resultmask'] = resultmask
        classparams['ImgExt'] =  ImgExt
        classparams['patchsize']   = ModelDict['params']['patchsize']
        classparams['labelpatchsize']   = ModelDict['params']['labelpatchsize']

        classparams['resizeratio'] = resizeratio

        classparams['model'] =     ModelDict['model']
        #classparams['steppool'] =  [ModelDict['params']['labelpatchsize']]
        classparams['steppool'] = steppool
        classparams['channel'] =   ModelDict['params']['channel']
        classparams['thismean']  = ModelDict['thismean']
        classparams['thisdev']  =  ModelDict['thisdev']

        classparams['Probrefresh']  =  Probrefresh
        classparams['Seedrefresh']  =  Seedrefresh

        classparams['lenpool']  =  lenpool
        classparams['showmap']  =  showmap
        classparams['showseed'] = showseed
        classparams['batchsize']  =  batchsize
        classparams['test_type'] = testtype
        #------------------------The rest param is for evaluation-----------
        # classparams['metric_name']  =  metric_name
        # classparams['gt_folder']  = 'gt_perimysium'
        # classparams['LabelExt']  = ['.mat']
        # classparams['contourext']  = ['']
        # classparams['contourname']   = 'paramesium'
        # classparams['thresh_pool']  =  thresh_pool

        tester = runtestImg(classparams)
        if test:
            print('start testing!')
        tester.runtesting(**testingParam)

        if printImg:
            tester.printCoords(threshhold = 0.5, step = 1, min_len = 8)
