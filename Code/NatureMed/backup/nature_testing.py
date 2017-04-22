from __future__ import absolute_import
import os
import sys
os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=True'

cloudRoot  = os.path.join('..','..','..','..')
projroot = os.path.join('..','..')

kerasversion = 'keras-1'
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras'))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras','layers'))
sys.path.insert(0, '..')
sys.path.insert(0, os.path.join('..', 'proj_utils') )

dataroot = os.path.join(projroot, 'Data')
modelroot = os.path.join(projroot, 'Data', 'Model')


import numpy as np
import deepdish as dd
from testingclass import runtestImg
import warnings
warnings.filterwarnings("ignore")

from kerasOneModel import buildCellModel as buildmodel
from loss_fun import weighted_loss
from proj_utils.keras_utils import elu
activ = elu(alpha=1.0) 
last_activ = 'relu'   
from keras.optimizers import Adadelta  ,SGD, RMSprop, adam

from lung_left_med import loop_configuration
#loop_configuration = [-1]

#trainingset  = "Mixture"
#modelsubfolder = 'mixtureModel'
trainingset = 'det_lung_left' 
modelsubfolder = 'deep_det_fcn'
testingimageroot = os.path.join(cloudRoot,'DataSet',  'Mixture_DataSets', 'TestingData')

sameModel = True
strumodel = None
if __name__ == "__main__":
    #Probrefresh = 0
    Seedrefresh = 0
    thresh_pool = np.arange(0.0,0.6,0.05)
    lenpool = [6] #[3,5,7] #for cell detection
    
    steppool = [1] #range(3,49,2);
    #stridepool = [1,3,6,9,12,15,18,21,24,27,30,33,36,40,43,46,48] #  ModelDict['params']['labelpatchsize']
    #stridepool = [3,6,12,18,24,30,48]
    #patchsize = 5
    #labelpatchsize = 5

    #steppool = [patchsize]
    showmap = 1
    showseed = 1
    batchsize = 1280
    resizeratio = 1
    predict_flag = 0 # for MDRNN
    printImg = 1
    #modelsubfolder = 'Gradu_relu_10' , 'Gradu_relu_10'
    #marker, modelsubfolder = 'MDCWRNN_toy_debug_5' , 'toy_debug_5'
    # testing_data, data_ext, training_Data, model_name,test_type, predict_flag, test, evaluate,Probrefresh
    testingpool = [ 
                    #('BoneMarrow' , [ '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('BrainTumor' , ['.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('BreastCancer' , ['.tif'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    ('Lung' , ['.tif'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('Net' , ['.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('PhaseContrast' , ['.tif'],  trainingset, modelsubfolder, 'fcn',0, True, False, True),
    
                    #('minitest' , ['.tif', '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('breast_TMI_testing' , ['.tif', '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('newTest' , ['.tif', '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('BM' , ['.tif', '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('testPhasecontrast_convert' , ['.tif', '.png'], trainingset, modelsubfolder, 'fcn',0, True, False, True),
                    #('imgs_lung' , ['.tif', '.png'], 'breastcancer_TMI_training', 'FCN_Demo_', 'fcn',0, True, False, True),
                    #('NET_whole_img' , ['.jpg'], 'NET_demo', 'FCN', 'fcn',0, True, False, True),
                    #('imgs_ki67' , ['.png'], 'NET_demo', 'FCN', 'fcn',0, True, False, True),
                    #('breastImg_5000x5000' , ['.png'], 'breastcancer_TMI_training', 'FCN_Demo_', 'fcn',0, False, False, True),
                    #('imgs_breast' , ['.png'], 'breastcancer_TMI_training', 'FCN_Demo_', 'fcn',0, True, False, True),

                    #('MIA_breast' , ['.tif'], 'breastcancer_TMI_training', 'FCN', 'fcn',0, True, False, True),
                    #('breastcancer_TMI_training' , ['.png'], 'breastcancer_TMI_training', 'FCN', 'fcn',0, True, False, True),
                    #('testPhasecontrast_convert' , ['.tif'], 'trainPhasecontrast_convert', 'FCN', 'fcn',0, True, False, True),
                    #('newTest' , ['.png'], 'Select_NET', 'FCN', 'fcn',0, True, False, True)
                   ]
    for tetstingset in testingpool:
        testtype = 'fcn'
        testingset, ImgExt, trainingset, modelsubfolder, testtype,predict_flag, test, evaluate ,Probrefresh = tetstingset

        for this_config in loop_configuration:
            if this_config == 'whole_data': 
                
                number_list = [-1]
                weights_name = 'weights.h5'
                params_name  = 'params.h5'
                maxepoch = 100
                trainable_list= [True, True] # do you wanna fine tune the weights
            else:
                number_list = [this_config]
                weights_name = 'weights' +'_'+str(this_config) + '.h5' 
                params_name  = 'params'  +'_'+str(this_config) + '.h5'
                maxepoch = 100
                trainable_list= [False, True] # do you wanna fine tune the weights

            weights_name_noext, _ = os.path.splitext(weights_name)
            metric_name  = modelsubfolder + 'metric' + '_' + weights_name_noext
            resultmask   = trainingset + '_' + modelsubfolder  + '_' + weights_name_noext

            testingimagefolder = os.path.join(testingimageroot, testingset)
            print testingimagefolder
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
        
                if test:
                    strumodel = buildmodel(img_channels = 3)
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

            classparams['predict_flag'] = predict_flag
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
            classparams['metric_name']  =  metric_name
            classparams['gt_folder']  = 'gt_perimysium'
            classparams['LabelExt']  = ['.mat']
            classparams['contourext']  = ['']
            classparams['contourname']   = 'paramesium'
            classparams['thresh_pool']  =  thresh_pool


            tester = runtestImg(classparams)
            if test:
                print('start testing!')
            tester.runtesting()


            if evaluate:
                metricsDict = tester.Evaluation()

            tester.analyzeMetric(metricsDict)
            if printImg:
                tester.printCoords(threshhold = 0.10, step = 1, min_len = 5)
