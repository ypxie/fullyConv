import sys
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=False'
cloudRoot  = os.path.join('..','..','..','..') 
projroot = os.path.join('..','..')
home = os.path.join(cloudRoot, '..')

modelroot = os.path.join(projroot, 'Data')
dataroot = os.path.join(projroot, 'Data')
trainingset = 'nature_mixed'
pooling = 'max'#or max
modelsubfolder = 'residule_fcn_' + pooling 

modelfolder = os.path.join(dataroot, 'Model',trainingset,modelsubfolder)
trainingimagefolder = os.path.join(dataroot,'TrainingData','general_model_learning', trainingset)
#trainingimagefolder = os.path.join(home,'DataSet', 'MIA_det', 'TrainingData','general_model_learning', trainingset)

kerasversion = 'keras-1'
#kerasversion = 'keras_classical'
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras'))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras','layers'))
sys.path.insert(0, '..')


from time import time
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

from proj_utils.train_eng import get_mean_std
from proj_utils.local_utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from Extractor import FcnExtractor
from keras.utils import  generic_utils

from kerasOneModel import buildMIAResidule as buildmodel

sys.setrecursionlimit(10000)
from loss_fun import weighted_loss
from proj_utils.keras_utils import elu
activ = elu(alpha=1.0) 
#activ = 'relu'
last_activ = 'relu'   
from keras.optimizers import Adadelta  ,SGD, RMSprop, adam
from MIA.mia_util import get_wight_mask

if  __name__ == '__main__':
    
    loss = weighted_loss(base = 'mse')
    use_weighted = 1    #means you have a weight mask in the last dimension of mask
    #loss = 'mse'
    #use_weighted = 0    #means you have a weight mask in the last dimension of mask

    # for build the model    
    img_channels = 3    
    lr = 0.0005
    #opt = adam(lr=lr, clipnorm=100)
    #opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=10)
    opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    weight_decay = 1e-8
    
    batch_size = 8
    meanstd = 0   # it is important to set this as 0 for FCN
    chunknum = int(320)
    maxepoch = 1280
    matrefresh = 1
    meanstdrefresh = 1
    refershfreq = 2
    savefre = 1  # np.mod(chunidx, savefre) == 0:

    rebuildmodel = 1
    reuseweigths = 1
    show_progress = 0 #if you want to show the testing cases.
    
    patchsize = 135
    labelpatchsize = 135
    label_channels = 2
    classparams = {}
    classparams['patchsize']   = patchsize
    classparams['labelpatchsize']   = labelpatchsize
    classparams['channel'] = img_channels
    classparams['label_channel'] = label_channels  # one for mask and last dimension is for weight mask
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


    decayparam['use_weighted'] = use_weighted
    decayparam['reverse_channel'] = True
    decayparam['filp_channel'] = True
    
    classparams['decayparam'] = decayparam
    classparams['w'] = 10 # weight wrt distance
    classparams['wc'] = 1 # weight wrt to class
    classparams['dilate'] = 8

    classparams['datadir'] = trainingimagefolder
    classparams['volume'] = 2
    classparams['dataExt'] = ['.tif', '.png', '.jpg']             # the data ext
    classparams['labelExt']    = ['.mat']          # the label ext
    classparams['contourname']   = 'Contours'      # the contour name in the cell array
    classparams['labelSuffix']  = ["",'_withcontour', '_gt','_seg'] # the suffix of label

    # classparams['datadir'] = trainingimagefolder
    # classparams['ImgExt'] = ['.bmp']
    # classparams['gt_folder']  = 'gt_perimysium'
    # classparams['LabelExt']  = ['.mat']
    # classparams['contourext']  = ['']
    # classparams['contourname']   = 'paramesium'
    # classparams['volume'] = 'paramesium'
    # classparams['contourname']   = 'paramesium'
    classparams['maxsamples']  = 1280000
    classparams['usecontour']  = 1 # this is just used to fill the cotour to get filled_img, 
    #classparams['dialate']     = 1
    classparams['usingindx']     = 1
    #classparams['padsize']     = labelpatchsize
    classparams['pickratio']   = 0.005  # 1 means take all the pixel
    classparams['crop_patch_size']   =  None #(0.95,0.95)
    classparams['selected_portion']   = 0.01  # 1 means take all the islet pixels
    #classparams['selected_num']   = 100  # how many patches you wanna take
    #classparams['mask_dilate'] = 30
    classparams['resizeratio'] =  [1]  #[1, 0.7]
    classparams['rotatepool']  = list(np.arange(0, 350, 15)) #[0,90]

    #classparams['step']  = labelpatchsize
    #classparams['mode'] = 'grid'
    classparams['maximg'] =  -1
    classparams['mask_thresh'] = 50
    classparams['mask_prob'] =0.1
    classparams['maxpatch'] = 10000
    classparams['random_pick'] =  True # if you wanna random pick
    
    StruExtractor = FcnExtractor(classparams)
    nb_class   = classparams['labelpatchsize']  ** 2
    
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)
    modelDict = {}
    modelpath = os.path.join(modelfolder, 'strumodel.h5') #should include model and other parameters
    weightspath = os.path.join(modelfolder,'weights.h5')
    best_weightspath = os.path.join(modelfolder,'best_weights.h5')
    arctecurepath = os.path.join(modelfolder,'arc.json')
    matpath = os.path.join(modelfolder,'matinfo.h5')
    meanstdpath = os.path.join(modelfolder, 'meanstd.h5')
    paramspath = os.path.join(modelfolder, 'params.h5')
    
    if not os.path.isfile(arctecurepath) or  rebuildmodel == 1:
        strumodel = buildmodel(img_channels,lr = lr, loss = loss,pooling=pooling, activ=activ, last_activ='relu')
        if reuseweigths == 1 and os.path.isfile(weightspath):
           strumodel.load_weights(weightspath )# 12)
          # import h5py
          # f = h5py.File(weightspath, mode='r')
          # g = f['graph']
          # weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
          # strumodel.set_weights(weights)
          # f.close()
    else:
       #strumodel = dd.io.load(modelpath)['model']
       strumodel = model_from_json(open(arctecurepath).read())
       strumodel.load_weights(weightspath)
    
    if not os.path.isfile(matpath) or matrefresh == 1:
       Matinfo = StruExtractor.getMatinfo_volume() # call this function to generate nece info
       #dd.io.save(matpath, Matinfo, compression='zlib')
    else:
       Matinfo = dd.io.load(matpath)
       StruExtractor.setMatinfo(Matinfo)
    datainfo = Matinfo['datainfo']

    meanstdDic = {}
    if not os.path.isfile(meanstdpath) or meanstdrefresh == 1:
       thismean, thisdev = get_mean_std(StruExtractor, meanstd)
       meanstdDic['thismean'] = thismean
       meanstdDic['thisdev'] = thisdev
       dd.io.save(meanstdpath, meanstdDic, compression='zlib')
    else:
       meanstdDic = dd.io.load(meanstdpath)
       thismean = meanstdDic['thismean']
       thisdev = meanstdDic['thisdev']


    thisbatch = np.zeros((chunknum,datainfo['inputdim']))
    thislabel = np.zeros((chunknum,datainfo['outputdim']))
 
    print('finish compiling!')
    best_score = 10000
    tolerance = 0.1
    for epochNumber in range(maxepoch):

        if np.mod(epochNumber+1, refershfreq) == 0:
                Matinfo = StruExtractor.getMatinfo_volume() # call this function to generate nece info
                thismean, thisdev = get_mean_std(StruExtractor, meanstd)
                datainfo = Matinfo['datainfo']
        Totalnum = datainfo['Totalnum']
        totalIndx = np.random.permutation(np.arange(Totalnum))

        numberofchunk = (Totalnum + chunknum - 1)// chunknum   # the floor
        chunkstart = 0
        progbar = generic_utils.Progbar(Totalnum)
        for chunkidx in range(numberofchunk):
                thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
                thisInd = totalIndx[chunkstart: chunkstart + thisnum]
                StruExtractor.getOneDataBatch_stru(thisInd, thisbatch[0:thisnum,:], thislabel[0:thisnum,:])
                chunkstart += thisnum
                BatchData = thisbatch[0:thisnum,:].astype(K.FLOATX)
                BatchLabel = thislabel[0:thisnum,:].astype(K.FLOATX)

                if nb_class == 2 and  labelpatchsize == 1:
                    BatchLabel = np.concatenate([BatchLabel, 1- BatchLabel], axis = -1)

                #---------------Train your model here using BatchData------------------
                BatchData -= thismean
                BatchData /= thisdev

                BatchData = np.reshape(BatchData, (-1,patchsize, patchsize, img_channels ))

                BatchLabel = np.reshape(BatchLabel, (-1,patchsize, patchsize, label_channels ))

                BatchLabel = np.transpose(BatchLabel, (0, 3,1,2))

                BatchData = np.transpose(BatchData, (0, 3,1,2))

                print('Training--Epoch--%d----chunkId--%d', (epochNumber, chunkidx))

                for X_batch, Y_batch in dataflow(BatchData, BatchLabel, batch_size ):
                    if use_weighted == 0:
                        loss = strumodel.train_on_batch({'input': X_batch},{'output_mask': Y_batch[:,0:-1,:,:]})
                    else:
                        Y_batch = get_wight_mask(Y_batch, weight_params)
                        loss = strumodel.train_on_batch({'input': X_batch}, {'output_mask': Y_batch})
                    if type(loss) == list:
                        loss = loss[0]
                    assert not np.isnan(loss) ,"nan error"
                
                progbar.add(BatchData.shape[0], values = [("train loss", loss)])
                if np.mod(chunkidx, savefre) == 0:
                    # print('\nTesting loss: {}, best_score: {}'.format(loss, best_score))
                    # if loss <=  best_score:
                    #     best_score = loss
                    #     print 'update to new best_score:', best_score
                    #     best_weight = strumodel.get_weights()
                    #     strumodel.save_weights(best_weightspath,overwrite = 1)
                    # elif loss - best_score  > best_score * tolerance: 
                    #     strumodel.set_weights(best_weight)
                    # print('weights have been reset to best_weights!')

                    dd.io.save(paramspath, classparams, compression='zlib')                      
                    json_string = strumodel.to_json()
                    open(arctecurepath, 'w').write(json_string)
                    strumodel.save_weights(weightspath,overwrite = 1)
             
                ndim = 1
                if show_progress == 1:
                    testingbatch = BatchData[ndim:ndim+1,...]
                    if use_weighted == 1:
                        testinglabel = strumodel.predict({'input': testingbatch})[0][0,:,:]
                        testingTrue = np.reshape(BatchLabel[1,...],(label_channels, labelpatchsize, labelpatchsize))[0,:,:]
                    else:
                        testinglabel = strumodel.predict({'input': testingbatch})[0]
                        testingTrue = np.reshape(BatchLabel[1,...], (label_channels, labelpatchsize, labelpatchsize) )[0,:,:]

                    plt.subplot(1,3,1)
                    plt.imshow(np.reshape(testingTrue,(labelpatchsize, labelpatchsize)))
                    plt.subplot(1,3,2)
                    plt.imshow(np.reshape(testinglabel,(labelpatchsize, labelpatchsize)))
                    plt.subplot(1,3,3)
                    plt.imshow(testingbatch[0,1,...])
                    plt.show()
# -*- coding: utf-8 -*-

