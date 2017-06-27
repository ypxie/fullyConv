import sys
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=False'
cloudRoot  = os.path.join('..','..','..','..')
projroot = os.path.join('..','..')

kerasversion = 'keras-1'
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras'))
sys.path.insert(0, os.path.join(cloudRoot, 'Code', kerasversion,'keras','layers'))
sys.path.insert(0, '..')
sys.path.insert(0, os.path.join('..', 'proj_utils') )
from train_eng import get_mean_std
from proj_utils.local_utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from Extractor import FcnExtractor
from keras.utils import  generic_utils

from kerasOneModel import buildCellModel as buildmodel

from loss_fun import weighted_loss
from proj_utils.keras_utils import elu
activ = elu(alpha=1.0) 
last_activ = 'relu'   
from keras.optimizers import Adadelta  ,SGD, RMSprop, adam


home = os.path.join(cloudRoot, '..')
#home =  '/home/yuanpu'
#dataroot = os.path.join(cloudRoot,'DataSet','Mixture_DataSets')
dataroot = os.path.join(projroot, 'Data')
modelroot = os.path.join(projroot, 'Data', 'Model')

trainingset = 'det_lung_left' 
modelsubfolder = 'deep_det_fcn'
modelfolder = os.path.join(modelroot,trainingset,modelsubfolder)
trainingimagefolder = os.path.join(cloudRoot,'DataSet',  'Mixture_DataSets', 'TrainingData','Seperate')

init_fine_tune_weights = True

#whole_data means use all the data except lung, other numbers mean the 
# number of images u wanna take.
loop_configuration = ['whole_data', 2,5,8,11,14,17,20,23,-1]
def get_loop(loop_configuration):
    for this_config in loop_configuration:
        folder_list = []   
        if this_config == 'whole_data': 
            folder_list.append(os.path.join(trainingimagefolder, 'BoneMarrow'))
            folder_list.append(os.path.join(trainingimagefolder, 'brainTumor'))
            folder_list.append(os.path.join(trainingimagefolder, 'BreastCancer'))
            folder_list.append(os.path.join(trainingimagefolder, 'Net'))
            folder_list.append(os.path.join(trainingimagefolder, 'Phase'))
            number_list = [-1]
            weights_name = 'weights.h5'
            params_name  = 'params.h5'
            maxepoch = 100
            trainable_list= [True, True] # do you wanna fine tune the weights
        else:
            folder_list.append(os.path.join(trainingimagefolder, 'Lung'))
            number_list = [this_config]
            weights_name = 'weights' +'_'+str(this_config) + '.h5' 
            params_name  = 'params'  +'_'+str(this_config) + '.h5'
            maxepoch = 100
            trainable_list= [False, True] # do you wanna fine tune the weights
        yield folder_list, number_list , weights_name, params_name, maxepoch, trainable_list
if  __name__ == '__main__':
    
    for this_config in loop_configuration:
        folder_list = []   
        if this_config == 'whole_data': 
            folder_list.append(os.path.join(trainingimagefolder, 'BoneMarrow'))
            folder_list.append(os.path.join(trainingimagefolder, 'brainTumor'))
            folder_list.append(os.path.join(trainingimagefolder, 'BreastCancer'))
            folder_list.append(os.path.join(trainingimagefolder, 'Net'))
            folder_list.append(os.path.join(trainingimagefolder, 'Phase'))
            number_list = [-1]
            weights_name = 'weights.h5'
            params_name  = 'params.h5'
            maxepoch = 100
            trainable_list= [True, True] # do you wanna fine tune the weights
        else:
            folder_list.append(os.path.join(trainingimagefolder, 'Lung'))
            number_list = [this_config]
            weights_name = 'weights' +'_'+str(this_config) + '.h5' 
            params_name  = 'params'  +'_'+str(this_config) + '.h5'
            maxepoch = 100
            trainable_list= [False, True] # do you wanna fine tune the weights

        loss = weighted_loss(base = 'mse')
        use_weighted = 1    #means you have a weight mask in the last dimension of mask
        #loss = 'mse'
        #use_weighted = 0    #means you have a weight mask in the last dimension of mask

        # for build the model    
        img_channels = 3    
        lr = 0.00005
        #opt = adam(lr=lr, clipnorm=100)
        opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=10)
        weight_decay = 1e-8
    
        
        batch_size = 2
        meanstd = 0   # it is important to set this as 0 for FCN
        chunknum = int(320)
        maxepoch = maxepoch
        matrefresh = 1
        meanstdrefresh = 1
        refershfreq = 10
        savefre = 1  # np.mod(chunidx, savefre) == 0:

        rebuildmodel = 1
        reuseweigths = 0
        show_progress = 1 #if you want to show the testing cases.
        
        patchsize = 200
        labelpatchsize = 200
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
        decayparam['r'] = 10
        decayparam['scale'] = 5
        decayparam['decalratio'] = 0.1
        decayparam['smallweight'] = 0.05
        decayparam['use_gauss'] = 1

        decayparam['use_weighted'] = use_weighted
        decayparam['reverse_channel'] = True
        decayparam['filp_channel'] = True
        
        classparams['decayparam'] = decayparam
        classparams['w'] = 10 # weight wrt distance
        classparams['wc'] = 1 # weight wrt to class
        classparams['dilate'] = 8

        classparams['datadir'] = trainingimagefolder
        classparams['volume'] = 2
        classparams['dataExt'] = ['.png','.tif','jpg']   # the data ext
        classparams['labelExt']    = ['.mat']            # the label ext
        classparams['contourname']   = 'Contours'        # the contour name in the cell array
        classparams['labelSuffix']  = ["",'_withcontour', '_gt','_seg'] # the suffix of label

        classparams['allDictList'] = getFromFolderList(folder_list,  number_list = number_list, contourextList = classparams['labelSuffix'], 
                                                    ImgExtList = classparams['dataExt'], LabelExt = '.mat')
        if classparams['allDictList'] is None:
            continue                                            
        classparams['maxsamples']  = 1280000
        classparams['usecontour']  = 1            # this is just used to fill the cotour to get filled_img, 
        classparams['usingindx']     = 1
        classparams['pickratio']   = 0.005        # 1 means take all the pixel
        classparams['crop_patch_size']   =  None  #(0.95,0.95)
        classparams['selected_portion']   = 0.01  # 1 means take all the islet pixels
        classparams['mask_dilate'] = 30
        classparams['resizeratio'] =  [1]         #[1, 0.7]
        classparams['rotatepool']  = [0, 90, 180,270]          #[0,90]
        classparams['maximg'] = 200
        classparams['mask_thresh'] = 50
        classparams['mask_prob'] =0.1
        classparams['maxpatch'] = 20
        classparams['random_pick'] =  True        # if you wanna random pick
        
        StruExtractor = FcnExtractor(classparams)
        nb_class   = classparams['labelpatchsize']  ** 2
        
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)

        
        modelDict = {}
        modelpath = os.path.join(modelfolder, 'strumodel.h5') #should include model and other parameters
        weightspath = os.path.join(modelfolder,weights_name)
        init_weightspath = os.path.join(modelfolder,'weights.h5')
        
        arctecurepath = os.path.join(modelfolder,'arc.json')
        meanstdpath = os.path.join(modelfolder, 'meanstd.h5')
        paramspath = os.path.join(modelfolder, params_name)
        
        if not os.path.isfile(arctecurepath) or  rebuildmodel == 1:
            strumodel = buildmodel(img_channels,lr = lr, loss = loss, activ=activ, last_activ='relu',trainable_list= trainable_list)
            if reuseweigths == 1 and os.path.isfile(weightspath):
                if init_fine_tune_weights:
                    strumodel.load_weights(init_weightspath)
                else:
                    strumodel.load_weights(weightspath)
                # import h5py
                # f = h5py.File(weightspath, mode='r')
                # g = f['graph']
                # weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                # strumodel.set_weights(weights)
                # f.close()
 

        Matinfo = StruExtractor.getMatinfo_volume() # call this function to generate nece info
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
        print('finish compiling!')

        thisbatch = np.zeros((chunknum,datainfo['inputdim']))
        thislabel = np.zeros((chunknum,datainfo['outputdim']))
        update_num = 0
	past_min_loss = 10000
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
                        loss = strumodel.train_on_batch({'input': X_batch}, {'output_mask': Y_batch})
                    if type(loss) == list:
                        loss = loss[0]
                    assert not np.isnan(loss) ,"nan error"
                progbar.add(BatchData.shape[0], values = [("train loss", loss)])

                if np.mod(chunkidx, savefre) == 0:
                    if  loss <= 4 * past_min_loss:
                        past_min_loss = loss if loss < past_min_loss else past_min_loss  
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
                if  loss <= 4 * past_min_loss:
                    past_min_loss = loss if loss < past_min_loss else past_min_loss  
                    strumodel.save_weights(weightspath,overwrite = 1)
                    dd.io.save(paramspath, classparams, compression='zlib')
