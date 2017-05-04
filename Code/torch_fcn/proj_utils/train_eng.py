# -*- coding: utf-8 -*-
# haha
import numpy as np
import os
import sys
from IPython.display import clear_output
from .Extractor import FcnExtractor
from .local_utils import batchflow, Indexflow
from .data_augmentor import ImageDataGenerator

import matplotlib.pyplot as plt

from .torch_utils import to_device

from .loss_fun import get as get_creteria
from . import generic_utils

import deepdish as dd

import torch
try:
    from visdom import Visdom
except:
    print('Better install visdom')

def get_wight_mask(Y_batch, params = None):
    label = Y_batch[:,0:-1,:,:]
    unchanged = params.get("unchanged", False)
    if unchanged:
        return Y_batch
    if params is not None:
        mean_label = np.mean(np.mean(label, axis = -1, keepdims=True), axis=-2, keepdims=True)
        mask = params['beta']*label + params['alpha']*mean_label
        Y_batch[:,-1::,:,:] = mask
        return Y_batch
    else:
        return Y_batch

def get_mean_std(StruExtractor,meanstd):
    if meanstd == 1:
        thismean, thisdev = StruExtractor.calMeanDevPts()
    if meanstd == 2:
        thismean, thisdev = StruExtractor.calMeanDevPts()
        thismean.fill(np.mean(thismean))
        thisdev.fill(np.mean(thisdev))
    if meanstd == 0:
        thismean = np.zeros((1,))
        thisdev = np.ones((1,))
    return thismean, thisdev

def get_validation(validation_params, img_channels=3, label_channels = 1):
    vpsize = validation_params['patchsize']
    lvpsize = validation_params['labelpatchsize']
    
    validation_extractor = FcnExtractor(validation_params)
    validationMatinfo = validation_extractor.getMatinfo_volume()
    datainfo = validationMatinfo['datainfo']
    Totalnum = datainfo['Totalnum']
    valid_batch = np.zeros((Totalnum, vpsize, vpsize, validation_params['channel']))
    valid_label = np.zeros((Totalnum, lvpsize, lvpsize, validation_params['label_channel']))

    validation_extractor.getOneDataBatch_stru(np.arange(Totalnum), thisbatch=valid_batch, thislabel=valid_label)

    valid_label = np.transpose(valid_label, (0, 3,1,2))
    valid_batch = np.transpose(valid_batch, (0, 3,1,2))

    del validationMatinfo
    return valid_batch, valid_label

def train_blocks(params, args=None):
    name = params['name']
    strumodel = params['strumodel']
    optimizer = params['optimizer']
    creteria  = get_creteria(params['creteria'])

    modelfolder = params['modelfolder']

    StruExtractor = params['StruExtractor']
    weight_params = params['weight_params']

    best_score = params['best_score']
    tolerance  = params['tolerance']
    worseratio = params['worseratio']
    classparams = params['classparams']
    # params for data augmentation
    number_repeat       = getattr(args, 'number_repeat', 1)
    elastic             = getattr(args, 'elastic', 1)
    elastic_label       = getattr(args, 'elastic_label', True)
    transform_label     = getattr(args, 'transform_label', True)

    rotation_range      = getattr(args, 'rotation_range', 180)
    width_shift_range   = getattr(args, 'width_shift_range', 0.)
    height_shift_range  = getattr(args, 'height_shift_range', 0.)
    channel_shift_range = getattr(args, 'channel_shift_range', False)

    meanstd = 0
    count_ = 0
    clear_freq = 15

    if args.use_validation:
        validation_params = params.get("validation_params", None)
        validation_batch_size = params.get('validation_batch_size',4)
        print('Start getting validation images.')
        valid_batch, valid_label = get_validation(validation_params, args.img_channels, args.label_channels)
        print('Finish getting validation images.')

    mydata_genrator = ImageDataGenerator(featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range= rotation_range,
                                    width_shift_range= width_shift_range,
                                    height_shift_range= height_shift_range,
                                    shear_range=0.,
                                    zoom_range=0.,
                                    channel_shift_range=0.,
                                    fill_mode='reflect',
                                    cval=0.,
                                    horizontal_flip=True,
                                    vertical_flip= True,
                                    rescale=None,
                                    preprocessing_function=None,
                                    elastic = elastic,
                                    elastic_label = elastic_label,
                                    transform_label = transform_label,
                                    number_repeat= number_repeat, 
                                    dim_ordering='default')

    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)
    modelDict = {}
    paramspath = os.path.join(modelfolder, 'params.h5')
    dd.io.save(paramspath, classparams, compression='zlib') 

    utilpath = os.path.join(modelfolder, 'utils.h5')

    weightspath = os.path.join(modelfolder,'weights.pth')
    best_weightspath = os.path.join(modelfolder,'best_weights.pth')
    util_dict = {}
    if args.reuse_weigths == 1:
        if os.path.exists(utilpath):
            try:
                util_dict = dd.io.load(utilpath)
                best_score = util_dict['acc_score']
            except:
                pass
            #best_weights_dict = torch.load(best_weightspath)
            #best_score = best_weights_dict.get('acc_score', best_score)
            #strumodel.load_state_dict(best_weights_dict['weights'])  # 12)
            #print('reload weights from {}, with score {}'.format(best_weightspath, best_score))
        if os.path.exists(weightspath):
            #print(weightspath)
            weights_dict = torch.load(weightspath,map_location=lambda storage, loc: storage)
            strumodel.load_state_dict(weights_dict)# 12)
        print('reload weights from {}, last best score {}'.format(weightspath, best_score))

    Matinfo = StruExtractor.getMatinfo_volume() # call this function to generate nece info
    datainfo = Matinfo['datainfo']

    thisbatch = np.zeros((args.chunknum,args.patchsize, args.patchsize, args.img_channels))
    thislabel = np.zeros((args.chunknum,args.patchsize, args.patchsize, args.label_channels))
    display_count = 0
    batch_count = 0
    model_dict = {}
    steps, vals = [], []
    for epochNumber in range(args.maxepoch):

        if np.mod(epochNumber+1, args.refershfreq) == 0 and epochNumber!=0:
            Matinfo = StruExtractor.getMatinfo_volume() #call this function to generate nece info
            datainfo = Matinfo['datainfo']
        Totalnum = datainfo['Totalnum']

        progbar = generic_utils.Progbar(Totalnum*args.number_repeat)
        chunkidx = 0
        total_chunks = (Totalnum + args.chunknum -1 )/args.chunknum
        for thisInd in Indexflow(Totalnum, args.chunknum):
            thisnum = len(thisInd)
            StruExtractor.getOneDataBatch_stru(thisInd, thisbatch[0:thisnum,:], thislabel[0:thisnum,:])
            BatchData  = thisbatch[0:thisnum,:].astype(np.float32)
            BatchLabel = thislabel[0:thisnum,:].astype(np.float32)

            #if nb_class == 2 and  args.labelpatchsize == 1:
            #    BatchLabel = np.concatenate([BatchLabel, 1- BatchLabel], axis = -1)
            #---------------Train your model here using BatchData------------------
            #BatchData = np.reshape(BatchData, (-1, args.patchsize, args.patchsize, args.img_channels ))
            #BatchLabel = np.reshape(BatchLabel, (-1, args.patchsize, args.patchsize, args.label_channels ))

            BatchLabel = np.transpose(BatchLabel, (0, 3,1,2))
            BatchData  = np.transpose(BatchData, (0, 3,1,2))
    
            chunkidx += 1
            print('Training--Epoch--%d----chunkId--%d-----BatchCount--%d', (epochNumber, chunkidx, batch_count))
            for X_batch, Y_batch in mydata_genrator.flow(BatchData, [BatchLabel], args.batch_size):
                batch_count += 1
                strumodel.train()
                optimizer.zero_grad()

                pred = strumodel(X_batch)
                if args.use_weighted == 0:
                    loss  =  creteria(to_device(Y_batch[:,0:-1,:,:], pred) ,pred)
                else:
                    Y_batch = get_wight_mask(Y_batch, weight_params)
                    loss  =  creteria(to_device(Y_batch, pred) ,pred)
                loss_val = loss.data.cpu().numpy().mean()
                assert not np.isnan(loss_val) ,"nan error"
                steps.append(batch_count)
                vals.append(loss_val)
                if batch_count % args.showfre == 0:
                    display_loss(steps, vals, plot=None, name= name)
                    steps[:] = []
                    vals[:]  = []

                loss.backward()
                optimizer.step()
                
                if args.show_progress == 1 and batch_count % args.showfre == 0:
                    display(strumodel, X_batch, Y_batch, args.patchsize,args.label_channels, 
                            args.use_weighted, name = name)

            progbar.add(BatchData.shape[0]*args.number_repeat, values = [("train loss", loss.cpu().data.numpy())])

            if np.mod(batch_count, args.savefre) == 0:
                if args.use_validation and np.mod(batch_count, args.validfreq) == 0:
                    valid_pred = strumodel.predict(valid_batch, batch_size = args.batch_size)
                    if args.use_weighted == 0:
                        valid_loss  =  creteria(to_device(valid_label[:,0:-1,:,:], strumodel.device_id, var =False), 
                                                to_device(valid_pred, strumodel.device_id, var =False))
                    else:
                        valid_mask  = get_wight_mask(valid_label, weight_params)
                        valid_loss  =  creteria(to_device(valid_mask, strumodel.device_id, var =False), 
                                                to_device(valid_pred, strumodel.device_id, var =False))
                        
                    print('\nTesting loss: {}, best_score: {}'.format(valid_loss, best_score))
                    if valid_loss <=  best_score:
                        best_score = valid_loss
                        print('update to new best_score: {}'.format(best_score))
                        model_dict = strumodel.state_dict()
                        util_dict['acc_score'] = best_score
                        torch.save(model_dict, best_weightspath)
                        dd.io.save(utilpath, util_dict)
                        
                        print('Save weights to: ', best_weightspath )
                        count_ = 0
                    else:
                        count_ = count_ + 1
                    #    if valid_loss - best_score  > best_score * worseratio:
                    #        strumodel.load_state_dict(model_dict['weights'])
                    #        print('weights have been reset to best_weights!')
                    if count_ >= tolerance:
                        assert 0, 'performance not imporoved for so long'
		    
                    #torch.save(model_dict, best_weightspath)
                    #print('Save weights to: ', best_weightspath)
                # save model anyway.
                model_dict = strumodel.state_dict()
                torch.save(model_dict, weightspath)
                print('Save weights to: ', weightspath)
        model_dict = strumodel.state_dict()
        torch.save(model_dict, weightspath)

def train_blocks_double(params, args=None):
    name = params['name']
    strumodel = params['strumodel']
    optimizer = params['optimizer']
    det_creteria,seg_creteria  = get_creteria(params['creteria'])

    modelfolder = params['modelfolder']

    StruExtractor = params['StruExtractor']
    weight_params = params['weight_params']

    best_score = params['best_score']
    tolerance  = params['tolerance']
    worseratio = params['worseratio']
    
    classparams = params['classparams']
    
    # params for data augmentation
    number_repeat       = getattr(args, 'number_repeat', 1)
    elastic             = getattr(args, 'elastic', 1)
    elastic_label       = getattr(args, 'elastic_label', True)
    transform_label     = getattr(args, 'transform_label', True)

    rotation_range      = getattr(args, 'rotation_range', 180)
    width_shift_range   = getattr(args, 'width_shift_range', 0.)
    height_shift_range  = getattr(args, 'height_shift_range', 0.)
    channel_shift_range = getattr(args, 'channel_shift_range', False)

    meanstd = 0
    count_ = 0
    clear_freq = 15

    if args.use_validation:
        validation_params = params.get("validation_params", None)
        validation_batch_size = params.get('validation_batch_size',4)
        print('start getting validation images.')
        valid_batch, valid_det_label,valid_seg_label = get_validation(validation_params, args.img_channels, args.label_channels)
        print('Finish getting validation images.')

    mydata_genrator = ImageDataGenerator(featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range= rotation_range,
                                    width_shift_range= width_shift_range,
                                    height_shift_range= height_shift_range,
                                    shear_range=0.,
                                    zoom_range=0.,
                                    channel_shift_range=0.,
                                    fill_mode='reflect',
                                    cval=0.,
                                    horizontal_flip=True,
                                    vertical_flip= True,
                                    rescale=None,
                                    preprocessing_function=None,
                                    elastic = elastic,
                                    elastic_label = elastic_label,
                                    transform_label = transform_label,
                                    number_repeat= number_repeat, 
                                    dim_ordering='default')

    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)
    modelDict = {}
    paramspath = os.path.join(modelfolder, 'params.h5')
    dd.io.save(paramspath, classparams, compression='zlib') 

    weightspath = os.path.join(modelfolder,'weights.pth')
    best_weightspath = os.path.join(modelfolder,'best_weights.pth')
    #best_score = 0

    if args.reuse_weigths == 1:
        if os.path.exists(best_weightspath):
            best_weights_dict = torch.load(best_weightspath)
            best_score = best_weights_dict.get('acc_score', best_score)
            #strumodel.load_state_dict(best_weights_dict['weights'])  # 12)
            #print('reload weights from {}, with score {}'.format(best_weightspath, best_score))
        if os.path.exists(weightspath):
            #print(weightspath)
            weights_dict = torch.load(weightspath,map_location=lambda storage, loc: storage)
            strumodel.load_state_dict(weights_dict['weights'])# 12)
        print('reload weights from {}, last best score {}'.format(weightspath, best_score))

    Matinfo = StruExtractor.getMatinfo_volume() # call this function to generate nece info
    datainfo = Matinfo['datainfo']

    thisbatch    = np.zeros((args.chunknum,args.patchsize, args.patchsize, args.img_channels))
    thisDetLabel = np.zeros((args.chunknum,args.patchsize, args.patchsize, args.label_channels))
    thisSegLabel = np.zeros((args.chunknum,args.patchsize, args.patchsize, args.label_channels))

    display_count = 0
    batch_count = 0
    model_dict = {}
    steps, vals, det_vals, seg_vals = [], [], [], []
    for epochNumber in range(args.maxepoch):
        if np.mod(epochNumber+1, args.refershfreq) == 0 and epochNumber!=0:
            Matinfo = StruExtractor.getMatinfo_volume() #call this function to generate nece info
            datainfo = Matinfo['datainfo']
        Totalnum = datainfo['Totalnum']

        progbar = generic_utils.Progbar(Totalnum*args.number_repeat)
        chunkidx = 0
        total_chunks = (Totalnum + args.chunknum -1 )/args.chunknum
        for thisInd in Indexflow(Totalnum, args.chunknum):
            thisnum = len(thisInd)
            StruExtractor.getOneDataBatch_stru(thisInd, thisbatch[0:thisnum,:], thisDetLabel[0:thisnum,:],thisSegLabel[0:thisnum,:])
            BatchData = thisbatch[0:thisnum,:].astype(np.float32)
            BatchDetLabel = thisDetLabel[0:thisnum,:].astype(np.float32)
            BatchSegLabel = thisSegLabel[0:thisnum,:].astype(np.float32)

            #if nb_class == 2 and  args.labelpatchsize == 1:
            #    BatchLabel = np.concatenate([BatchLabel, 1- BatchLabel], axis = -1)
            #---------------Train your model here using BatchData------------------
            BatchData = np.reshape(BatchData, (-1, args.patchsize, args.patchsize, args.img_channels ))
            BatchDetLabel = np.reshape(BatchDetLabel, (-1, args.patchsize, args.patchsize, args.label_channels ))
            BatchSegLabel = np.reshape(BatchSegLabel, (-1, args.patchsize, args.patchsize, args.label_channels ))

            BatchData  = np.transpose(BatchData, (0, 3,1,2))
            BatchDetLabel = np.transpose(BatchDetLabel, (0, 3,1,2))
            BatchSegLabel  = np.transpose(BatchSegLabel, (0, 3,1,2))
            
            chunkidx += 1
            print('Training--Epoch--%d----chunkId--%d', (epochNumber, chunkidx))
            for X_batch, det_batch, seg_batch in mydata_genrator.flow(BatchData, [BatchDetLabel, BatchSegLabel], args.batch_size):
                batch_count += 1
                strumodel.train()
                optimizer.zero_grad()

                det_pred, seg_pred, value = strumodel(X_batch)
                if args.use_weighted == 0:
                    det_loss  =  det_creteria(to_device(det_batch[:,0:-1,:,:], strumodel.device_id, var =False) ,
                                              to_device(det_pred,strumodel.device_id, var =False))
                    seg_loss  =  seg_creteria(to_device(seg_batch[:,0:-1,:,:], strumodel.device_id, var =False),
                                              to_device(seg_pred,strumodel.device_id, var =False))
                else:
                    #we dont use weight mask for segmentation
                    det_batch = get_wight_mask(det_batch, weight_params)
                    
                    det_loss  =  det_creteria(to_device(det_batch, strumodel.device_id, var =False),
                                              to_device(det_pred,strumodel.device_id, var =False))
                    seg_loss  =  seg_creteria(to_device(seg_batch[:,0:-1,:,:], strumodel.device_id, var =False),
                                              to_device(seg_pred,strumodel.device_id, var =False))
                
                det_val = det_loss.data.cpu().numpy().mean()
                seg_val = seg_loss.data.cpu().numpy().mean()
                alpha_ =  seg_val/det_val
                #alpha_ = 1
                total_loss = float(alpha_)*det_loss +  seg_loss

                if  args.use_reinforce:
                    dou_seg = torch.cat([seg_pred[0,0], 1-seg_pred[0,0]], 2) # row*col*2
                    res_action = multinomial(dou_seg).data   # row*col*1

                    res_prob = torch.gather(seg_pred,2, res_action)[:,:,0]
                    res_mask = res_action[:,:,0]

                    gt_mask  = seg_batch[:,0,:,:]
                    res_seed_map = det_pred
                    gt_seed_map = det_batch[:,0,:,:]

                    neg_dis, dsc = seg_reward(res_seed_map, gt_seed_map, res_mask, gt_mask,
                                   det_thresh = 0.2,min_len= 5, radius = 8)
                    G = neg_dis + dsc

                    advantage = G - value
                    value_loss = torch.mean(0.5 * advantage.pow(2))
                    policy_loss = - torch.log(res_prob) * Variable(advantage)

                    log_prob = log_prob.gather(1, Variable(action))

                    values.append(Variable(R))
                    policy_loss = 0
                    value_loss = 0
                    R = Variable(R)
                    gae = torch.zeros(1, 1)
                    for i in reversed(range(len(rewards))):
                        R = args.gamma * R + rewards[i]
                        advantage = R - values[i]
                        value_loss = value_loss + 0.5 * advantage.pow(2)
                        # Generalized Advantage Estimataion

                        delta_t = rewards[i] + args.gamma * \
                            values[i + 1].data - values[i].data
                        gae = gae * args.gamma * args.tau + delta_t

                        policy_loss = policy_loss - \
                            log_probs[i] * Variable(gae) - 0.01 * entropies[i]

                    optimizer.zero_grad()

                    (policy_loss + 0.5 * value_loss).backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 40)

                loss_val = total_loss.data.cpu().numpy().mean()
                assert not np.isnan(loss_val) ,"nan error"
                steps.append(batch_count)
                vals.append(loss_val)
                det_vals.append(det_val)
                seg_vals.append(seg_val)
                if batch_count % args.showfre == 0:
                    display_loss(steps, [vals], plot=None, name= name+'_total_loss')
                    display_loss(steps, [det_vals], plot=None, name= name+'_det_loss')
                    display_loss(steps, [seg_vals], plot=None, name= name+'_seg_loss')
                    #display_loss(steps, [vals, det_vals, seg_vals], plot=None,
                    #             name= name, legend= ['total_loss', 'det_loss', 'seg_loss'])
                    steps[:]    = []
                    vals[:]     = []
                    det_vals[:] = []
                    seg_vals[:] = []

                total_loss.backward()
                optimizer.step()
                
                if args.show_progress == 1 and batch_count % args.showfre == 0:
                    display_double(strumodel, X_batch, det_batch, seg_batch, args.patchsize,args.label_channels, 
                            args.use_weighted, name = name)

            progbar.add(BatchData.shape[0]*args.number_repeat, values = [("train loss", loss_val)])

            if np.mod(batch_count, args.savefre) == 0:
                if args.use_validation and np.mod(batch_count, args.validfreq) == 0:
                    valid_det_pred, valid_seg_pred, valid_adv = strumodel.predict(valid_batch, batch_size = args.batch_size)
                    if args.use_weighted == 0:
                        valid_det_loss  =  det_creteria(to_device(valid_det_label[:,0:-1,:,:], valid_det_pred) ,valid_det_pred)
                        valid_seg_loss  =  seg_creteria(to_device(valid_seg_label[:,0:-1,:,:], valid_seg_pred) ,valid_seg_pred)
                    else:
                        #we dont use weight mask for segmentation
                        valid_det_label = get_wight_mask(valid_det_label, weight_params)

                        valid_det_loss  =  det_creteria(to_device(valid_det_label, pred) ,pred)
                        valid_seg_loss  =  seg_creteria(to_device(valid_seg_label[:,0:-1,:,:], valid_seg_pred) ,valid_seg_pred)

                    valid_total_loss = valid_det_loss +  valid_seg_loss

                   # valid_loss = valid_total_loss.data.cpu().numpy().mean()
                    valid_loss = valid_total_loss
                    print(type(valid_loss))
                    print('\nTesting loss: {}, best_score: {}'.format(valid_loss, best_score))
                    if valid_loss <=  best_score:
                        best_score = valid_loss
                        print('update to new best_score: {}'.format(best_score))
                        model_dict['weights'] = strumodel.state_dict()
                        model_dict['acc_score'] = best_score.cpu()
                        torch.save(model_dict, best_weightspath)
                        count_ = 0
                    else:
                        count_ = count_ + 1
                    #    if valid_loss - best_score  > best_score * worseratio:
                    #        strumodel.load_state_dict(model_dict['weights'])
                    #        print('weights have been reset to best_weights!')
                    if count_ >= tolerance:
                        assert 0, 'performance not imporoved for so long'
                    #torch.save(model_dict, best_weightspath)
                
                model_dict['weights'] = strumodel.state_dict()
                torch.save(model_dict, weightspath)
                print('Save weights to: ', weightspath )

            model_dict['weights'] = strumodel.state_dict()
            torch.save(model_dict, weightspath)

def display_loss(steps, values, plot=None, name='default', legend= None):
    if plot is None:
        plot = Visdom()
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps  = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    win = name + '_loss'
    res = plot.line(
            X = steps,
            Y=  values,
            win= win,
            update='append',
            opts=dict(title = win, legend=legend)
        )
    if res != win:
        plot.line(
            X = steps,
            Y=  values,
            win=win,
            opts=dict(title = win,legend=legend)
        )


def display(strumodel, BatchData, BatchLabel, labelpatchsize, label_channels, use_weighted, plot=None, name='default'):
    if plot is None:
        plot = Visdom()
    ndim =  0
    testingbatch = BatchData[ndim:ndim+1,...]
    testinglabel = strumodel.predict(testingbatch)[0,0,:,:]
    testingTrue  = np.reshape(BatchLabel[ndim,...],(label_channels, labelpatchsize, labelpatchsize))[0,:,:]
    
    plot.heatmap(X = testingTrue, win = name + '_GroundTruth',
           opts=dict(title = name + '_GroundTruth'))
    plot.heatmap(X = testinglabel, win = name + '_Prediction',
           opts=dict(title = name + '_Prediction'))
    plot.heatmap(X = testingbatch[0,1,...], win = name + '_Original Image',
           opts=dict(title = name + '_Original Image'))

def display_double(strumodel, BatchData, BatchDetLabel, BatchSegLabel, labelpatchsize, 
                   label_channels, use_weighted, plot=None, name='default'):
    if plot is None:
        plot = Visdom()
    ndim =  0
    testingbatch = BatchData[ndim:ndim+1,...]
    testingDetlabel,testingSeglabel = strumodel.predict(testingbatch)
    testingDetlabel = testingDetlabel[0,0,:,:]
    testingSeglabel = testingSeglabel[0,0,:,:]

    testingDetTrue  = np.reshape(BatchDetLabel[ndim,...],(label_channels, labelpatchsize, labelpatchsize))[0,:,:]
    testingSegTrue  = np.reshape(BatchSegLabel[ndim,...],(label_channels, labelpatchsize, labelpatchsize))[0,:,:]

    plot.heatmap(X = testingDetTrue, win = name + '_DetGroundTruth',
           opts=dict(title = name + '_DetGroundTruth'))
    plot.heatmap(X = testingDetlabel, win = name + '_DetPrediction',
           opts=dict(title = name + '_DetPrediction'))
    plot.heatmap(X = testingSegTrue, win = name + '_SegGroundTruth',
           opts=dict(title = name + '_SegGroundTruth'))
    plot.heatmap(X = testingSeglabel, win = name + '_SegPrediction',
           opts=dict(title = name + '_SegPrediction'))
    plot.heatmap(X = testingbatch[0,1,...], win = name + '_Original Image',
           opts=dict(title = name + '_Original Image'))


