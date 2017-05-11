import os
import time
import cv2
import random
import numpy as np

from scipy import ndimage
import scipy.io as sio
from scipy.io import loadmat

from  skimage.feature import peak_local_max
from  skimage import data, color, io, img_as_float
from timeit import default_timer as timer

from PIL import Image
import matplotlib.pyplot as plt

import deepdish as dd
import shutil

from .post_processing import *
from .local_utils import *

def get_seed_name(step, threshhold, min_len, resultmask=""):
    if resultmask is not '': # simply for legancy consideration
        resultmask = resultmask + '_'
    name  =(resultmask + 's_' + '{:02d}'.format(step) + '_t_'   + '{:01.02f}'.format(threshhold) \
             + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name

def get_seg_seed_name(step,threshhold,seg_thresh, min_len,resultmask=""):
    if resultmask is not '':
        resultmask = resultmask + '_'
    name  =(resultmask + 's_' + '{:02d}'.format(step) + '_t_'   + '{:01.02f}'.format(threshhold) \
            +'_st_'   + '{:01.02f}'.format(seg_thresh) + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name
    
class runtestImg(object):
    def __init__(self, kwargs):
        self.ImgDir = ''
        self.savefolder = ''
        self.ImgExt = ['.jpg']
        self.resultmask = ''
        self.patchsize = 48
        self.labelpatchsize = 48
        self.model = None
        self.stridepool = [self.labelpatchsize]
        self.arctecurepath = ''
        self.Probrefresh = 1
        self.Seedrefresh = 1
        self.thresh_pool = np.arange(0.05, 1, 0.05)
        self.lenpool = [8]
        self.show = 0
        self.batch_size = 256
        self.resizeratio = 1
        self.test_type = 'fcn'
        self.ol_folder = 'masked'
        self.testingInd = None
        self.steppool = [1]
        self.showseg = 0
        self.showseed =0
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #assert self.model != None,"Model does not exist!"

    def shortCut_FCN(self, inputfile, model=None,windowsize = 1000, board = 40,fixed_window= False, step_size=None,
                     batch_size= 1,**kwargs):
        # receive one image and run the routine job

        if type(self.model) is list:
            for thismodel in self.model:
                double_output.append(self.get_output(img, model=thismodel, windowsize=param.windowsize,
                                                     batch_size=param.batch_size, fixed_window=param.fixed_window,
                                                     step_size=None))
        else:
            double_output = self.get_output(img, model=self.model, windowsize=param.windowsize,
                                            batch_size=param.batch_size, fixed_window=param.fixed_window,
                                            step_size=None)

        VotingMap_list = [np.squeeze(out) for out in double_output]
        if len(VotingMap_list) == 1:
            VotingMap_list.append(VotingMap_list[0])

        orgimg = imread(inputfile) if isinstance(inputfile, str) else inputfile
        if  model == None:
            model = self.model
        if len(orgimg.shape) == 2:
           orgimg =  orgimg.reshape(orgimg.shape[0],orgimg.shape[1],1)
        img = pre_process_img(orgimg, yuv = False).astype(np.float32)
        thisprediction = self.split_testing(BatchData, model=model,  windowsize   = windowsize, 
                                            board = board, batch_size = batch_size, 
                                            fixed_window = fixed_window, 
                                            step_size  = step_size,  **kwargs
                                            )
        if type(thisprediction) is list:
            VotingMap = []
            for pre in thisprediction:
                VotingMap.append(np.squeeze(pre))
        else:
            VotingMap = np.squeeze(thisprediction)
        return VotingMap
    
    def split_testing(self, img, model=None, windowsize = 500, board = 30, batch_size = 4, 
                      fixed_window= False, step_size=None, adptive_batch_size=True, **kwargs):
        '''
        This function is used to do prediction for fcn and it's variation.
        img should be of size (row, col,channel)
        '''
        assert len(img.shape) == 3, "Input image must be have three dimension"
        img = np.transpose(img, (2, 0, 1))
        if model == None:
            model = self.model
        PatchDict = split_img(img, windowsize=windowsize, board = board, fixed_window= fixed_window,step_size=step_size)
        output = None
        all_keys = PatchDict.keys()
        for this_size in all_keys:
            #print(this_size)
            BatchData, org_slice_list, extract_slice_list = PatchDict[this_size]
            if adptive_batch_size == True:
                old_volume = batch_size * windowsize * windowsize
                new_bs = int(np.floor( 1.0*old_volume/np.prod(this_size)))
            else:
                new_bs = batch_size
            #print(new_bs, BatchData.shape[0])
            if self.test_type in ['fcn', 'double_output']:
                thisprediction  =  model.predict(BatchData, batch_size = new_bs)
            else:
                raise Exception('Unknown prediction mode {}'.format(self.test_type))
            if type(thisprediction) != list:
               thisprediction = [thisprediction]
            if output is None:
                output = [np.zeros(img.shape[1:]) for _ in range(len(thisprediction))]

            for odx, pred in enumerate(thisprediction) :
                for idx, _ in enumerate(org_slice_list):
                    org_slice = org_slice_list[idx]
                    extract_slice = extract_slice_list[idx]
                    output[odx][org_slice[0], org_slice[1]] += np.squeeze(pred[idx])[extract_slice[0], extract_slice[1]]
        if len(output) == 1:
            output = output[0]
        return output
    def get_coordinate(self, inputfile, model=None,windowsize = 1000, board = 40,fixed_window= False,step_size=None,
                       probmap=None, threshhold = 0.1, batch_size= 1,min_len=5, **kwargs):
        if probmap is None:
            probmap = self.shortCut_FCN(inputfile = inputfile, model=model,windowsize = windowsize, board = board,
                        batch_size= batch_size, fixed_window= fixed_window,step_size=step_size, **kwargs)
        voting_map = probmap.copy()

        voting_map[voting_map < threshhold*np.max(voting_map[:])] = 0
        coordinates = peak_local_max(voting_map, min_distance= min_len, indices = True) # N by 2
        if coordinates.size == 0:
           coordinates = np.asarray([])
        return coordinates, probmap

    def get_segmentation(self, inputfile=None, model=None, windowsize = 1000, board = 40,fixed_window= False,step_size=None, returnImg = True,
                         probmap=None, coordinates=None, thresh_seg= 0.5, batch_size= 1,get_contour=True,tiny_size=100, **kwargs):
        if isinstance(inputfile, str):
           inputfile = imread(inputfile)
 
        if probmap is None:
            probmap = self.shortCut_FCN(inputfile = inputfile, model=model,windowsize = windowsize, board = board,
                        batch_size= batch_size, fixed_window= fixed_window,step_size=step_size, **kwargs)

        seg_prob = probmap.copy()
        
        label_img = overal_watershed(seg_prob, thresh_water = 0.5, thresh_seg = thresh_seg, 
                                     ratio = 0.3, dist_f = np.median, tiny_size=tiny_size)
        start = timer()
        class_label, new_label, new_coord = residual_markers(label_img, coordinates)
        end = timer()
        print('time for residual marker: {}',format(end-start))
        marker_map = np.zeros_like(seg_prob).astype(np.uint8)
        marker_map[new_coord[:, 0], new_coord[:, 1]] = 1
        residual_label = overal_watershed_marker(seg_prob*(new_label>0), marker_map, thresh_seg= thresh_seg, ratio = 0.05, dist_f = np.median, tiny_size=tiny_size)
        final_label = class_label + (residual_label + np.max(class_label))*( residual_label !=0 )
        
        #final_label = label_img
        #new_coord = combine_markers(label_img, coordinates)
        #marker_map = np.zeros_like(seg_prob).astype(np.uint8)
        #marker_map[new_coord[:, 0], new_coord[:, 1]] = 1
        #label_img = overal_watershed_marker(label_img>0, marker_map, thresh_seg= thresh_seg)
        return label2contour(final_label,org=inputfile, linewidth=3,returnImg=returnImg)

    def runtesting(self, **params):
        
        if self.test_type in ['double_output']:
            self.folder_seg(**params)
        else:
            self.folder_det(**params)
             
    def folder_det(self, **kwargs):
        param = myobj()
        param.windowsize = 500
        param.batch_size = 8
        param.fixed_window = False
        param.step_size = None

        for key in kwargs:
            setattr(param, key, kwargs[key])
        thismodel = self.model if type(self.model) is not list else self.model[0]
        
        imglist, imagenamelist = getfilelist(self.ImgDir, self.ImgExt)
        #BatchData = np.zeros((self.batchsize, self.patchsize*self.patchsize*self.channel))
        print(self.ImgDir + self.ImgExt[0])
        for imgindx in range(0,len(imglist)):
            print('processing image {ind}'.format(ind = imgindx))
            if os.path.isfile(imglist[imgindx]):
              orgimg = imread(imglist[imgindx])
              if len(orgimg.shape) == 2:
                 orgimg =  orgimg.reshape(orgimg.shape[0],orgimg.shape[1],1)
                 orgimg = np.concatenate((orgimg,orgimg,orgimg),axis = 2)
            imgname = imagenamelist[imgindx]
            #resultDictPath = os.path.join(self.savefolder,  imgname + '_'+ self.resultmask + '.h5')
            #resultDictPath_mat = os.path.join(self.savefolder, imgname + '_'+ self.resultmask + '.mat')
            resultDictPath = os.path.join(self.savefolder, imgname + '.h5')
            resultDictPath_mat = os.path.join(self.savefolder, imgname + '.mat')

            if os.path.isfile(resultDictPath):
               resultsDict = dd.io.load(resultDictPath)
            else:
               resultsDict = {}
            orgRowSize , orgColSize = orgimg.shape[0], orgimg.shape[1]
            for step in self.steppool:
                self.step = step
                #print('step is not used in fcn, if you want to use, please modify the folderTesting function. \n')
                votingmapname    = 's_' + '{:02d}'.format(self.step) + '_vm'
                voting_time_name = 's_' + '{:02d}'.format(self.step) + '_time'
                if self.Probrefresh or votingmapname not in resultsDict.keys():
                    # first pad the image to make it dividable by the labelpatchsize
                    votingStarting_time = time.time()
                    img = pre_process_img(orgimg, yuv = False)

                    VotingMap = np.squeeze(self.split_testing(img, model = thismodel, windowsize = param.windowsize, batch_size = param.batch_size,
                                          fixed_window= param.fixed_window, step_size= None))
                    votingEnding_time = time.time()
                    resultsDict[voting_time_name] = votingEnding_time - votingStarting_time
                    print(resultsDict[voting_time_name])
                    resultsDict[votingmapname] = np.copy(VotingMap)
                else:
                    VotingMap = resultsDict[votingmapname]
                # display the map if you want
                if self.showmap:
                  plt.imshow(VotingMap, cmap = 'hot')
                  plt.show()
                for threshhold  in self.thresh_pool:
                    for min_len in self.lenpool:
                        thisStart = time.time()
                        localseedname = get_seed_name(self.step, threshhold, min_len)

                        localseedtime = get_seed_name(self.step, threshhold, min_len) + '_time'

                        if self.Seedrefresh or localseedname not in resultsDict.keys():
                           VotingMap[VotingMap < threshhold*np.max(VotingMap[:])] = 0

                           coordinates = peak_local_max(VotingMap, min_distance= min_len, indices = True) # N by 2,

                           #aa = extrema(VotingMap, kernel_size=(min_len,min_len))

                           if coordinates.size == 0:
                               coordinates = np.asarray([])
                               print("you have empty coordinates for img:{s}".format(s=imgname))
                           thisEnd = time.time()
                           resultsDict[localseedname] = coordinates
                           resultsDict[localseedtime] = thisEnd - thisStart +  resultsDict[voting_time_name]
                           if self.showseed:
                              if coordinates.size > 0:
                                 plt.figure('showseeds')
                                 plt.imshow(orgimg)
                                 plt.plot(coordinates[:,1], coordinates[:,0], 'r.')
                                 plt.show()

            dd.io.save(resultDictPath, resultsDict, compression=None)    #compression='zlib'
            sio.savemat(resultDictPath_mat, resultsDict)

    def folder_seg(self, **kwargs):
        param = myobj()
        param.windowsize = 500
        param.batch_size = 8
        param.fixed_window = False
        param.step_size = None

        for key in kwargs:
            setattr(param, key, kwargs[key])

        imglist, imagenamelist = getfilelist(self.ImgDir, self.ImgExt)

        for imgindx in range(0,len(imglist)):
            print('processing image {ind}'.format(ind = imgindx))
            if os.path.isfile(imglist[imgindx]):
              orgimg = imread(imglist[imgindx])
              if len(orgimg.shape) == 2:
                 orgimg =  orgimg.reshape(orgimg.shape[0],orgimg.shape[1],1)
                 orgimg = np.concatenate((orgimg,orgimg,orgimg),axis = 2)
            imgname = imagenamelist[imgindx]
            resultDictPath = os.path.join(self.savefolder,  imgname + '_'+ self.resultmask + '.h5')
            resultDictPath_mat = os.path.join(self.savefolder, imgname + '_'+ self.resultmask + '.mat')
            if os.path.isfile(resultDictPath):
               try:
                  resultsDict = dd.io.load(resultDictPath)
               except:
                  resultsDict = {}
            else:
               resultsDict = {}
            orgRowSize , orgColSize = orgimg.shape[0], orgimg.shape[1]
            for step in self.steppool:
                self.step = step
                print('step is not used in fcn, if you want to use, please modify the folderTesting function. \n')
                votingmapname    = 's_' + '{:02d}'.format(self.step) + '_vm'
                voting_time_name = 's_' + '{:02d}'.format(self.step) + '_time'
                if self.Probrefresh or votingmapname not in resultsDict.keys():
                   # first pad the image to make it dividable by the labelpatchsize
                    votingStarting_time = time.time()

                    img = pre_process_img(orgimg, yuv = False)
                    double_output = []
                    
                    if type(self.model) is list:
                        for thismodel in self.model:
                            double_output.append(self.split_testing(img, model = thismodel,windowsize = param.windowsize,
                                                 batch_size = param.batch_size, fixed_window= param.fixed_window, step_size= None))
                    else:
                        double_output = self.split_testing(img, model = self.model,windowsize = param.windowsize,
                                                        batch_size = param.batch_size, fixed_window= param.fixed_window, step_size= None)
                        
                    VotingMap_list = [np.squeeze(out) for out in double_output]
                    if len(VotingMap_list) == 1:
                        VotingMap_list.append(VotingMap_list[0])
                    votingEnding_time = time.time()
                    resultsDict[voting_time_name] = votingEnding_time - votingStarting_time
                    print(resultsDict[voting_time_name])
                    resultsDict[votingmapname] = np.copy(VotingMap_list)
                else:
                    VotingMap_list = resultsDict[votingmapname]
                # display the map if you want
                if self.showmap:
                  plt.imshow(VotingMap_list[1], cmap = 'hot')
                  plt.show()
                VotingMap_det = VotingMap_list[0]
                VotingMap_seg = VotingMap_list[1]

                for threshhold  in self.thresh_pool:
                    for min_len in self.lenpool:
                        thisStart = time.time()
                        localseedname = get_seed_name(self.step, threshhold, min_len)
                        if self.Seedrefresh or localseedname not in resultsDict.keys():
                            VotingMap_det[VotingMap_det < threshhold*np.max(VotingMap_det[:])] = 0

                            coordinates = peak_local_max(VotingMap_det, min_distance= min_len, indices = True) # N by 2
                            if coordinates.size == 0:
                                coordinates = np.asarray([])
                                print("you have empty coordinates for img:{s}".format(s=imgname))

                            resultsDict[localseedname] = coordinates

                            if self.showseed:
                                if coordinates.size > 0:
                                    plt.figure('showseeds')
                                    plt.imshow(orgimg)
                                    plt.plot(coordinates[:,1], coordinates[:,0], 'r.')
                                    plt.show()
                            for seg_thresh in self.seg_thresh_pool:        
                                localsegname = get_seg_seed_name(self.step, threshhold,seg_thresh, min_len)
                                localsegtime = get_seg_seed_name(self.step, threshhold,seg_thresh, min_len) + '_seg_time'
                                marked_img, contours  = self.get_segmentation(inputfile=orgimg, probmap=VotingMap_seg, coordinates=coordinates, 
                                                                            threshhold = seg_thresh, returnImg = self.showseg)

                                resultsDict[localsegname] = contours
                                thisEnd = time.time()
                                resultsDict[localsegtime] = thisEnd - thisStart +  resultsDict[voting_time_name]
                                if self.showseg:
                                    plt.figure('showseg')
                                    plt.imshow(marked_img)
                                    plt.show()
            dd.io.save(resultDictPath, resultsDict, compression=None)    #compression='zlib'
            sio.savemat(resultDictPath_mat, resultsDict)

    def seg_img(self, inputfile, min_len, threshhold, seg_thresh, **kwargs):
        '''
        Given an image or image name, return the contours segmentation of the images for a given set of 
        parameters.
        
        :param inputfile: 
        :param min_len: 
        :param threshhold: 
        :param seg_thresh: 
        :param kwargs: 
        :return: 
        '''
        if isinstance(inputfile, str):
           inputfile = imread(inputfile)
           img = pre_process_img(inputfile, yuv=False)
        else:
            img = inputfile
        param = myobj()
        param.windowsize = 500
        param.batch_size = 8
        param.fixed_window = False
        param.step_size = None

        for key in kwargs:
            setattr(param, key, kwargs[key])
        double_output = []
        if type(self.model) is list:
            for thismodel in self.model:
                double_output.append(self.split_testing(img, model=thismodel, windowsize=param.windowsize,
                                                        batch_size=param.batch_size,
                                                        fixed_window=param.fixed_window, step_size=None))
        else:
            double_output = self.split_testing(img, model=self.model, windowsize=param.windowsize,
                                               batch_size=param.batch_size, fixed_window=param.fixed_window,
                                               step_size=None)

        VotingMap_list = [np.squeeze(out) for out in double_output]
        if len(VotingMap_list) == 1:
            VotingMap_list.append(VotingMap_list[0])
        VotingMap_det = VotingMap_list[0]
        VotingMap_seg = VotingMap_list[1]

        VotingMap_det[VotingMap_det < threshhold * np.max(VotingMap_det[:])] = 0

        coordinates = peak_local_max(VotingMap_det, min_distance=min_len, indices=True)  # N by 2
        if coordinates.size == 0:
            coordinates = np.asarray([])
            print("you have empty coordinates for img")

        marked_img, contours = self.get_segmentation(inputfile=inputfile, probmap=VotingMap_seg,
                                                     coordinates=coordinates,
                                                     threshhold=seg_thresh)
        return marked_img, contours, 

    def printMask(self, img, mask, savepath=None,alpha = 0.618):
        '''
        overlay mask onto the image
        '''
        #img = img_as_float(data.camera())
        rows, cols = img.shape[0:2]
        # Construct a colour image to superimpose
        color_mask = np.zeros((rows, cols, 3))
        color_mask[mask == 1] = [5,119,72]

        if len(img.shape) == 2:
           img_color = np.dstack((img, img, img))
        else:
           img_color = img
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(color_mask)

        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)
        # Display the output
        f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                          subplot_kw={'xticks': [], 'yticks': []})
        ax0.imshow(img, cmap=plt.cm.gray)
        ax1.imshow(color_mask)
        ax2.imshow(img_masked)
        plt.show()

        img_masked = np.asarray((img_masked/np.max(img_masked) ) * 255, dtype = np.uint8)
        if savepath is not None:
           im = Image.fromarray(img_masked)
           im.save(savepath)
        return img_masked

    def printCoord(self, Img=None, coordinates=None, savepath=None, **kwargs):
        '''
        print the coordinates onto the Image
        Img should be (row, col,channel)
        coordinates: should be (n,2) with (row, col) order
        '''
        assert Img!=None and coordinates != None, 'input field not valid'
        param = myobj()
        param.linewidth = 7
        param.color = [1,0,0] #[5,119,72]
        param.alpha = 0.85
        for key in kwargs:
            setattr(param, key, kwargs[key])

        dot_mask  = np.zeros(Img.shape[0:2])
        if coordinates.size != 0:
           dot_mask[coordinates[:, 0], coordinates[:, 1]] = 1

        # dialte the image based on linewidth
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(param.linewidth,param.linewidth))
        dot_mask = cv2.dilate(dot_mask,se)

        #imshow(dot_mask)
        overlaiedRes =  overlayImg(Img, dot_mask , print_color = param.color, linewidth = 1, alpha = param.alpha)

        im_masked = Image.fromarray(overlaiedRes)
        if savepath:
           im_masked.save(savepath)
        return overlaiedRes

    def printMasks(self,ImgDir= None,ImgExt = None, eva_info= None, threshhold = 0.15):
        if ImgExt is None:
            ImgExt = self.ImgExt
        if ImgDir is None:
            ImgDir = self.ImgDir

        imglist_, imagenamelist_ = getfilelist(ImgDir, ImgExt)
        if eva_info is not None:
           valid_ind = eva_info['ind']
        else:
           valid_ind = range(0, len(imglist_))

        imglist = [imglist_[i] for i in valid_ind]
        imagenamelist = [imagenamelist_[i] for i in valid_ind]

        ol_folder = os.path.join(self.savefolder, self.ol_folder)
        if  os.path.exists(ol_folder):
           shutil.rmtree(ol_folder)
        os.makedirs(ol_folder)
        for imgindx in range(0,len(imglist)):
            print('overlay image {ind}'.format(ind = imgindx))

            assert os.path.isfile(imglist[imgindx]), 'image does not exist!'
            thisimg = imread(imglist[imgindx])

            imgname = imagenamelist[imgindx]

            savepath = os.path.join(ol_folder, imgname + '_ol.bmp')

            resultDictPath = os.path.join(self.savefolder,imgname + '_'+ self.resultmask + '.h5')
            if os.path.isfile(resultDictPath):
               resultsDict = dd.io.load(resultDictPath)
            votingmapname  = 's_' + '{:02d}'.format(self.step) + '_vm'
            seg_map = resultsDict[votingmapname]

            binary_map = (seg_map > np.max(seg_map) * threshhold).astype(np.float32)
            mask_org = process_mask_paramisum(binary_map)

            mask = np.asarray(process_mask_paramisum(mask_org), dtype = thisimg.dtype)

            thisimg = imresize_shape(thisimg, mask.shape)

            thisimg = RGB2GRAY(thisimg)
            thisimg = np.asarray(thisimg, dtype= np.uint8)
            mask = np.asarray(mask, dtype= np.uint8)
            self.printMask(img = thisimg,  mask = mask, savepath = savepath, alpha = 0.85)

    def printCoords(self, eva_info= None, threshhold = 0.10, step = 1, min_len = 5):
        imglist_, imagenamelist_ = getfilelist(self.ImgDir, self.ImgExt)

        if eva_info is not None:
           valid_ind = eva_info['ind']
        else:
           valid_ind = range(0, len(imglist_))

        imglist = [imglist_[i] for i in valid_ind]
        imagenamelist = [imagenamelist_[i] for i in valid_ind]

        ol_folder = os.path.join(self.savefolder, self.ol_folder + str(threshhold).replace('.','_'))
        if  os.path.exists(ol_folder):
           #os.rmdir(ol_folder)
           shutil.rmtree(ol_folder)
        os.makedirs(ol_folder)
        for imgindx in range(0,len(imglist)):
            print('overlay image {ind}'.format(ind = imgindx))
            assert os.path.isfile(imglist[imgindx]), 'image does not exist!'
            thisimg = imread(imglist[imgindx])
            imgname = imagenamelist[imgindx]
            savepath = os.path.join(ol_folder, imgname + '_ol.bmp' )

            resultDictPath = os.path.join(self.savefolder, imgname +  '.h5')
            if os.path.isfile(resultDictPath):
               resultsDict = dd.io.load(resultDictPath)
            #votingmapname  = self.resultmask + '_s_' + '{:02d}'.format(step) + '_vm'
            #seg_map = resultsDict[votingmapname]
            localseedname = get_seed_name(step, threshhold, min_len)
            coordinates = resultsDict[localseedname]
            Return_masked = self.printCoord(Img = thisimg,  coordinates = coordinates, savepath = savepath, alpha = 0.85)
            if self.show:
               imshow(Return_masked)

    def printContours(self, eva_info= None, threshhold = 0.10, seg_thresh=0.1, step = 1, min_len = 5):
        imglist_, imagenamelist_ = getfilelist(self.ImgDir, self.ImgExt)

        if eva_info is not None:
           valid_ind = eva_info['ind']
        else:
           valid_ind = range(0, len(imglist_))

        imglist = [imglist_[i] for i in valid_ind]
        imagenamelist = [imagenamelist_[i] for i in valid_ind]

        ol_folder = os.path.join(self.savefolder, self.ol_folder + str(threshhold).replace('.','_'))
        if  os.path.exists(ol_folder):
           #os.rmdir(ol_folder)
           shutil.rmtree(ol_folder)
        os.makedirs(ol_folder)
        for imgindx in range(0,len(imglist)):
            print('overlay image {ind}'.format(ind = imgindx))
            assert os.path.isfile(imglist[imgindx]), 'image does not exist!'
            thisimg = imread(imglist[imgindx])
            imgname = imagenamelist[imgindx]
            savepath = os.path.join(ol_folder, imgname + '_ol.bmp' )

            resultDictPath = os.path.join(self.savefolder,imgname +  '.h5')
            print(resultDictPath)
            if os.path.isfile(resultDictPath):
               resultsDict = dd.io.load(resultDictPath)
            #votingmapname  = self.resultmask + '_s_' + '{:02d}'.format(step) + '_vm'
            #seg_map = resultsDict[votingmapname]

            localsegname = get_seg_seed_name(step, threshhold, seg_thresh, min_len, self.resultmask ) 
            contours = resultsDict[localsegname]
            Return_masked = markcontours(thisimg, contours, print_color = [0,0,1], linewidth = 2, alpha = 1)
            im_masked = Image.fromarray(Return_masked)
            if savepath:
               im_masked.save(savepath)

