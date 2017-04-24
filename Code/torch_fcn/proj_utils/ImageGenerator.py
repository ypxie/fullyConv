from __future__ import absolute_import
import numpy as np
import os, math
from PIL import Image
import deepdish as dd
from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
import scipy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import skimage, skimage.morphology
import cv2
from scipy.ndimage.interpolation import rotate
from skimage import color, measure
import scipy.ndimage
from .generic_utils import get_from_module
from .local_utils import *

try:
  from numba import jit,autojit
except:
  print('check if you  have installed numba')

from sklearn.metrics.pairwise import pairwise_distances
from random import shuffle as randshuffle

def get(identifier):
    return get_from_module(identifier, globals(), 'ImageGenerator')

def process_mask_volume(myobj,contour_mat, mask):
    [temprow,tempcol] = [mask.shape[0], mask.shape[1]]
    boundary_mask = np.zeros([temprow,tempcol])
    filled_img  = np.zeros([temprow,tempcol])
    if len(contour_mat) != 0:
        contour_mat = contour_mat.astype(int)

        xcontour = contour_mat[0,:]
        ycontour = contour_mat[1,:]

        filled_img[ycontour, xcontour] = 1
        seedCollection = contour_mat.transpose()

        [tmpmeshRowInd, tmpmeshColomnInd] = np.meshgrid(range(0,tempcol),range(0,temprow))
        tmprealmaskind = np.concatenate((tmpmeshRowInd.flatten().reshape(-1,1),\
        tmpmeshColomnInd.flatten().reshape(-1,1)),axis =1);
        _, D = knnsearch(seedCollection,tmprealmaskind,1)
        realvalue_ = (np.exp(myobj.decayparam['alpha'] *(1- (D)/myobj.decayparam['r']))-1)/ (np.exp(myobj.decayparam['alpha']) -1)
          #realvalue_m1 = (1-np.exp(self.decayparam['alpha'] *((D)/self.decayparam['r']) - 1))/ (np.exp(self.decayparam['alpha']) -1)
    else:
        realvalue_ = D
        realvalue_[(realvalue_ <= 0).flatten()] = 0

        se = skimage.morphology.disk(2)
        tmp_mask = skimage.morphology.binary_dilation(mask,se)

        realvalue_[(tmp_mask==0).flatten()] = 0
        boundary_mask = realvalue_.reshape((temprow,tempcol,1))
    return boundary_mask, mask

def yieldImagesfromVolume(myobj):
    alllist  = [f for f in os.listdir(myobj.datadir)]
    for f in alllist:
        if os.path.isfile(os.path.join(myobj.datadir,f)) and \
                   os.path.splitext(f)[1] in myobj.dataExt:
            thisimgfile = os.path.join(myobj.datadir,f)
            #print thisimgfile
            thismat = loadmat(thisimgfile)[myobj.thisdata]
            volume = thismat.tolist()[0][0]
            volanno = thismat.tolist()[0][1]
            boundary= thismat.tolist()[0][2]

            dim = volume.shape[2]
            for depthid in range(dim):
                b_m, mask = process_mask_volume(myobj,boundary[0,depthid], volanno[:,:,depthid])
                if np.sum(mask[:]) < myobj.mask_thresh  and np.random.choice(100,  1) > myobj.mask_prob *100:
                   continue
                yield volume[:,:,depthid], mask, b_m

def process_contour_gaussian(myobj, contour_mat, mask_shape, resizeratio = 1):
    '''
    contour_mat: list of 2*N array of contours
    mask_shape: the resized mask_shape from contour_mat
    resizeratio: we need to resize the contour_mat
    '''
    [temprow,tempcol] = [mask_shape[0], mask_shape[1]]
    #boundary_mask = np.zeros([temprow,tempcol])
    filled_img  = np.zeros([temprow,tempcol])
    numCell = len(contour_mat)
    duplicate =0
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour] * float(resizeratio)

        thiscontour = safe_boarder(thiscontour.T, tempcol,temprow).T

        xcontour = thiscontour[0,:].astype(int)
        ycontour = thiscontour[1,:].astype(int)
        center_x = int(np.mean(xcontour))
        center_y = int(np.mean(ycontour))
        if filled_img[center_y,center_x] != 1:
           filled_img[center_y,center_x] = 1
        else:
           print('duplicate center!')
           duplicate = duplicate + 1
    fimg = scipy.ndimage.filters.gaussian_filter(filled_img, int(2/resizeratio)) * int(100/(resizeratio*resizeratio))
    #imshow(fimg)

    if len(find(filled_img == 1)) != numCell -duplicate:
        assert 0, "Wrong number of positive seed {ps} in filled_img with gt {gt}.".format(ps=len(find(filled_img == 1)), gt = numCell)
    return {'mask': np.reshape(fimg, (fimg.shape + (1,) ) )} #np.reshape(fimg, (fimg.shape + (1,) ) )#{'mask':fimg}

def safe_boarder(boarder_seed, row, col):
    '''
    board_seed: N*2 represent row and col for 0 and 1 axis.
    '''
    boarder_seed[boarder_seed[:,0] < 0, 0] = 0
    boarder_seed[boarder_seed[:,0] >= row,0]   = row-1
    boarder_seed[boarder_seed[:,1] < 0, 1]  = 0
    boarder_seed[boarder_seed[:,1] >= col, 1]  = col-1
    return boarder_seed


def get_info_from_contours(myobj,contour_mat, mask_shape, resizeratio = 1):
    '''
    contour_mat: list of contours, each of which is 2*N, [x; y]
    mask_shape:  the mask shape of which the contour will be plotted on.
    res
    '''
    seedCollection = np.zeros([0,2])
    boarder_seed = np.zeros((0,2))
    mask_shape = (np.asarray(mask_shape)*resizeratio).astype(int)
    [temprow,tempcol] = mask_shape #(resizeratio * mask_shape).astype(int)
    filled_img  = np.zeros([temprow,tempcol])
    numCell = len(contour_mat)
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        xcontour = (resizeratio * np.reshape(thiscontour[0,:].astype(int), (1,-1)))
        ycontour = (resizeratio * np.reshape(thiscontour[1,:].astype(int), (1,-1)))
        center_x = int(np.mean(xcontour))
        center_y = int(np.mean(ycontour))
        seedCollection = np.append(seedCollection,np.array([center_y,center_x])[np.newaxis,:], axis = 0)
        boarder_seed = np.append(boarder_seed,np.array(np.concatenate([ycontour.reshape((-1,1)), xcontour.reshape((-1,1))], axis = -1)), axis = 0)

        if myobj.usecontour == 1 or myobj.usecontour == 'fill':
           tempmask = roipoly(temprow,tempcol,xcontour, ycontour)
           filled_img = np.logical_or(filled_img, tempmask)

        elif myobj.usecontour == 'boarder':
            filled_img[ycontour, xcontour] = 1

    return {'seedCollection':seedCollection,
            'boarder_seed':  boarder_seed,
            'filled_img' :   filled_img,
            'mask_shape': mask_shape
            }

def process_mask_with_weight(myobj, contour_mat, mask_shape, resizeratio = 1):
    contour_info = \
    get_info_from_contours(myobj,contour_mat, mask_shape, resizeratio = resizeratio)

    seedCollection = contour_info['seedCollection']
    boarder_seed = contour_info['boarder_seed']
    filled_img = contour_info['filled_img']
    mask_shape = contour_info['mask_shape']
    [temprow,tempcol] = mask_shape

    filled_img_org = filled_img.copy()

    boarder_seed = safe_boarder(boarder_seed.astype(int), temprow,tempcol )
    if myobj.remove_boundary:
        filled_img[boarder_seed[:,0], boarder_seed[:,1]] = 0

    if  myobj.dilate:
        se = skimage.morphology.disk(myobj.dilate)
        filled_img = skimage.morphology.binary_dilation(filled_img,se)

    if  myobj.erode:
        se = skimage.morphology.disk(myobj.erode)
        filled_img = skimage.morphology.binary_erosion(filled_img,se)

    filled_img = filled_img.astype(np.float)
    [tmpmeshColomnInd, tmpmeshRowInd] = np.meshgrid(range(0,tempcol),range(0,temprow)) #because the flatten is column first
    tmprealmaskind = np.concatenate((tmpmeshRowInd.flatten().reshape(-1,1),\
    tmpmeshColomnInd.flatten().reshape(-1,1)),axis =1);
    _, D = knnsearch(seedCollection,tmprealmaskind,1)
    if myobj.decayparam:
        if myobj.decayparam['use_gauss'] == 1:
            realvalue_=  process_contour_gaussian(myobj = myobj, contour_mat=contour_mat, mask_shape=mask_shape, resizeratio = resizeratio)['mask']
        else:
            realvalue_ = myobj.decayparam['scale'] * ((np.exp(myobj.decayparam['alpha'] *(1 -(D)/myobj.decayparam['r'])) - 1)/ (np.exp(myobj.decayparam['alpha']) -1))
    else:
        realvalue_ = D
    class_mask = np.zeros((temprow,tempcol))
    freq_pos = len(find(filled_img == 1))
    freq_neg = len(find(filled_img == 0))

    class_mask[filled_img == 1] = float(freq_neg)/max(freq_pos, freq_neg)
    class_mask[filled_img == 0] = float(freq_pos)/max(freq_pos, freq_neg)

    realvalue_[(realvalue_ <= 0)] = 0
    final_mask = np.reshape(realvalue_, (temprow, tempcol, -1) )

    weight_value = 1/(1+ myobj.decayparam['decalratio']*D)

    weight_value[(weight_value <= myobj.decayparam['smallweight']).flatten()] = myobj.decayparam['smallweight']
    weight_mask = weight_value.reshape((temprow,tempcol))

    final_weight_mask = myobj.wc*class_mask +  myobj.w * weight_mask

    det_mask = np.zeros((temprow, tempcol, 2))
    det_mask[:,:,0:1] = final_mask
    det_mask[:,:,1] = final_weight_mask
    filled_img = filled_img[:,:,np.newaxis]

    seg_mask = np.zeros((temprow, tempcol, 2), dtype=bool)
    seg_mask[:,:,0] =  filled_img_org
    seg_mask[:,:,1] =  np.ones_like(filled_img_org)

    return  {'det_mask': det_mask, 'seg_mask': seg_mask, 'filled_img': filled_img}

def yieldImages(myobj):
    # all the image are substrate by the mean and divided by its std for RGB channel, respectively.
    mask_process_f = get(myobj.ImageGenerator_Identifier)
    if hasattr(myobj, 'allDictList'):
        allDictList = myobj.allDictList
    else:
        allDictList = getfileinfo(myobj.datadir, myobj.labelSuffix,myobj.dataExt,myobj.labelExt[0])

    index_list = range(0, len(allDictList))
    resizelist = myobj.resizeratio
    #randshuffle(index_list)
    #randshuffle(resizelist)
    for imgindx, thisindex in enumerate(index_list[2:]):
        if imgindx == myobj.maximg:
               break
        returnDict = allDictList[thisindex]
        thismatfile = returnDict['thismatfile']
        thisimgfile = returnDict['thisfile']

        print(thisimgfile)
        img_org = imread(thisimgfile) #np.asarray(Image.open(thisimgfile))
        loaded_mt = loadmat(thismatfile)
        if type( myobj.contourname) is not list:
             myobj.contourname = [ myobj.contourname]
        contour_mat = None
        for contourname in myobj.contourname:
            if contourname in loaded_mt.keys():
               contour_mat = loaded_mt[contourname].tolist()[0]
               break
        if not contour_mat:
            contour_mat = loaded_mt.values()[0].tolist()[0]
            print('check the mat keys, we use the first one default key: ' + loaded_mt.keys()[0])
        outputs_dict = dict()

        for resizeratio in resizelist:
            img_res =  imresize(img_org, resizeratio)
            #print('start mask')
            process_dict = mask_process_f(myobj,contour_mat, img_org.shape[0:2], resizeratio = resizeratio)
            #print('end mask')
            mask_res = process_dict['mask']
            filled_img_res = process_dict['filled_img'] if 'filled_img' in process_dict.keys() else mask_org
            shed_mask_res = process_dict['shed']

            #We may only interested in the region inside one region.
            #since mask and filled_img are already resized version, only image need to be resized
            #mask_res = mask_org #imresize(mask_org, resizeratio)
            #crop the boarder image
            [rowsize, colsize] = [img_res.shape[0], img_res.shape[1]]
            if myobj.boarder < 1:
                row_board = myobj.boarder * rowsize
                col_board = myobj.boarder * colsize
            elif len(myobj.boarder) == 2:
                row_board, col_board = myobj.boarder
            else:
                row_board = col_board = myobj.boarder

            row_start, col_start= row_board * resizeratio, col_board * resizeratio
            row_end = rowsize - row_board * resizeratio
            col_end = colsize - col_board  * resizeratio

            img_res        =  img_res[row_start:row_end, col_start:col_end,...]
            mask_res       =  mask_res[row_start:row_end, col_start:col_end,...]
            filled_img_res =  filled_img_res[row_start:row_end, col_start:col_end,...]
            shed_mask_res  =  shed_mask_res[row_start:row_end, col_start:col_end,...]

            outputs_dict['img'] =   pre_process_img(img_res.copy(), yuv = False)
            outputs_dict['mask'] =  mask_res
            outputs_dict['filled_img'] =   filled_img_res
            outputs_dict['shed_mask'] =    shed_mask_res
            yield outputs_dict

def process_muscle_with_weight(myobj, contour_mat, mask_shape,resizeratio = 1):

    contour_info = \
    get_info_from_contours(myobj,contour_mat, mask_shape, resizeratio = resizeratio)

    #seedCollection = contour_info['seedCollection']
    boarder_seed = contour_info['boarder_seed']
    filled_img = contour_info['filled_img']
    mask_shape = contour_info['mask_shape']
    [temprow,tempcol] = mask_shape

    if  myobj.dilate:
        se = skimage.morphology.disk(myobj.dilate)
        filled_img = skimage.morphology.binary_dilation(filled_img,se)


    boarder_seed = safe_boarder(boarder_seed.astype(int), temprow,tempcol )
    if myobj.remove_boundary:
        filled_img[boarder_seed[:,0], boarder_seed[:,1]] = 0

    if  myobj.erode:
        se = skimage.morphology.disk(myobj.erode)
        filled_img = skimage.morphology.binary_erosion(filled_img,se)

    filled_img = filled_img.astype(np.float32)
    if myobj.use_weighted:
        [tmpmeshRowInd, tmpmeshColomnInd] = np.meshgrid(range(0,tempcol),range(0,temprow))
        tmprealmaskind = np.concatenate((tmpmeshColomnInd.flatten().reshape(-1,1), tmpmeshRowInd.flatten().reshape(-1,1)),axis =1)
        _, double_distance = knnsearch(boarder_seed,tmprealmaskind,1)
        double_distance_sq  = (double_distance ** 2).astype(np.float32)
        weight_mask = np.exp(-double_distance_sq/(2.0*myobj.decayparam['sigma']**2))
        weight_mask = np.reshape(weight_mask, (temprow,tempcol))

        class_mask = np.zeros((temprow,tempcol))
        freq_pos = len(find(filled_img == 1))
        freq_neg = len(find(filled_img == 0))

        class_mask[filled_img == 1] = float(freq_neg)/min(freq_pos, freq_neg)
        class_mask[filled_img == 0] = float(freq_pos)/min(freq_pos, freq_neg)

        final_mask = filled_img  # np.reshape(realvalue_, (temprow, tempcol, -1) )
        final_weight_mask = myobj.wc*class_mask +  myobj.w * weight_mask
    else:
        final_mask = filled_img  # np.reshape(realvalue_, (temprow, tempcol, -1) )
        final_weight_mask = filled_img # means we dont use any weight mask

    Totalmask = np.zeros((temprow, tempcol, 2))
    Totalmask[:,:,0:1] = final_mask[:,:,np.newaxis]
    Totalmask[:,:,1] = final_weight_mask
    filled_img = filled_img[:,:,np.newaxis]

    shed_img = 1 - filled_img
    TotalshedMask = Totalmask.copy()
    TotalshedMask[:,:,0:1] =  shed_img

    return  {'mask': Totalmask, 'shed': TotalshedMask, 'filled_img': filled_img}

