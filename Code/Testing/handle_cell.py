# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sys
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=True'
CopyRoot  = os.path.join('..','..','..','..')
projroot = os.path.join('..','..')
#dataroot = os.path.join(CopyRoot,'WorkStation','MIA_stru', 'Data')
dataroot = os.path.join(projroot, 'Data')

test_type = 'fcn'
kerasversion = 'keras-1'
sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion))
sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion,'keras'))
sys.path.insert(0, os.path.join(CopyRoot, 'Code', kerasversion,'keras','layers'))
sys.path.insert(0, '..')
sys.path.insert(0,  os.path.join('..', 'proj_utils') )

from local_utils import *
import cv2

from testingclass import runtestImg
import warnings
import time
warnings.filterwarnings("ignore")
from post_processing import *
#from post_process import watershed_marker, label2rgb, overal_cell_watershed
from proj_utils.keras_utils import elu

#------------------------old--------------------------------model
# det_dataset = 'Com_Det'    
# det_modelname = 'deep_det_fcn'
# det_weights_name = 'weights.h5'
# from kerasOneModel import buildCellModel as build_det_model

seg_dataset = 'Com_Seg'    
seg_modelname = 'cell_seg_fcn'
seg_weights_name = 'weights.h5' 
from kerasOneModel import buildCellSegModel as build_seg_model

#----------------------new--model---------------------------------
det_dataset = 'Com_Det'    
det_modelname = 'Det_64_sc_1_m_0'
det_weights_name = 'best_weights.h5'
from kerasOneModel import buildGLUModel as build_det_model

#seg_dataset = 'Com_Seg'    
#seg_modelname = 'Seg_64_sc_1_m_0'
#seg_weights_name = 'best_weights.h5' 
#from kerasOneModel import buildGLUModel as build_seg_model

nb_filters = 64
pool_depth = 4

#modelname = 'deep_det_cell_seg_model'
#from kerasOneModel import buildCellSegModel as build_det_model
activ = elu(alpha=1.0) 
det_last_activ = 'relu'
seg_last_activ = 'sigmoid'

classparams = {'test_type':test_type}
tester = runtestImg(classparams)

def writeResults(coordinates, filepath):
    N = coordinates.shape[0]
    with open(filepath, 'w') as rf:
        rf.write(str(N) + '\n')
        rf.write(str(N) + '\n')
        rf.write(str(N) + '\n') 
        for line in coordinates:
            rf.write(" ".join(map(str, line)) + '\n' )            
    rf.close()

def load_weights(strumodel =None, modelbasefolder = None, dataset = 'Mixture',  
                 weights_name='weights.h5', modelname='dilated_residual_fcn'):
    weightspath = os.path.join(modelbasefolder, dataset,modelname,weights_name )
    strumodel.load_weights(weightspath)
    return strumodel

def seg_prob_fn(imgfile, model=None, windowsize = 500, batch_size = 1, linewidth = 7,color=[0,1,0], tester = None):
    #this function is for web interface 
    if tester is None:
        classparams = {'test_type':test_type}
        tester = runtestImg(classparams)
        
    if isinstance(imgfile, str):
        orgImg = imread(imgfile)
    else:
        orgImg = imgfile
    TestingFunc = tester.shortCut_FCN
    mask = TestingFunc(orgImg.copy(), model = model,windowsize = windowsize, batch_size = batch_size, linewidth = linewidth, color= color)
    #mask = tester.printMask(img= orgImg.copy(), mask=mask)
    return mask

def seg_post_process(seg_prob, coordinates, thresh_water = 0.8, thresh_seg = 0.2):
    marker_map = np.zeros_like(seg_prob).astype(np.uint8)
    marker_map[coordinates[:, 0], coordinates[:, 1]] = 1
    label_img = overal_watershed_marker(seg_prob, marker_map, thresh_water = thresh_water, thresh_seg= thresh_seg)
    return label_img
    
def det_fn(imgfile, model=None, tester= None, windowsize = 500, batch_size = 1, threshhold = 0.1, probmap = None,
           min_len=7, linewidth = 7,color=[0,1,0]):
    if tester is None:
        classparams = {'test_type':test_type}
        tester = runtestImg(classparams)
       
    if isinstance(imgfile, str):
       orgImg = imread(imgfile)
    else:
       orgImg = imgfile
    coordinates, probmap_ = detection_coord(orgImg, model=model, windowsize = windowsize, batch_size = batch_size, 
                                            probmap=probmap,threshhold = threshhold,min_len=min_len, linewidth = linewidth, color= color)
    overlaied_img = tester.printCoord(Img=orgImg, coordinates=coordinates, savepath=None, linewidth = linewidth,color=color)
    return overlaied_img, coordinates, probmap_
    
def detection_coord(imgfile, model=None, tester=None, windowsize = 500,threshhold = 0.1, batch_size = 2,min_len=7, 
                    linewidth = 7,color=[0,1,0], probmap=None):
    #this function is for web interface  
    if tester is None:
        classparams = {'test_type':test_type}
        tester = runtestImg(classparams)
    
    if isinstance(imgfile, str):
       orgImg = imread(imgfile)
    else:
       orgImg = imgfile
    TestingFunc = tester.get_coordinate
    coordinates, probmap_ = TestingFunc(orgImg.copy(), model = model,windowsize = windowsize, min_len= min_len, probmap= probmap,
                              threshhold = threshhold, batch_size = batch_size, linewidth = linewidth, color= color)
    
    return coordinates, probmap_

    
def setpath(KerasRoot):
    sys.path.insert(0, os.path.join(KerasRoot))
    sys.path.insert(0, os.path.join(KerasRoot,'keras'))
    sys.path.insert(0, os.path.join(KerasRoot,'keras','layers'))

def get_model(modelbasefolder = None):
    #modelbasefolder = os.path.join(projroot, 'Data', 'Model') 
      
    det_model = build_det_model(img_channels = 3, nb_filters = nb_filters, activ=activ,
                                last_activ = det_last_activ, pool_depth=pool_depth)   
    det_model = load_weights(strumodel = det_model,modelbasefolder = modelbasefolder, weights_name= det_weights_name,
                            modelname = det_modelname, dataset = det_dataset)

    seg_model = build_seg_model(img_channels = 3,  nb_filters = nb_filters, activ=activ, 
                                last_activ = seg_last_activ, pool_depth=pool_depth)   
    seg_model = load_weights(strumodel = seg_model, modelbasefolder = modelbasefolder, weights_name= seg_weights_name,
                            modelname = seg_modelname, dataset = seg_dataset)

    return det_model, seg_model
    print("Finihsed compilation!")


def cell_det(imgfile, det_model, threshhold = 0.1,min_len=7,windowsize = None,batch_size=1, probmap = None): 

    start = time.time()
    img = imread(imgfile)
    det_masked,coordinates , probmap_ = det_fn(img, model = det_model, tester = tester, threshhold = threshhold, 
                                        windowsize = windowsize,batch_size=batch_size, probmap=probmap,min_len= min_len)
    print(time.time() - start)  
    print('finished detection computation')
    return det_masked, coordinates, probmap_

def cell_seg(imgfile, seg_model, det_model= None, coordinates = None,windowsize=None,mask_or_contour='contour',
             batch_size=1, thresh_water = 0.8, thresh_seg = 0.7, probmap = None): 
    start = time.time()
    img = imread(imgfile)
    if coordinates is None:
        if det_model is not None:
            det_masked, coordinates,_ = det_fn(img, model = det_model, tester = tester)
            print(time.time() - start)    
        else:
            raise RuntimeError('det_model and coordinates can be None at the same time!!')
    if  probmap is None:
        seg_map =  seg_prob_fn(img, model = seg_model, windowsize= windowsize, batch_size=batch_size,tester = tester)
        print(time.time() - start)  
        print('finished cell seg computation')
    else:
        seg_map = probmap

    returned_map = seg_map.copy()   
    seg_label = seg_post_process(seg_map, coordinates, thresh_water = thresh_water, thresh_seg= thresh_seg)
    print('finish post processsing')
    if mask_or_contour == 'mask':
        seg_result = label2rgb(seg_label.astype(np.uint8),img, alpha=0.2).astype(np.uint8)
    else:
        seg_result,_ = label2contour(seg_label,img, linewidth=3)
        print('finish generating contour')    
    return seg_result, returned_map

