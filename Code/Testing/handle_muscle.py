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

from testingclass import runtestImg
import warnings
import time

from proj_utils.keras_utils import elu
from proj_utils.post_processing import *

#modelname = 'MixModel'
#from kerasOneModel import buildMixModel as buildmodel
#activ = 'relu'

#-----------------------old------------model---------
# modelname = 'down_to_3_muscle'
# from kerasOneModel import buildMuscleSegModel as buildmodel
# activ = elu(alpha=1.0)  

# #modelname = 'deep_muscle_fcn'
# #from kerasOneModel import buildCellSegModel as buildmodel
# #activ = elu(alpha=1.0)

# #weights_name = 'weights_0829_perfect.h5'
# #weights_name = 'weights_09_07_night_bp.h5'
# #weights_name = 'weights_maker_great.h5'
# #weights_name = 'weights_demo_823.h5'
# weights_name = 'weights.h5'
# weights_name = 'weights_926_perfect.h5'
# #weights_name = 'weights_0829_perfect.h5'
# #weights_name = 'weights_0.17.h5'
# #weights_name = 'weights_b4adadelta.h5'
# #weights_name = 'weights_good_friday.h5'
# #weights_name = 'weights_almostFinal.h5'
# #weights_name = 'weights_perfect_0829.h5'
# warnings.filterwarnings("ignore")

#------------------new------------------model
modelname = 'gateConv_64_sc_1_m_0'
from kerasOneModel import buildGLUModel as buildmodel
activ = elu(alpha=1.0)  
last_activ = 'sigmoid'
weights_name = 'best_weights.h5'
nb_filters = 64
pool_depth = 5

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

def load_weights(strumodel =None, modelbasefolder = None, dataset = 'Mixture', modelname='dilated_residual_fcn'):
    weightspath = os.path.join(modelbasefolder, dataset,modelname,weights_name )
    strumodel.load_weights(weightspath)
    return strumodel
    

def seg_fn(imgfile, model=None, windowsize = 500, batch_size = 2, linewidth = 7,color=[0,1,0]):
    #this function is for web interface  
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
    
def setpath(KerasRoot):
    sys.path.insert(0, os.path.join(KerasRoot))
    sys.path.insert(0, os.path.join(KerasRoot,'keras'))
    sys.path.insert(0, os.path.join(KerasRoot,'keras','layers'))

def get_model(dataset = 'Muscle', modelbasefolder = None):
    #modelbasefolder = os.path.join(projroot, 'Data', 'Model')   
    strumodel = buildmodel(img_channels = 3,nb_filters=nb_filters, activ=activ, last_activ = last_activ, pool_depth=pool_depth)   
    strumodel = load_weights(strumodel = strumodel,modelbasefolder = modelbasefolder, modelname = modelname, dataset = dataset)
    return strumodel
    print("Finihsed compilation!")

def muscle_seg(imgfile, strumodel= None, mask_or_contour = 'mask', single_or_not = False, 
                windowsize=None, thresh_water = 0.8, thresh_seg = 0.3, probmask = None): 
    from skimage.color import label2rgb

    TestingFunc = tester.get_coordinate

    #coordinates = TestingFunc(imgfile, model = strumodel)
    start = time.time()
    img_org = imread(imgfile)
    img_res = imresize(img_org, 1)
    if probmask is None:
        probmask = seg_fn(img_res, windowsize = windowsize, model = strumodel)
    else:
        probmask = probmask
    
    img_res = img_res.astype(np.uint8)

    if single_or_not:   
        markers_single = single_watershed(probmask,thresh_water = thresh_water, thresh_seg = thresh_seg,ratio = 0.2, org_img = img_res)
        #imshow(markers_single)
        if mask_or_contour == 'mask':
            return label2rgb(markers_single,img_res, alpha=0.2), probmask
        #imshow(markers_single)
        #imshow(label2rgb(markers_single,img_res, alpha=0.2), (10,10))
        else:
            contourMarkeredImg, _ = label2contour(markers_single, img_res)
            return contourMarkeredImg, probmask
        #imshow( contourMarkeredImg, (10,10))
    else:      
        markers_overral = overal_watershed(probmask,thresh_water = thresh_water, thresh_seg = thresh_seg,ratio = 0.2, org_img = img_res)
        #imshow(markers_overral) 
        if mask_or_contour == 'mask':
            return label2rgb(markers_overral,img_res, alpha=0.2), probmask
        else:
        #imshow(markers_overral)
        #imshow(label2rgb(markers_overral,img_org, alpha=0.3), (10,10))
        #imshow(label2rgb(markers_overral,img_res, alpha=0.2), (10,10))
            contourMarkeredImg,_ = label2contour(markers_overral, img_res)
            return contourMarkeredImg, probmask
        #imshow( contourMarkeredImg, (10,10))



