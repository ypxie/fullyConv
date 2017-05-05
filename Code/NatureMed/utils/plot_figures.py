import sys, os
import numpy as np
projroot   = os.path.join('..','..','..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))

from visdom import Visdom

from torch_fcn.proj_utils.Extractor import FcnExtractor
from torch_fcn.Models import MultiContex as build_model
from torch_fcn.proj_utils.local_utils import Indexflow, imread, imshow,pre_process_img
from torch_fcn.proj_utils.testingclass import runtestImg
from torch_fcn.proj_utils.ImageGenerator import get_mask
import torch

#trainingDataroot = os.path.join(home,'Dropbox','DataSet', 'Nature', 'TrainingData')
#validationDataroot = os.path.join(home,'Dropbox','DataSet', 'Nature', 'ValidationData')

modelroot = os.path.join(projroot, 'Data','NatureModel', 'YuanpuModel')
#file_list = os.path.join(trainingDataroot, 'All', 'a1.jpg')
weightspath = os.path.join(modelroot, 'All', 'multicontex','best_weights.pth')
device = 0
strumodel = build_model()
strumodel.cuda(device)

weights_dict = torch.load(weightspath)
strumodel.load_state_dict(weights_dict['weights'])# 12)

tester = runtestImg({'model': strumodel})
plot = Visdom()

classparams = {}
classparams['ImageGenerator_Identifier'] = 'process_mask_with_weight'
# 'process_contour_cellcounting' # 'process_cell_mask'
# the following is for Image generator parameters
decayparam = {}
decayparam['alpha'] = 3
decayparam['r'] = 15
decayparam['scale'] = 5
decayparam['decalratio'] = 0.1
decayparam['smallweight'] = 0.05
decayparam['use_gauss'] = 0

weight_params = {}
weight_params['beta'] = 1.0 / decayparam['scale']
weight_params['alpha'] = (1.0 * 5) / decayparam['scale']

classparams['decayparam'] = decayparam
classparams['w'] = 10  # weight wrt distance
classparams['wc'] = 1  # weight wrt to class
classparams['dilate'] = 8

classparams['volume'] = 2
classparams['dataExt'] = ['.tif', '.png', '.jpg']  # the data ext
classparams['labelExt'] = ['.mat']  # the label ext
classparams['contourname'] = 'Contours'  # the contour name in the cell array
classparams['labelSuffix'] = ["", '_withcontour', '_gt', '_seg']  # the suffix of label
classparams['maxsamples'] = 1280000
classparams['usecontour'] = 1  # this is just used to fill the cotour to get filled_img,
classparams['pickratio'] = 0.05  # 1 means take all the pixel

classparams['maximg'] = 30
classparams['mask_thresh'] = 50
classparams['mask_prob'] = 0.1
classparams['maxpatch'] = 10
classparams['random_pick'] = True  # if you wanna random pick

StruExtractor = FcnExtractor(classparams)

def get_file_info(filefolder, filename):
    # return image path, mat_path, and one corresponding map.
    # map may be directly constructed form the model.predict(img)
    img_path = os.path.join(filefolder, filename + '.jpg')
    mat_path = os.path.join(filefolder, filename + '_withcontour.mat')

    return img_path, mat_path

data_root = os.path.join(home, 'Dropbox/DataSet/NatureData/YuanpuData/TrainingData')
tuple_list = [
    ('AdrenalGland','TCGA-OR-A5JC-40x-01', '.jpg'),
    ('HeadNeck','TCGA-BA-A6DJ-40x-1', '.tif'),
    ('Kidney','TCGA-KL-8324-40x-1', '.tif'),
    ('Ovary','TCGA-3P-A9WA-01Z-00-DX1', '.tif'),
    ('Pleura','TCGA-3U-A98G-01Z-00-DX1', '.tif'),
]
for this_tuple in tuple_list:
    sub_folder, img_name, ext = this_tuple
    img_path = os.path.join(data_root, sub_folder, img_name+ext)
    mat_path = os.path.join(data_root, sub_folder, img_name + '_withcontour.mat')
    img_org = imread(img_path)

    img = pre_process_img(img_org, yuv=False)

    VotingMap = np.squeeze(tester.split_testing(img, model = strumodel, windowsize = 1000,
                                                batch_size = 2,fixed_window= False))

    predMap = get_mask(StruExtractor, img_org, mat_path, contourname = ['Contours'], resizeratio=1)[:,:,0]

    this_marker = sub_folder + '_' + img_name
    plot.image(img_org.transpose(2,0,1),     opts=dict(title = this_marker))
    plot.heatmap(X=VotingMap, win= this_marker + '_prediction', opts=dict(title=this_marker + '_prediction'))
    plot.heatmap(X=predMap,   win= this_marker + '_groundTruth', opts=dict(title=this_marker + '_groundTruth'))
    imshow(VotingMap)




