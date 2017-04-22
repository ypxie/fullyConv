import sys
import os
import argparse

projroot   = os.path.join('..','..', '..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))
dataroot = '/media/yuanpuxie/lab_book/TCGA_diagnostic_data'

from time import time
import numpy as np
import matplotlib.pyplot as plt

from torch_fcn.proj_utils.local_utils import *

from openslide import open_slide
#img_slide = open_slide(slidepath)

def readpatch(img_slide, sizes=(500,500)):
    row, col = img_slide.dimensions

    rowsize, colsize = sizes
    row_cent, col_cent = row//2, col//2
    img = img_slide.read_region((row_cent, col_cent), 0, (rowsize, colsize))

    return np.asarray(img)

def getfolders(imgdir):
    '''
    return a list of folders
    '''
    alllist  = [f for f in os.listdir(imgdir)]
    returnList = []
    nameList = []
    for f in alllist:
        folder = os.path.join(imgdir,f)
        if os.path.isdir(folder):
            returnList.append(folder)
            nameList.append(f)
    return returnList, nameList

def getfiles(imgdir, Exts):
    '''
    return a list of dictionary {'thisfile':os.path.join(imgdir,f)}
    '''
    alllist  = [f for f in os.listdir(imgdir)]
    fileList = []
    nameList = []
    for f in alllist:
        name = os.path.splitext(f)[0]
        path = os.path.join(imgdir,f)
        if os.path.isfile(path) and \
                   os.path.splitext(f)[1] in Exts:
            fileList.append(path)
            nameList.append(name)
    return fileList, nameList

diseaseFolders = [ 'Thymus', 'Thyroid', 'Uterus','LymphNodes',]
saveroot = os.path.join(home,'CropData')

nums = 50



for disease in diseaseFolders:
    counter = 0 
    name_dict = {}
    print('processing', disease)
    thisSave = os.path.join(saveroot, disease)
    if not os.path.exists(thisSave):
        os.mkdir(thisSave)
    disFolder = os.path.join(dataroot, disease)
    subfolderList, subfolderName = getfolders(disFolder)
    for subfolder, subname in zip(subfolderList, subfolderName):
        if counter == nums:
            break
            counter = 0

        filelist, namelist = getfiles(subfolder,'.tif')
        for svsfile, svsname in zip(filelist, namelist):
            try:
                #this_img_slide = open_slide(svsfile)
                this_img_slide = imread(svsfile)
                #this_patch = readpatch(this_img_slide, sizes=(500,500))
                this_patch = this_img_slide[0:500,0:500, :]
                split_name = svsname.split('-')
                AA   = split_name[1]
                BBBB = split_name[2]
                if name_dict.get(BBBB, None) is None:
                    name_dict[BBBB] = counter
                    counter = counter + 1
                
                    CC = '20x' if '20X' in split_name or '20x' in split_name else '40x'
                    D = '1'
                    img_name = 'TCGA-' + AA + '-' + BBBB + '-' + CC + '-' + D + '.tif'
                    
                    writeImg(this_patch, os.path.join(thisSave, img_name))
    
                else:
                    print(BBBB)
                    continue
            except:
                print('pass', svsfile)
    print(disease, counter, 'images found.\n')
                
    
