import sys
import os
import argparse
import shutil
import random

random.seed(5)
projroot   = os.path.join('..','..', '..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))

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
    Exts = [Exts] if type(Exts) is not list else Exts
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

def doit(all_tuple, mode='copy'):
    '''
    :param all_tuple: each element is (dst_root, (src_mat, src_img, dst_sub) ) 
    :return: None, just perform the file transfer.
    '''
    operate = shutil.copy if mode=='copy' else shutil.move

    for dst_folder, tuple_process in all_tuple:
        for this_tuple in tuple_process:
            src_mat, src_img, dst_sub = this_tuple
            dst_subfolder = os.path.join(dst_folder, dst_sub)
            if not os.path.exists(dst_subfolder):
                os.makedirs(dst_subfolder)
            operate(src_mat, dst_subfolder)
            operate(src_img, dst_subfolder)
            
if __name__ == '__main__':
    ImgFloder   = os.path.join(home,  'DataSet', 'Nature', 'TrainingData')
    valid_save  = os.path.join(home,  'DataSet', 'Nature', 'ValidationData')
    valid_num = 5

    folderlist, foldernamelist = getfolders(ImgFloder)
    for subfolder, subname in zip(folderlist, foldernamelist):
        filelist, namelist = getfiles(subfolder,['.tif', '.png', '.jpg'])
        operation_tuple = []
        for imgpath, imgname in zip(filelist, namelist):
            matfilename = imgname + '_withcontour.mat'
            src_mat = os.path.join(subfolder, matfilename)
            dst_sub = os.path.join(subname)
            operation_tuple.append((src_mat, imgpath, dst_sub))

        random.shuffle(operation_tuple)
        valid_tuple = operation_tuple[0:valid_num]
        all_tuple = [
            (valid_save, valid_tuple)
        ]
        doit(all_tuple, mode='move')

    # ImgFloder  = os.path.join(home, 'DataSet', 'crop_anno_patches')
    # train_save = os.path.join(home, 'DataSet', 'Nature', 'TrainingData')
    # test_save  = os.path.join(home, 'DataSet', 'Nature',  'TestingData')
    #
    # folderlist, foldernamelist = getfolders(ImgFloder)
    # for subfolder, subname in zip(folderlist, foldernamelist):
    #     filelist, namelist = getfiles(subfolder,['.tif', '.png', '.jpg'])
    #     operation_tuple = []
    #     for imgpath, imgname in zip(filelist, namelist):
    #         matfilename = imgname + '_withcontour.mat'
    #         src_mat = os.path.join(subfolder, matfilename)
    #         dst_sub = os.path.join(subname)
    #         #dst_img = os.path.join(subname)
    #         operation_tuple.append((src_mat, imgpath, dst_sub))
    #
    #     random.shuffle(operation_tuple)
    #     totalnum = len(operation_tuple)
    #     midnum = totalnum // 2
    #     train_tuple = operation_tuple[0:midnum]
    #     test_tuple = operation_tuple[midnum:]
    #     all_tuple = [
    #         (train_save, train_tuple),
    #         (test_save, test_tuple)
    #     ]
    #     doit(all_tuple, mode='copy')
