import sys
import os
import argparse
import shutil

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

def doit(tuple_list, train_save, test_save):
    totalnum = len(tuple_list)
    midnum = totalnum//2
    train_tuple = tuple_list[0:midnum]
    test_tuple = tuple_list[midnum:]
    all_tuple = [
                 (train_save, train_tuple),
                 (test_save, test_tuple)
                ]
    for dst_folder, tuple_process in all_tuple:
    
        for this_tuple in tuple_process:
            src_mat, dst_mat_, src_img, dst_img_ = this_tuple
            dst_mat = os.path.join(dst_folder, dst_mat_)   
            dst_img = os.path.join(dst_folder, dst_img_)  
            if not os.path.exists(dst_mat):
                os.makedirs(dst_mat)
            if not os.path.exists(dst_img):
                os.makedirs(dst_img)    
            shutil.copy(src_mat, dst_mat)
            shutil.copy(src_img, dst_img)
            
if __name__ == '__main__':

    ImgFloder  = os.path.join(home, 'DataSet', 'crop_anno_patches')
    train_save = os.path.join(home, 'DataSet', 'Nature', 'TrainingData')
    test_save  = os.path.join(home, 'DataSet', 'Nature',  'TestingData')
    folderlist, foldernamelist = getfolders(ImgFloder)
    for subfolder, subname in zip(folderlist, foldernamelist):
        filelist, namelist = getfiles(subfolder,['.tif', '.png', '.jpg'])
        operation_tuple = []
        for imgpath, imgname in zip(filelist, namelist):
            matfilename = imgname + '_withcontour.mat'
            src_mat = os.path.join(subfolder, matfilename)
            dst_mat = os.path.join(subname)

            dst_img = os.path.join(subname)

            operation_tuple.append((src_mat, dst_mat, imgpath, dst_img))

        doit(operation_tuple, train_save, test_save)
