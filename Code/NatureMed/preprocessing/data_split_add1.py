import os, sys, pdb
import numpy as np
import glob, shutil

HomeDir = os.path.expanduser('~')
NatureDataDir = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureData')
YuanpuTrainingDataDir = os.path.join(NatureDataDir, 'YuanpuData', 'TrainingData')
YuanpuValidationDataDir = os.path.join(NatureDataDir, 'YuanpuData', 'ValidationData')
PingjunTrainingDataDir = os.path.join(NatureDataDir, 'PingjunData', 'TrainingData')
PingjunValidationDataDir = os.path.join(NatureDataDir, 'PingjunData', 'ValidationData')
DiseaseNames = ['AdrenalGland', 'Bladder', 'Breast', 'Colorectal', 'Eye', 'Kidney',
                'Lung', 'Ovary', 'Pleura', 'Skin', 'Stomach', 'Thymus',
                'Uterus', 'BileDuct', 'Brain', 'Cervix', 'Esophagus', 'HeadNeck',
                'Liver', 'LymphNodes', 'Pancreas', 'Prostate', 'SoftTissue', 'Testis',
                'Thyroid']
NumberSamples = 5
ExtraDiseaseNum = 3
ImgFormats = ['tif', 'png', 'jpg']


def get_image_names(img_dir, img_formats):
    img_list = []
    for img_ext in img_formats:
        img_pattern = os.path.join(img_dir, '*.' + img_ext)
        img_list.extend(glob.glob(img_pattern))
    return img_list


def data_random_select(disease_name, sample_num):
    disease_dir = os.path.join(YuanpuTrainingDataDir, disease_name)
    img_list = get_image_names(disease_dir, ImgFormats)
    # shuffle image list
    np.random.shuffle(img_list)
    selected_list = img_list[:sample_num]

    return selected_list

if __name__ == '__main__':
    seed_value = 1236
    np.random.seed(seed_value)
    # random select disease
    select_disease = np.random.choice(DiseaseNames)
    # random get image list
    img_list = data_random_select(select_disease, NumberSamples)
    # copy to pingjun training folder as well as validation
    disease_dir = os.path.join(PingjunTrainingDataDir, select_disease)

    selection_record = 'selection.txt'
    f_select = open(os.path.join(disease_dir, selection_record), 'a')
    f_select.write('\n')

    base_folder = select_disease + 'Base'
    base_dir = os.path.join(disease_dir, base_folder)

    tmp_dis = np.random.choice(DiseaseNames)
    while tmp_dis == select_disease:
        tmp_dis = np.random.choice(DiseaseNames)


    select_num = 5
    # add disease * 5
    f_select.write(select_disease + tmp_dis + str(select_num) + '\n')
    cur_g_folder = select_disease + tmp_dis + str(select_num)
    cur_g_dir = os.path.join(disease_dir, cur_g_folder)
    shutil.copytree(base_dir, cur_g_dir)

    cur_list = data_random_select(tmp_dis, select_num)
    for img in cur_list:
        shutil.copy(img, cur_g_dir)
        img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
        shutil.copy(img_gt, cur_g_dir)
        f_select.write(os.path.basename(img) + '\n')
    f_select.write('\n')

    select_num = 15
    # add disease * 15
    f_select.write(select_disease + tmp_dis + str(select_num) + '\n')
    cur_g_folder = select_disease + tmp_dis + str(select_num)
    cur_g_dir = os.path.join(disease_dir, cur_g_folder)
    shutil.copytree(base_dir, cur_g_dir)

    cur_list = data_random_select(tmp_dis, select_num)
    for img in cur_list:
        shutil.copy(img, cur_g_dir)
        img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
        shutil.copy(img_gt, cur_g_dir)
        f_select.write(os.path.basename(img) + '\n')
    f_select.write('\n')
    
    f_select.close()
