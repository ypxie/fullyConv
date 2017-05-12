import os, sys, pdb
import numpy as np
import glob, shutil

BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
NatureDataDir = os.path.join(BaseDir, 'NatureData')
YuanpuTrainingDataDir = os.path.join(NatureDataDir, 'YuanpuData', 'TrainingData')
YuanpuValidationDataDir = os.path.join(NatureDataDir, 'YuanpuData', 'ValidationData')

PingjunTrainingDataDir = os.path.join(NatureDataDir, 'PingjunData', 'TrainingData')
PingjunValidationDataDir = os.path.join(NatureDataDir, 'PingjunData', 'ValidationData')

# PingjunTrainingDataDir = os.path.join(HomeDir, 'Test', 'TrainingData')
# PingjunValidationDataDir = os.path.join(HomeDir, 'Test', 'ValidationData')

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
    seed_value = 1234
    np.random.seed(seed_value)
    # random select disease
    select_disease = np.random.choice(DiseaseNames)
    # random get image list
    img_list = data_random_select(select_disease, NumberSamples)

    # copy to pingjun training folder as well as validation
    disease_dir = os.path.join(PingjunTrainingDataDir, select_disease)
    if os.path.exists(disease_dir):
        shutil.rmtree(disease_dir)
    os.makedirs(disease_dir)
    disease_val_dir = os.path.join(PingjunValidationDataDir, select_disease)
    if os.path.exists(disease_val_dir):
        shutil.rmtree(disease_val_dir)
    os.makedirs(disease_val_dir)

    selection_record = 'selection.txt'
    f_select = open(os.path.join(disease_dir, selection_record), 'w')
    f_select.write('# Seed Value: ' + str(seed_value) + '\n')
    # For Base images
    f_select.write('# Base' + '\n')
    f_select.write('## ' + select_disease + '\n')
    base_folder = select_disease + 'Base'
    base_dir = os.path.join(disease_dir, base_folder)
    os.makedirs(base_dir)
    for img in img_list:
        shutil.copy(img, base_dir)
        img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
        shutil.copy(img_gt, base_dir)
        f_select.write(os.path.basename(img) + '\n')
    f_select.write('\n')
    ln_src_dir = os.path.join(YuanpuValidationDataDir, select_disease)
    ln_dst_dir = os.path.join(PingjunValidationDataDir, select_disease, base_folder)
    os.system("ln -s " + ln_src_dir + ' ' + ln_dst_dir)

    # Random add other disease to assist
    number_disease = np.arange(3, 5)
    number_cases = np.arange(NumberSamples, 10)

    # For each group
    for cur_num in range(1, ExtraDiseaseNum+1):
        f_select.write("Group" + str(cur_num) + '\n')
        cur_g_folder = select_disease + str(cur_num)
        cur_g_dir = os.path.join(disease_dir, cur_g_folder)
        shutil.copytree(base_dir, cur_g_dir)

        # For each disease
        num_dis = number_disease[0]
        cur_dis_num = 0
        disease_list = [select_disease]
        while cur_dis_num < num_dis:
            tmp_dis = np.random.choice(DiseaseNames)
            if tmp_dis in disease_list:
                continue

            f_select.write('## ' + tmp_dis + '\n')
            cur_list = data_random_select(tmp_dis, number_cases[0])
            for img in cur_list:
                shutil.copy(img, cur_g_dir)
                img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
                shutil.copy(img_gt, cur_g_dir)
                f_select.write(os.path.basename(img) + '\n')
            disease_list.append(tmp_dis)
            cur_dis_num += 1
        f_select.write('\n')
        # link to validation
        ln_dst_dir = os.path.join(PingjunValidationDataDir, select_disease, cur_g_folder)
        os.system("ln -s " + ln_src_dir + ' ' + ln_dst_dir)
    f_select.close()
