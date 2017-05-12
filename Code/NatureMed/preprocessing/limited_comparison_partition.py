import os, sys, pdb
import numpy as np
import glob, shutil, copy

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
                'Liver', 'LymphNodes', 'Pancreas', 'Prostate', 'SoftTissue', 'Testis', 'Thyroid']
ImgFormats = ['tif', 'png', 'jpg']
TestDiseasesNum = 5
BaseImageNum = 5
ExtraImageNum = 15
ExtraDiseaseNum = 2
OtherDiseaseNum = 2
selection_record = '_selection.txt'

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

    # For selected diseases
    select_disease_names = np.random.choice(DiseaseNames, TestDiseasesNum, replace=False)
    for cur_disease in select_disease_names:
        disease_dir = os.path.join(PingjunTrainingDataDir, cur_disease)
        if os.path.exists(disease_dir):
            shutil.rmtree(disease_dir)
        os.makedirs(disease_dir)
        f_select = open(os.path.join(disease_dir, cur_disease + selection_record), 'w')

        # random get image list
        img_list = data_random_select(cur_disease, BaseImageNum)
        BaseName = cur_disease + str(BaseImageNum)
        f_select.write('# ' + BaseName + '\n')
        base_dir = os.path.join(disease_dir, BaseName)
        os.makedirs(base_dir)
        for img in img_list:
            shutil.copy(img, base_dir)
            img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
            shutil.copy(img_gt, base_dir)
            f_select.write(os.path.basename(img) + '\n')
        f_select.write('\n')

        remain_diseases = copy.deepcopy(DiseaseNames)
        remain_diseases.remove(cur_disease)
        extra2disease = np.random.choice(remain_diseases, ExtraDiseaseNum, replace=False)

        for extra_disease in extra2disease:
            extra_list = data_random_select(extra_disease, ExtraImageNum)
            # Extra*5
            cur_dir_name = BaseName + extra_disease + str(BaseImageNum)
            f_select.write('# ' + cur_dir_name + '\n')
            cur_extra_dir = os.path.join(disease_dir, cur_dir_name)
            extra5_dir = cur_extra_dir
            shutil.copytree(base_dir, cur_extra_dir) # copy base images
            for img in extra_list[:BaseImageNum]:
                shutil.copy(img, cur_extra_dir)
                img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
                shutil.copy(img_gt, cur_extra_dir)
                f_select.write(os.path.basename(img) + '\n')
            f_select.write('\n')

            # Extra*15
            cur_dir_name = BaseName + extra_disease + str(ExtraImageNum)
            f_select.write('# ' + cur_dir_name + '\n')
            cur_extra_dir = os.path.join(disease_dir, cur_dir_name)
            shutil.copytree(base_dir, cur_extra_dir) # copy base images
            for img in extra_list[:ExtraImageNum]:
                shutil.copy(img, cur_extra_dir)
                img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
                shutil.copy(img_gt, cur_extra_dir)
                f_select.write(os.path.basename(img) + '\n')
            f_select.write('\n')

            # Extra5+Extra5+Extra5
            other_diseases = copy.deepcopy(remain_diseases)
            other_diseases.remove(extra_disease)
            other2diseases = np.random.choice(other_diseases, OtherDiseaseNum, replace=False)

            cur_dir_name = BaseName + extra_disease + str(BaseImageNum) + other2diseases[0] \
                + str(BaseImageNum) + other2diseases[1] + str(BaseImageNum)

            f_select.write('# ' + cur_dir_name + '\n')
            cur_extra_dir = os.path.join(disease_dir, cur_dir_name)
            shutil.copytree(extra5_dir, cur_extra_dir)
            for other_disease in other2diseases:
                other_list = data_random_select(other_disease, BaseImageNum)
                for img in other_list:
                    shutil.copy(img, cur_extra_dir)
                    img_gt = os.path.splitext(img)[0] + '_withcontour.mat'
                    shutil.copy(img_gt, cur_extra_dir)
                    f_select.write(os.path.basename(img) + '\n')
                f_select.write('\n')

        f_select.close()
