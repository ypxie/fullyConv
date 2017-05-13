import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
ExperimentsResults = os.path.join(BaseDir, 'NatureData', 'YuanpuData', 'Experiments', 'evaluation_pinjun')

DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)

DetectModel = dict()
# DetectModel['Cervix'] = ['CervixBase', 'CervixLiver5', 'CervixLiver15', 'Cervix3Extra', 'Cervix', 'All']
# DetectModel['Colorectal'] = ['ColorectalBase', 'ColorectalEye5', 'ColorectalEye15', 'Colorectal3Extra', 'Colorectal', 'All']
# DetectModel['Thymus'] = ['ThymusBase', 'ThymusThyroid5', 'ThymusThyroid15', 'Thymus3Extra', 'Thymus', 'All']

DetectModel['Bladder'] = ['Bladder5', 'Bladder5Prostate5', 'Bladder5Prostate15', 'Bladder5Prostate5Pleura5Breast5', 'Bladder5Liver5',
                          'Bladder5Liver15', 'Bladder5Liver5Cervix5Skin5', 'Bladder', 'All']
DetectModel['Breast'] = ['Breast5', 'Breast5Esophagus5', 'Breast5Esophagus15', 'Breast5Esophagus5Pancreas5Eye5', 'Breast5Uterus5',
                         'Breast5Uterus15', 'Breast5Uterus5Lung5HeadNeck5', 'Breast', 'All']
DetectModel['Colorectal'] = ['Colorectal5', 'Colorectal5Bladder5', 'Colorectal5Bladder15', 'Colorectal5Bladder5Prostate5Eye5', 'Colorectal5Testis5',
                         'Colorectal5Testis15', 'Colorectal5Testis5BileDuct5LymphNodes5', 'Colorectal', 'All']
DetectModel['Esophagus'] = ['Esophagus5', 'Esophagus5AdrenalGland5', 'Esophagus5AdrenalGland15', 'Esophagus5AdrenalGland5Breast5Lung5',
                            'Esophagus5Prostate5', 'Esophagus5Prostate15', 'Esophagus5Prostate5AdrenalGland5HeadNeck5', 'Esophagus', 'All']
DetectModel['Ovary'] = ['Ovary5', 'Ovary5Breast5', 'Ovary5Breast15', 'Ovary5Breast5HeadNeck5Thyroid5', 'Ovary5SoftTissue5',
                        'Ovary5SoftTissue15', 'Ovary5SoftTissue5Lung5Testis5', 'Ovary', 'All']


base_means, base_std = [], []
a1extra5_means, a1extra5_std = [], []
a1extra15_means, a1extra15_std = [], []
a3extra5_means, a3extra5_std = [], []
b1extra5_means, b1extra5_std = [], []
b1extra15_means, b1extra15_std = [], []
b3extra5_means, b3extra5_std = [], []
disease_means, disease_std = [], []
generic_means, generic_std = [], []


best_key = 'thresh_0.75_len_11_radius_16.00'
surfix = '_multicontex_best_weights_res.json'

for cur_d in DetectModel.keys():
    for ind, cur_model in enumerate(DetectModel[cur_d]):
        cur_f = os.path.join(ExperimentsResults, cur_d, cur_model+surfix)
        cur_results = json.load(open(cur_f))
        cur_f1score = cur_results[best_key]['f1_mean']
        cur_f1std = cur_results[best_key]['f1_std']
        if ind == 0:
            base_means.append(cur_f1score)
            base_std.append(cur_f1std)
        elif ind == 1:
            a1extra5_means.append(cur_f1score)
            a1extra5_std.append(cur_f1std)
        elif ind == 2:
            a1extra15_means.append(cur_f1score)
            a1extra15_std.append(cur_f1std)
        elif ind == 3:
            a3extra5_means.append(cur_f1score)
            a3extra5_std.append(cur_f1std)
        elif ind == 4:
            b1extra5_means.append(cur_f1score)
            b1extra5_std.append(cur_f1std)
        elif ind == 5:
            b1extra15_means.append(cur_f1score)
            b1extra15_std.append(cur_f1std)
        elif ind == 6:
            b3extra5_means.append(cur_f1score)
            b3extra5_std.append(cur_f1std)
        elif ind == 7:
            disease_means.append(cur_f1score)
            disease_std.append(cur_f1std)
        elif ind == 8:
            generic_means.append(cur_f1score)
            generic_std.append(cur_f1std)

# Drawing
bar_width = 0.30                   # the width of the bars
ind = np.arange(0, 3*len(DetectModel.keys()), 3)        # the x locations for the groups

start_m = (3 - bar_width * 9) / 2
start_w = start_m + bar_width
# fig, ax = plt.subplots(figsize=(20, 15))
fig, ax = plt.subplots()
# pdb.set_trace()
Base = ax.bar(ind+start_m+bar_width*0, base_means, bar_width, color='#e41a1c', yerr=base_std)
A1extra5 = ax.bar(ind+start_m+bar_width*1, a1extra5_means, bar_width, color='#369ead', yerr=a1extra5_std)
A1extra15 = ax.bar(ind+start_m+bar_width*2, a1extra15_means, bar_width, color='#7f6084', yerr=a1extra15_std)
A3extra5 = ax.bar(ind+start_m+bar_width*3, a3extra5_means, bar_width, color='#a2d1c5', yerr=a3extra5_std)
B1extra5 = ax.bar(ind+start_m+bar_width*4, b1extra5_means, bar_width, color='#369ead', yerr=b1extra5_std)
B1extra15 = ax.bar(ind+start_m+bar_width*5, b1extra15_means, bar_width, color='#7f6084', yerr=b1extra15_std)
B3extra5 = ax.bar(ind+start_m+bar_width*6, b3extra5_means, bar_width, color='#a2d1c5', yerr=b3extra5_std)
Disease = ax.bar(ind+start_m+bar_width*7, disease_means, bar_width, color='#1f78e4', yerr=disease_std)
Generic = ax.bar(ind+start_m+bar_width*8, generic_means, bar_width, color='#984ea3', yerr=generic_std)


# add some text for labels, title and axes ticks
ax.set_xlabel('Randomly selectedFive diseases')
ax.set_ylabel('F1 score')
# ax.set_title('Comparision Between Generic Model and Indivisual model', fontsize=16, fontweight='bold')
ax.set_xticks(ind + 1.5)
ax.set_xticklabels(DetectModel.keys())
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)
ax.legend((Base[0], A1extra5[0], A1extra15[0], A3extra5[0], B1extra5[0], B1extra15[0], B3extra5[0], Disease[0], Generic[0]),
          ('Base5', 'Base5+1Extra5', 'Base+1Extra15', 'Base+3Extra5', 'Base+1Extra5', 'Base+1Extra15', 'Base+3Extra5', 'Indivisual', 'Generic'),  ncol=2)
ax.set_ylim([0.0, 1.0])
ax.grid(True)
plt.show()
