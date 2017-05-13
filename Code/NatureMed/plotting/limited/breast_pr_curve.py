import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
ExperimentsResults = os.path.join(BaseDir, 'NatureData', 'YuanpuData', 'Experiments', 'evaluation_pinjun')

ModelNames = ['Breast5', 'Breast5Esophagus5', 'Breast5Esophagus15', 'Breast5Esophagus5Pancreas5Eye5', 'Breast5Uterus5',
              'Breast5Uterus15', 'Breast5Uterus5Lung5HeadNeck5', 'Breast', 'All']
surfix = '_multicontex_best_weights_res.json'
select_len = '_len_11_'
DiseaseName = 'Breast'
pr_dict = dict()

for ind, cur_model in enumerate(ModelNames):
    cur_f = os.path.join(ExperimentsResults, DiseaseName, cur_model+surfix)
    cur_results = json.load(open(cur_f))
    thresh_sums = cur_results.keys()
    filter_thresh = [thresh for thresh in thresh_sums if select_len in thresh]
    filter_thresh.sort(key=lambda x:float(x[x.find('_')+1:x.find('_len')]))
    cur_recall_list = [cur_results[thresh]['rec_mean'] for thresh in filter_thresh]
    cur_precision_list = [cur_results[thresh]['pre_mean'] for thresh in filter_thresh]

    if ind == 0:
        pr_dict['Base'] = dict()
        pr_dict['Base']['recall'] = cur_recall_list
        pr_dict['Base']['precision'] = cur_precision_list
    elif ind == 1:
        pr_dict['A1Extra5'] = dict()
        pr_dict['A1Extra5']['recall'] = cur_recall_list
        pr_dict['A1Extra5']['precision'] = cur_precision_list
    elif ind == 2:
        pr_dict['A1Extra15'] = dict()
        pr_dict['A1Extra15']['recall'] = cur_recall_list
        pr_dict['A1Extra15']['precision'] = cur_precision_list
    elif ind == 3:
        pr_dict['A3Extra5'] = dict()
        pr_dict['A3Extra5']['recall'] = cur_recall_list
        pr_dict['A3Extra5']['precision'] = cur_precision_list
    elif ind == 4:
        pr_dict['B1Extra5'] = dict()
        pr_dict['B1Extra5']['recall'] = cur_recall_list
        pr_dict['B1Extra5']['precision'] = cur_precision_list
    elif ind == 5:
        pr_dict['B1Extra15'] = dict()
        pr_dict['B1Extra15']['recall'] = cur_recall_list
        pr_dict['B1Extra15']['precision'] = cur_precision_list
    elif ind == 6:
        pr_dict['B3Extra5'] = dict()
        pr_dict['B3Extra5']['recall'] = cur_recall_list
        pr_dict['B3Extra5']['precision'] = cur_precision_list
    elif ind == 7:
        pr_dict['Indivisual'] = dict()
        pr_dict['Indivisual']['recall'] = cur_recall_list
        pr_dict['Indivisual']['precision'] = cur_precision_list
    elif ind == 8:
        pr_dict['Generic'] = dict()
        pr_dict['Generic']['recall'] = cur_recall_list
        pr_dict['Generic']['precision'] = cur_precision_list

# fig, ax = plt.subplots(figsize=(10, 8))
# line0, = ax.plot(pr_dict['Base']['recall'], pr_dict['Base']['precision'], linewidth=3, color='#e54c4c', label='Base5')
# line1, = ax.plot(pr_dict['BaseExtra5']['recall'], pr_dict['BaseExtra5']['precision'], linewidth=3, color='#99f052', label='Base5+Extra5')
# line2, = ax.plot(pr_dict['BaseExtra15']['recall'], pr_dict['BaseExtra15']['precision'], linewidth=3, color='#5f5f5f', label='Base5+Extra15')
# line3, = ax.plot(pr_dict['BaseExtra35']['recall'], pr_dict['BaseExtra35']['precision'], linewidth=3, color='#99a3eb', label='Base5+3Extra*5')
# line4, = ax.plot(pr_dict['Indivisual']['recall'], pr_dict['Indivisual']['precision'], linewidth=3, color='#4c4cc2', label='Indivisual')
# line5, = ax.plot(pr_dict['Generic']['recall'], pr_dict['Generic']['precision'], linewidth=3, color='#4cb399', label='Generic')
fig, ax = plt.subplots(figsize=(10, 8))
line0, = ax.plot(pr_dict['Base']['recall'], pr_dict['Base']['precision'], linewidth=3, color='#e41a1c', label='Base5')
line1, = ax.plot(pr_dict['A1Extra5']['recall'], pr_dict['A1Extra5']['precision'], linewidth=3, color='#369ead', label='A1Extra5')
line2, = ax.plot(pr_dict['A1Extra15']['recall'], pr_dict['A1Extra15']['precision'], linewidth=3, color='#7f6084', label='A1Extra15')
line3, = ax.plot(pr_dict['A3Extra5']['recall'], pr_dict['A3Extra5']['precision'], linewidth=3, color='#a2d1c5', label='A3Extra5')
line4, = ax.plot(pr_dict['B1Extra5']['recall'], pr_dict['B1Extra5']['precision'], linewidth=3, color='#369ead', label='B1Extra5', linestyle='--')
line5, = ax.plot(pr_dict['B1Extra15']['recall'], pr_dict['B1Extra15']['precision'], linewidth=3, color='#7f6084', label='B1Extra15', linestyle='--')
line6, = ax.plot(pr_dict['B3Extra5']['recall'], pr_dict['B3Extra5']['precision'], linewidth=3, color='#a2d1c5', label='B3Extra5', linestyle='--')
line7, = ax.plot(pr_dict['Indivisual']['recall'], pr_dict['Indivisual']['precision'], linewidth=3, color='#1f78e4', label='Indivisual')
line8, = ax.plot(pr_dict['Generic']['recall'], pr_dict['Generic']['precision'], linewidth=3, color='#984ea3', label='Generic')


ax.set(xlabel='Recall', ylabel='Precision')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.legend(loc='lower left')
plt.title('Comparison of different models on Breast', fontsize=16, fontweight='bold')
plt.show()
