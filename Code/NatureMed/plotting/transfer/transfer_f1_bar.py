import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
ExperimentsResults = os.path.join(BaseDir, 'NatureData', 'YuanpuData', 'Experiments', 'evaluation_other')

DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)

DetectModel = dict()
DetectModel['BM'] = ['All_multicontex_best', 'BM_multicontex_ind', 'BM_multicontex']
DetectModel['brain'] = ['All_multicontex_best', 'brain_multicontex_ind', 'brain_multicontex']
DetectModel['breast'] = ['All_multicontex_best', 'breast_multicontex_ind', 'breast_multicontex']
DetectModel['NET'] = ['All_multicontex_best', 'NET_multicontex_ind', 'NET_multicontex']
DetectModel['phasecontrast'] = ['All_multicontex_best', 'phasecontrast_multicontex_ind', 'phasecontrast_multicontex']
ModelNum = len(DetectModel['BM'])

best_key = 'thresh_0.70_len_11_radius_16.00'
surfix = '_weights_res.json'

generic_means, generic_std = [], []
indivisual_means, indivisual_std = [], []
ft_means, ft_std = [], []


for cur_d in DetectModel.keys():
    for ind, cur_model in enumerate(DetectModel[cur_d]):
        cur_f = os.path.join(ExperimentsResults, cur_d, cur_model+surfix)

        cur_results = json.load(open(cur_f))
        cur_f1score = cur_results[best_key]['f1_mean']
        cur_f1std = cur_results[best_key]['f1_std']

        if ind == 0:
            generic_means.append(cur_f1score)
            generic_std.append(cur_f1std)
        elif ind == 1:
            indivisual_means.append(cur_f1score)
            indivisual_std.append(cur_f1std)
        elif ind == 2:
            ft_means.append(cur_f1score)
            ft_std.append(cur_f1std)

# Drawing
bar_width = 0.3                                    # the width of the bars
ind = np.arange(0, len(DetectModel.keys()), 1)    # the x locations for the groups
start_m = (1 - bar_width * ModelNum)
start_w = start_m + bar_width
fig, ax = plt.subplots(figsize=(20, 15))

generic = ax.bar(ind+start_m, generic_means, bar_width, color='#369ead', yerr=generic_std)
ft = ax.bar(ind+start_m + bar_width*1, ft_means, bar_width, color='#c24642', yerr=ft_std)
indivisual = ax.bar(ind+start_m + bar_width*2, indivisual_means, bar_width, color='#7f6084', yerr=indivisual_std)

# add some text for labels, title and axes ticks
ax.set_xlabel('Five non-TCGA diseases')
ax.set_ylabel('F1 score')

ax.set_xticks(ind + 0.5)
ax.set_xticklabels(DetectModel.keys())
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)
ax.legend((generic[0], ft[0], indivisual[0]), ('Generic', 'Fine-tune', 'Indivisual'))
ax.set_ylim([0.0, 1.0])
ax.grid(True)
plt.show()
