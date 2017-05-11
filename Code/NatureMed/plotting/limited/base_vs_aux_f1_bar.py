import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

ExperimentsResults = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection',
                                  'NatureData', 'YuanpuData', 'Experiments', 'evaluation_pinjun')
DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)


DetectModel = dict()
DetectModel['Cervix'] = ['CervixBase', 'CervixLiver5', 'CervixLiver15', 'Cervix3Extra', 'Cervix', 'All']
DetectModel['Colorectal'] = ['ColorectalBase', 'ColorectalEye5', 'ColorectalEye15', 'Colorectal3Extra', 'Colorectal', 'All']
DetectModel['Thymus'] = ['ThymusBase', 'ThymusThyroid5', 'ThymusThyroid15', 'Thymus3Extra', 'Thymus', 'All']

base_means, base_std = [], []
disease5_means, disease5_std = [], []
disease15_means, disease15_std = [], []
disease35_means, disease35_std = [], []
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
            disease5_means.append(cur_f1score)
            disease5_std.append(cur_f1std)
        elif ind == 2:
            disease15_means.append(cur_f1score)
            disease15_std.append(cur_f1std)
        elif ind == 3:
            disease35_means.append(cur_f1score)
            disease35_std.append(cur_f1std)
        elif ind == 4:
            disease_means.append(cur_f1score)
            disease_std.append(cur_f1std)
        elif ind == 5:
            generic_means.append(cur_f1score)
            generic_std.append(cur_f1std)


# Drawing
bar_width = 0.25                   # the width of the bars
ind = np.arange(0, 2*len(DetectModel.keys()), 2)        # the x locations for the groups

start_m = (2 - bar_width * 6) / 2
start_w = start_m + bar_width
fig, ax = plt.subplots(figsize=(20, 15))

# pdb.set_trace()
base = ax.bar(ind+start_m*1, base_means, bar_width, color='#369ead', yerr=base_std)
disease5 = ax.bar(ind+start_m*2, disease5_means, bar_width, color='#c24642', yerr=disease5_std)
disease15 = ax.bar(ind+start_m*3, disease15_means, bar_width, color='#7f6084', yerr=disease15_std)
disease35 = ax.bar(ind+start_m*4, disease35_means, bar_width, color='#86b402', yerr=disease35_std)
disease = ax.bar(ind+start_m*5, disease_means, bar_width, color='#a2d1c5', yerr=disease_std)
generic = ax.bar(ind+start_m*6, generic_means, bar_width, color='#c8b631', yerr=generic_std)


# add some text for labels, title and axes ticks
ax.set_xlabel('Randomly selected three diseases')
ax.set_ylabel('F1 score')
# ax.set_title('Comparision Between Generic Model and Indivisual model', fontsize=16, fontweight='bold')
ax.set_xticks(ind + 1.0)
ax.set_xticklabels(DetectModel.keys())
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)
ax.legend((base[0], disease5[0], disease15[0], disease35[0], disease[0], generic[0]),
          ('Base', 'Base+1Extra5', 'Base+1Extra15', 'Base+3Extra*5', 'Indivisual', 'Generic'))
ax.set_ylim([0.0, 1.0])
ax.grid(True)
plt.show()
