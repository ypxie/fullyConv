import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

ExperimentsResults = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection',
                                  'NatureData', 'YuanpuData', 'Experiments', 'evaluation_pinjun')
ModelNames = ['ColorectalBase', 'ColorectalEye5', 'ColorectalEye15', 'Colorectal3Extra', 'Colorectal', 'All']
surfix = '_multicontex_best_weights_res.json'
select_len = '_len_11_'
DiseaseName = 'Colorectal'
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
        pr_dict['BaseExtra5'] = dict()
        pr_dict['BaseExtra5']['recall'] = cur_recall_list
        pr_dict['BaseExtra5']['precision'] = cur_precision_list
    elif ind == 2:
        pr_dict['BaseExtra15'] = dict()
        pr_dict['BaseExtra15']['recall'] = cur_recall_list
        pr_dict['BaseExtra15']['precision'] = cur_precision_list
    elif ind == 3:
        pr_dict['BaseExtra35'] = dict()
        pr_dict['BaseExtra35']['recall'] = cur_recall_list
        pr_dict['BaseExtra35']['precision'] = cur_precision_list
    elif ind == 4:
        pr_dict['Indivisual'] = dict()
        pr_dict['Indivisual']['recall'] = cur_recall_list
        pr_dict['Indivisual']['precision'] = cur_precision_list
    elif ind == 5:
        pr_dict['Generic'] = dict()
        pr_dict['Generic']['recall'] = cur_recall_list
        pr_dict['Generic']['precision'] = cur_precision_list

fig, ax = plt.subplots()
line0, = ax.plot(pr_dict['Base']['recall'], pr_dict['Base']['precision'], linewidth=3, color='#e54c4c', label='Base')
line1, = ax.plot(pr_dict['BaseExtra5']['recall'], pr_dict['BaseExtra5']['precision'], linewidth=3, color='#99f052', label='BaseExtra5')
line2, = ax.plot(pr_dict['BaseExtra15']['recall'], pr_dict['BaseExtra15']['precision'], linewidth=3, color='#5f5f5f', label='BaseExtra15')
line3, = ax.plot(pr_dict['BaseExtra35']['recall'], pr_dict['BaseExtra35']['precision'], linewidth=3, color='#99a3eb', label='BaseExtra35')
line4, = ax.plot(pr_dict['Indivisual']['recall'], pr_dict['Indivisual']['precision'], linewidth=3, color='#b34c70', label='Indivisual')
line5, = ax.plot(pr_dict['Generic']['recall'], pr_dict['Generic']['precision'], linewidth=3, color='#4c4cc2', label='Generic')

ax.set(xlabel='Recall', ylabel='Precision')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.2, 1.0])
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.legend(loc='lower left')
plt.show()
