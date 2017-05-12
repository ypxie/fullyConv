import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json


BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
ExperimentsResults = os.path.join(BaseDir, 'NatureData', 'YuanpuData', 'Experiments', 'evaluation_other')

ModelNames = ['All_multicontex_best', 'brain_multicontex_ind', 'brain_multicontex']
surfix = '_weights_res.json'
select_len = '_len_11_'
DiseaseName = 'brain'
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
        pr_dict['generic'] = dict()
        pr_dict['generic']['recall'] = cur_recall_list
        pr_dict['generic']['precision'] = cur_precision_list
    elif ind == 1:
        pr_dict['indivisual'] = dict()
        pr_dict['indivisual']['recall'] = cur_recall_list
        pr_dict['indivisual']['precision'] = cur_precision_list
    elif ind == 2:
        pr_dict['ft'] = dict()
        pr_dict['ft']['recall'] = cur_recall_list
        pr_dict['ft']['precision'] = cur_precision_list



fig, ax = plt.subplots(figsize=(10, 8))
line0, = ax.plot(pr_dict['generic']['recall'], pr_dict['generic']['precision'], linewidth=3, color='#e54c4c', label='Generic')
line1, = ax.plot(pr_dict['indivisual']['recall'], pr_dict['indivisual']['precision'], linewidth=3, color='#99f052', label='Indivisual')
line2, = ax.plot(pr_dict['ft']['recall'], pr_dict['ft']['precision'], linewidth=3, color='#5f5f5f', label='Fine-tuned')


ax.set(xlabel='Recall', ylabel='Precision')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.legend(loc='lower left')
plt.title('Comparison of different models on Brain', fontsize=16, fontweight='bold')
plt.show()
