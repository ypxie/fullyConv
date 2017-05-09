import os, sys, pdb
import numpy as np
import glob, shutil, json
import matplotlib.pyplot as plt

select_len = '_len_11_'
ExperimentsResults = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection',
                                  'NatureData', 'YuanpuData', 'Experiments', 'evaluation')
DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)

pr_dict = dict()
for cur_d in DiseaseNames:
    pr_dict[cur_d] = dict()
    pr_dict[cur_d]['generic'] = dict()
    pr_dict[cur_d]['indivisual'] = dict()
    pr_dict[cur_d]['generic']['recall'] = []
    pr_dict[cur_d]['generic']['precision'] = []
    pr_dict[cur_d]['indivisual']['recall'] = []
    pr_dict[cur_d]['indivisual']['precision'] = []

    glob_path = ExperimentsResults + '/' + cur_d + '/*.json'
    f_res = glob.glob(glob_path)

    for i_f in f_res:
        f_res_name = os.path.basename(i_f)
        model_name = f_res_name[0:f_res_name.find('_')]
        model_results = json.load(open(i_f))

        thresh_sums = model_results.keys()
        filter_thresh = [thresh for thresh in thresh_sums if select_len in thresh]
        filter_thresh.sort(key=lambda x:float(x[x.find('_')+1:x.find('_len')]))
        cur_recall_list = [model_results[thresh]['rec_mean'] for thresh in filter_thresh]
        cur_precision_list = [model_results[thresh]['pre_mean'] for thresh in filter_thresh]

        if model_name == 'All':
            pr_dict[cur_d]['generic']['recall'] = cur_recall_list
            pr_dict[cur_d]['generic']['precision'] = cur_precision_list
        elif model_name == cur_d:
            pr_dict[cur_d]['indivisual']['recall'] = cur_recall_list
            pr_dict[cur_d]['indivisual']['precision'] = cur_precision_list


## Drawing
plt.rc('lines', linewidth=2)
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(24, 10))
# fig.suptitle('Precision Recall Curve of Generic Model and Indivisual Model', fontsize=16, fontweight='bold')
# draw generic f1-score all diseases
for cur_d in pr_dict.keys():
    ax0.plot(pr_dict[cur_d]['generic']['recall'], pr_dict[cur_d]['generic']['precision'], color='k')
    ax1.plot(pr_dict[cur_d]['indivisual']['recall'], pr_dict[cur_d]['indivisual']['precision'], color='k')

ax0.set_xlim([0.0, 1.0])
ax0.set_ylim([0.0, 1.0])
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax0.set(title='Generic Model', xlabel='Recall', ylabel='Precision')
ax1.set(title='Indivisual Model', xlabel='Recall')
ax0.yaxis.label.set_size(14)
ax0.xaxis.label.set_size(14)
ax1.yaxis.label.set_size(14)
ax1.xaxis.label.set_size(14)
ax0.text(-0.1, 0.95, 'A', transform=ax0.transAxes, size=16, weight='bold')
ax1.text(-0.1, 0.95, 'B', transform=ax1.transAxes, size=16, weight='bold')
ax0.grid(True)
ax1.grid(True)
# fig.tight_layout()

plt.show()
