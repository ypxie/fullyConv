import os, sys, pdb
import numpy as np
import glob, shutil, json
import matplotlib.pyplot as plt

select_len = '_len_11_'
ExperimentsResults = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection',
                                  'NatureData', 'YuanpuData', 'Experiments', 'evaluation')
# Find out all diseases
DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)
assert DiseaseNum == 25

# Collecting F1-score across all threshold values for all disease
pr_dict = dict()
all_generic = []
all_indivisual = []
for cur_d in DiseaseNames:
    pr_dict[cur_d] = dict()
    pr_dict[cur_d]['generic'] = dict()
    pr_dict[cur_d]['indivisual'] = dict()
    pr_dict[cur_d]['generic']['thresh'] = []
    pr_dict[cur_d]['generic']['f1score'] = []
    pr_dict[cur_d]['indivisual']['thresh'] = []
    pr_dict[cur_d]['indivisual']['f1score'] = []

    glob_path = ExperimentsResults + '/' + cur_d + '/*.json'
    f_res = glob.glob(glob_path)

    for i_f in f_res:
        f_res_name = os.path.basename(i_f)
        model_name = f_res_name[0:f_res_name.find('_')]
        model_results = json.load(open(i_f))

        thresh_sums = model_results.keys()
        filter_thresh = [thresh for thresh in thresh_sums if select_len in thresh]
        filter_thresh.sort(key=lambda x:float(x[x.find('_')+1:x.find('_len')]))

        thresh_list = []
        f1score_list = []
        for cur_key in filter_thresh:
            thresh_list.append(cur_key[cur_key.find('_')+1:cur_key.find('_len')])
            f1score_list.append(model_results[cur_key]['f1_mean'])

        # pdb.set_trace()
        if model_name == 'All':
            pr_dict[cur_d]['generic']['thresh'] = thresh_list
            pr_dict[cur_d]['generic']['f1score'] = f1score_list
            all_generic.append(f1score_list)
        elif model_name == cur_d:
            pr_dict[cur_d]['indivisual']['thresh'] = thresh_list
            pr_dict[cur_d]['indivisual']['f1score'] = f1score_list
            all_indivisual.append(f1score_list)

# for cur_d in DiseaseNames:
#     cur_f1 = pr_dict[cur_d]['generic']['f1score']
#     # print("Current disease: {}, length of f1: {}, values are: {}".format(cur_d, len(cur_f1), cur_f1))
#     print("Current disease: {}, length of f1: {}".format(cur_d, len(cur_f1)))
# pdb.set_trace()

## Drawing
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(24, 10))
# fig.suptitle('F1 Score of Generic Model and Indivisual Across Threshold', fontsize=16, fontweight='bold')
# draw generic f1-score all diseases
for cur_d in pr_dict.keys():
    ax0.plot(pr_dict[cur_d]['generic']['thresh'], pr_dict[cur_d]['generic']['f1score'], color='k', linewidth=2)
    ax1.plot(pr_dict[cur_d]['indivisual']['thresh'], pr_dict[cur_d]['indivisual']['f1score'], color='k', linewidth=2)

#draw average
avg_generic = [np.average(x) for x in zip(*all_generic)]
std_generic = [np.std(x) for x in zip(*all_generic)]
avg_indivisual = [np.average(x) for x in zip(*all_indivisual)]
std_indivisual = [np.std(x) for x in zip(*all_indivisual)]
# ax0.errorbar(thresh_list, avg_generic, yerr=std_generic, color='r', linewidth=3, zorder=10)
# ax1.errorbar(thresh_list, avg_indivisual, yerr=std_indivisual, color='r', linewidth=3, zorder=10)
ax0.errorbar(thresh_list, avg_generic, color='r', linewidth=3, zorder=10)
ax1.errorbar(thresh_list, avg_indivisual, color='r', linewidth=3, zorder=10)


diseaseArtist = plt.Line2D((0,1), (0,0), color='k', linewidth=2)
avgArtist = plt.Line2D((0,1), (0,0), color='r', linewidth=2)

#Create legend from custom artist/label lists
handles0, labels0 = ax0.get_legend_handles_labels()
ax0.legend([handle for i,handle in enumerate(handles0) if i in [0, 1]]+[diseaseArtist, avgArtist],
          [label for i,label in enumerate(labels0) if i in [0, 1]]+['Single Disease', 'Average Performance'], loc='lower left')
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend([handle for i,handle in enumerate(handles1) if i in [0, 1]]+[diseaseArtist, avgArtist],
          [label for i,label in enumerate(labels1) if i in [0, 1]]+['Single Disease', 'Average Performance'], loc='lower left')

ax0.set_xlim([0.0, 1.0])
ax0.set_ylim([0.0, 1.0])
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax0.set(title='Generic Model', xlabel='Threshold', ylabel='F1 score')
ax1.set(title='Indivisual Model', xlabel='Threshold')
ax0.yaxis.label.set_size(14)
ax0.xaxis.label.set_size(14)
ax1.yaxis.label.set_size(14)
ax1.xaxis.label.set_size(14)
ax0.text(0.05, 0.95, 'A', transform=ax0.transAxes, size=16, weight='bold')
ax1.text(0.05, 0.95, 'B', transform=ax1.transAxes, size=16, weight='bold')
ax0.grid(True)
ax1.grid(True)
plt.show()
