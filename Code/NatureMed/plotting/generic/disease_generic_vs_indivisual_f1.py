import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

ExperimentsResults = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection',
                                  'NatureData', 'YuanpuData', 'Experiments', 'evaluation')
DiseaseNames = []
for root, dirs, _ in os.walk(ExperimentsResults):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)

generic_means, genderic_std = [], []
indivisual_means, indivisual_std = [], []

for disease_name in DiseaseNames:
    glob_path = ExperimentsResults + '/' + disease_name + '/*.json'
    f_res = glob.glob(glob_path)
    for i_f in f_res:
        f_res_name = os.path.basename(i_f)
        model_name = f_res_name[0:f_res_name.find('_')]
        model_results = json.load(open(i_f))
        best_acc = 0.0
        accompany_std = 10.e10
        for i_t in model_results.keys():
            if model_results[i_t]['f1_mean'] > best_acc:
                best_acc = model_results[i_t]['f1_mean']
                accompany_std = model_results[i_t]['f1_std']
        if model_name == 'All':
            generic_means.append(best_acc)
            genderic_std.append(accompany_std)
        elif model_name == disease_name:
            indivisual_means.append(best_acc)
            indivisual_std.append(accompany_std)

# indivisual_means = generic_means
# indivisual_std = genderic_std

# Drawing
bar_width = 0.35                   # the width of the bars
ind = np.arange(DiseaseNum)        # the x locations for the groups
start_m = (1 - bar_width * 2) / 2
start_w = start_m +bar_width
fig, ax = plt.subplots()

# pdb.set_trace()
generic = ax.bar(ind+start_m, generic_means, bar_width, color='#9cb9da', yerr=genderic_std)
indivisual = ax.bar(ind+start_w, indivisual_means, bar_width, color='#f0c287', yerr=indivisual_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('F1 score')
ax.set_title('Comparision Between Generic Model and Indivisual model', fontsize=16, fontweight='bold')
ax.set_xticks(ind + 1.0 / 2)
ax.set_xticklabels(DiseaseNames)
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)
ax.legend((generic[0], indivisual[0]), ('Generic', 'Individual'))
ax.grid(True)
# fig.tight_layout()
plt.show()
