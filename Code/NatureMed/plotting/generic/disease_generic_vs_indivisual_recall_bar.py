import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import glob, json

BaseDir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'GenericCellDetection')
ExperimentsResults = os.path.join(BaseDir, 'NatureData', 'YuanpuData', 'Experiments')
ResultsTest = os.path.join(ExperimentsResults, 'evaluation')
ResultsValidation = os.path.join(ExperimentsResults, 'evaluation_validation')

DiseaseNames = []
for root, dirs, _ in os.walk(ResultsValidation):
    for d in dirs:
        DiseaseNames.append(d)
DiseaseNum = len(DiseaseNames)

generic_means, genderic_std = [], []
indivisual_means, indivisual_std = [], []

for disease_name in DiseaseNames:
    glob_path = ResultsValidation + '/' + disease_name + '/*.json'
    f_res = glob.glob(glob_path)
    for i_f in f_res:
        f_res_name = os.path.basename(i_f)
        model_name = f_res_name[0:f_res_name.find('_')]
        val_results = json.load(open(i_f))

        best_f1 = 0.0
        best_key = ''

        for i_t in val_results.keys():
            if val_results[i_t]['f1_mean'] > best_f1:
                best_f1 = val_results[i_t]['f1_mean']
                best_key = i_t

        test_results = json.load(open(os.path.join(ResultsTest, disease_name, f_res_name)))
        if model_name == 'All':
            generic_means.append(test_results[best_key]['rec_mean'])
            genderic_std.append(test_results[best_key]['rec_std'])
        elif model_name == disease_name:
            indivisual_means.append(test_results[best_key]['rec_mean'])
            indivisual_std.append(test_results[best_key]['rec_std'])

# Drawing
bar_width = 0.35                   # the width of the bars
ind = np.arange(DiseaseNum)        # the x locations for the groups
start_m = (1 - bar_width * 2) / 2
start_w = start_m +bar_width
fig, ax = plt.subplots(figsize=(40, 24))

generic = ax.bar(ind+start_m, generic_means, bar_width, color='#9cb9da', yerr=genderic_std)
indivisual = ax.bar(ind+start_w, indivisual_means, bar_width, color='#f0c287', yerr=indivisual_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Recall')
ax.set_xlabel('25 diseases')
ax.set_ylim([0.0, 1.1])
# ax.set_title('Comparision Between Generic Model and Indivisual model', fontsize=16, fontweight='bold')
ax.set_xticks(ind + 1.0 / 2)
ax.set_xticklabels(DiseaseNames)
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14)
ax.legend((generic[0], indivisual[0]), ('Generic', 'Individual'))
ax.grid(True)
plt.show()
