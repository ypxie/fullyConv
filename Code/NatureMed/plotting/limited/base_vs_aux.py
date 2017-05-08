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

###
cur_disease = 'Thymus'
glob_path = ExperimentsResults + '/' + cur_disease + '/*.json'
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
    print("Model name: {}".format(model_name))
    print("mean f1 is {}, f1 std is {}".format(best_acc, accompany_std))
    # if model_name == 'All':
    #     generic_means.append(best_acc)
    #     genderic_std.append(accompany_std)
    # elif model_name == disease_name:
    #     indivisual_means.append(best_acc)
    #     indivisual_std.append(accompany_std)
