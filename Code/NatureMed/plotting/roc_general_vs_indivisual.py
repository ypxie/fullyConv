import os, sys, pdb
import numpy as np
import glob, shutil, json

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
    pr_dict[cur_d]['generic']['fpr'] = []
    pr_dict[cur_d]['generic']['tpr'] = []
    pr_dict[cur_d]['indivisual']['fpr'] = []
    pr_dict[cur_d]['indivisual']['tpr'] = []

    glob_path = ExperimentsResults + '/' + cur_d + '/*.json'
    f_res = glob.glob(glob_path)

    for i_f in f_res:
        f_res_name = os.path.basename(i_f)
        model_name = f_res_name[0:f_res_name.find('_')]
        model_results = json.load(open(i_f))

        thresh_sums = model_results.keys()
        filter_thresh = [thresh for thresh in thresh_sums if select_len in thresh]
        filter_thresh.sort(key=lambda x:float(x[x.find('_')+1:x.find('_len')]))
        cur_tpr_list = []
        cur_fpr_list = []
