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
    pr_dict[cur_d]['generic']['thresh'] = []
    pr_dict[cur_d]['generic']['f1score'] = []
    pr_dict[cur_d]['indivisual']['fpr'] = []
    pr_dict[cur_d]['indivisual']['tpr'] = []
