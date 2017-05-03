import sys, os
import numpy as np
projroot   = os.path.join('..','..','..')
coderoot   = os.path.join(projroot, 'Code')
home = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))
sys.path.insert(0, '..')

from torch_fcn.proj_utils.local_utils import Indexflow
from nature_train import train_worker
import torch.multiprocessing as mp

trainingDataroot = os.path.join(home,'Dropbox','DataSet', 'NatureData', 'YuanpuData', 'TrainingData')
validationDataroot = os.path.join(home,'Dropbox','DataSet', 'NatureData','YuanpuData', 'ValidationData')

modelroot = os.path.join(projroot, 'Data','NatureModel','YuanpuModel')

training_pool = np.array([
                 ('All'),
                 ('AdrenalGland'),
                 ('Bladder'),
                 ('Breast'),
                 ('Colorectal'),
                 ('Eye'),
                 ('Kidney'),
                 ('Lung'),
                 ('Ovary'),
                 ('Pleura'),
                 ('Skin'),
                 ('Stomach'),
                 ('Thymus'),
                 ('Uterus'),
                 ('BileDuct'),
                 ('Brain'),
                 ('Cervix'),
                 ('Esophagus'),
                 ('HeadNeck'),
                 ('Liver'),
                 ('LymphNodes'),
                 ('Pancreas'),
                 ('Prostate'),
                 ('SoftTissue'),
                 ('Testis'),
                 ('Thyroid')
                ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

process_size = 9
device_pool = [3,3,2,2,2,2,0,0,0]

for select_ind in Indexflow(Totalnum, process_size, random=False):
    select_pool = training_pool[select_ind]
    print(select_pool)
    for idx, (dataset, device) in enumerate(zip(select_pool,device_pool)):
        p = mp.Process(target=train_worker, args=(trainingDataroot, validationDataroot, dataset, modelroot, device, show_progress))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with',dataset)
