import sys, os, pdb
import numpy as np
projroot   = os.path.join('..','..','..')
coderoot   = os.path.join(projroot, 'Code')
HomeDir = os.path.expanduser('~')
sys.path.insert(0, os.path.join(coderoot))
sys.path.insert(0, '..')

from torch_fcn.proj_utils.local_utils import Indexflow
from nature_train import train_worker
import torch.multiprocessing as mp

DiseaseName = 'Cervix'

trainingDataroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureData', 'PingjunData', 'TrainingData', DiseaseName)
validationDataroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureData','PingjunData', 'ValidationData', DiseaseName)

# modelroot = os.path.join(projroot, 'Data','NatureModel','YuanpuModel')
modelroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureModel', 'PingjunModel', DiseaseName)

pool_collection = []
task_num = 4
for ind in range(task_num):
    if ind == 0:
        pool_collection.append((DiseaseName + 'Base'))
    else:
        pool_collection.append((DiseaseName + str(ind)))

# training_pool = np.array([('ThymusBase'), ('Thymus1'), ('Thymus2'), ('Thymus3')])
training_pool = np.array(pool_collection)

show_progress = 0
processes = []
Totalnum = len(training_pool)


process_size = 2
device_pool = [0, 0]


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
