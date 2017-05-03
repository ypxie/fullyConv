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

trainingDataroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureData', 'PingjunData', 'TrainingData', 'Thymus')
validationDataroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureData','PingjunData', 'ValidationData', 'Thymus')

# modelroot = os.path.join(projroot, 'Data','NatureModel','YuanpuModel')
modelroot = os.path.join(HomeDir, 'Dropbox', 'GenericCellDetection', 'NatureModel', 'PingjunModel', 'Thymus')

training_pool = np.array([('ThymusBase'), ('Thymus1'), ('Thymus2'), ('Thymus3')])

show_progress = 0
processes = []
Totalnum = len(training_pool)


process_size = 2
device_pool = [1, 1]


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
