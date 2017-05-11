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

DiseaseName = 'Colorectal'
BaseFolder = '/data/Pingjun/CellDetectionData'

trainingDataroot = os.path.join(BaseFolder, 'NatureData', 'TrainingData', DiseaseName)
validationDataroot = os.path.join(BaseFolder, 'NatureData',  'ValidationData', DiseaseName)

# modelroot = os.path.join(projroot, 'Data','NatureModel','YuanpuModel')
modelroot = os.path.join(BaseFolder, 'NatureModel', DiseaseName)

# pool_collection = []
# task_num = 4
# for ind in range(task_num):
#     if ind == 0:
#         pool_collection.append((DiseaseName + 'Base'))
#     else:
#         pool_collection.append((DiseaseName + str(ind)))
# training_pool = np.array(pool_collection)
# training_pool = np.array([('ColorectalEye5'), ('ColorectalEye15'), ('Colorectal3Extra')])
## Bladder

# training_folders = []
# for root, dirs, _ in os.walk(trainingDataroot):
#     for d in dirs:
#         training_folders.append((d, ))
# training_pool = training_folders
# pdb.set_trace()
training_pool = np.array([('Colorectal5Testis5'), ('Colorectal5Testis15')])
show_progress = 0
processes = []
Totalnum = len(training_pool)

process_size = 2
device_pool = [2, 2]

for select_ind in Indexflow(Totalnum, process_size, random=False):
    select_pool = training_pool[select_ind]
    print(select_pool)
    for idx, (dataset, device) in enumerate(zip(select_pool,device_pool)):
        p = mp.Process(target=train_worker, args=(trainingDataroot, validationDataroot, dataset, modelroot,
                                                  device, show_progress, 'multicontex',True, 128))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with',dataset)
