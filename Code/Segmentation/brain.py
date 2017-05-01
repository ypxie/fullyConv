import sys, os
import numpy as np
projroot   = os.path.join('..','..')
coderoot   = os.path.join(projroot, 'Code')
sys.path.insert(0, os.path.join(coderoot))
from segmentation import train_worker

home = os.path.expanduser('~')
trainingDataroot = os.path.join(home,'DataSet', 'FY_TMI', 'train')
modelroot = os.path.join(projroot, 'Data','Model')

train_worker(trainingDataroot = trainingDataroot, trainingset= 'brain',
             show_progress=True, modelroot=modelroot, modelsubfolder = 'multiout')