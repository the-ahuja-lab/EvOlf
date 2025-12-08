import os
import csv
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc

from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_value = 42
rs = RandomState(MT19937(SeedSequence(seed_value))) 
np.random.seed(seed_value)
batch_size = 64

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def writeIntoFile(file_path, file_name, content, onlyIds = False):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        if onlyIds == True:
            writer.writerow(['ID'])
            for row in range(len(content)):
                writer.writerow([content[row]])
        else:
            for row in range(len(content)):
                writer.writerow(content[row])

def pickleDump(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    with open(file_path + file_name, 'wb') as f:
        pickle.dump(content, f)

def pickleRead(file_path, file_name):
    with open(file_path + file_name, 'rb') as f:
        data = pickle.load(f)
    return data


import warnings
warnings.filterwarnings('ignore')