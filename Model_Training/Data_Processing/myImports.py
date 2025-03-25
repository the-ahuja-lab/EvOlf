import os
import csv
import copy
import time
import pickle
import random
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
from sklearn.calibration import calibration_curve

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

from os import listdir
from os.path import isfile, join

from imblearn.over_sampling import SMOTE
smote = SMOTE()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import torch.backends.cudnn


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed_value = 42):
    rs = RandomState(MT19937(SeedSequence(seed_value))) 
    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

import warnings
warnings.filterwarnings('ignore')