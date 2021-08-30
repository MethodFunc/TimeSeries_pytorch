import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import os
import fnmatch
import glob
import time
import datetime
import re

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset

except ModuleNotFoundError:
    raise 'torch is not installed. please install torch'

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20, 8)

pd.options.display.precision = 4
pd.options.display.float_format = "{:.4f}".format


device = 'cuda' if torch.cuda.is_available() else 'cpu'
