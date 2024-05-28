import numpy as np
#import torchvisionO
from datetime import datetime
import os
import scipy.io  
from torch.utils.data import DataLoader
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import math
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import torch
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
import Models as M
import Helpers as H
import Cleancoder as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa'
        #parameters_dict['key']={'values':[key]}
    wandb.login(key=key)# 89a70fbc572a495206df640bd6c9cbf2a4a0dcaa

    project='Fully unlab 2.0'



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweep_id ='rlbur2wk'
    wandb.agent(sweep_id, C.encoder_decoder, project=project,count=1)
