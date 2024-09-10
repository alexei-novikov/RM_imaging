import numpy as np
#import torchvision
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
from torchmetrics import TotalVariation
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import TrainingFunctions_Jan25version as ttf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



sweep_config = {
    'method': 'grid',
    "metric": {"goal": "minimize", "name": 'validation loss'}
}

 
grid = [0.25, 0.025, 0.0025, 0.00025, 0.000025]

parameters_dict = {
    'super_param' :{'values':['raw']},#full_cint, cc,  raw, cc_5e-1_5e-1, cc_5e-1_5e-1_full ,'cc_75e-2_5e-2','cc_75e-2_75e-2_full'
    #'weights' : {'values': [[1.0, i, j] for i in grid for j in grid]},
    'labeled_loss': {'values': ['MSE']},#,'MSE_target', 'TV'], ['MSE','TV', 'L1_penalty'], ['MSE','L1', 'TV']]},
    'weight_decay' :{'values':[.01]},#weightt decay during gradient descent. \theta_n+1=weight_decay*theta_n-grad_L(theta_n)
    'hidden_dims': {
        'values' :[[5000,2500,1000,500]]#,[1000],[1000,500],[1000,1000],[1000,1000,500],[2500],[2500,500],[2500,1000],[2500,1000,500],[2500,2500],[5000],[5000,500],[5000,1000],[5000,2500],[5000,5000],[5000,1000,500],[5000,2500,1000],[5000,2500,1000,500]]
        
        
        # [[2000, 500],[2000],[2000,1000],[4000,500],[4000],[4000,2000],[4000,500]]#[[1000, 500],[1000],[1000,1000,500]]
        #[[10000],[20000],[10000,10000],[5000,5000],[5000,5000,5000],[1500,3000,3000,3000,900],[20000,20000],[10000,10000,10000],[5000],[20000,20000,20000],[1000,2000,2000,2000,1000,1000,600,600,1000,1000,2000,3000,2000,1000,1000,2000,1400,1400,2000]]
    },             
    'num_epochs' :{ 'values': [150] },#number of data passes
    'unlabled_data_used': { #amouth of unlabeled data use
        'values': [0]
    },
    'front_load_n': {#number of passes of labeled before moving to unlabeled data
        'values': [0]
    },
    'extra_val':{'values':[False]},#whether or not to use extra validation sets
    'two_opts': {
        'values': [False]#to use one or two optimizers
    },
    'scheduler': {
        'values': ['ReduceLROnPlateau']#Learning rate scheduler
    },
    "lr": {'values': [5e-1]},#.005, .
    'seed':{'values':[0]},
    "min_lr": {'values': [0]}, #minimum learning rate
    'batch_size': {
        'values' :[128]
    },
    'val_batch_size': {
        'values' :[128]
    },
    'weight': {
        'values' :[0]#unlabled L1 data weight
    },'TV_weight': {
        'values' :[0]#TV weight
    },    
    'TV_weight_labeled_data': {
        'values': [0]},
    'L1_weight_labeled_data': {
        'values': [0]},
    'data_type': {'values': ['p1_IIDreal_6e-1_all_seeds']},#'p1_IIDreal_6e-1_all_seeds','p1_400real_6e-1_all_seeds','p1_4000real_6e-1_all_seeds', ,'p1_20real_6e-1_all_seeds'
    'target': {'values':['rho']},
    'patience': {'values':[4]},#after how many epochs to update learning rate
    'gamma': {'values':[.3]},#how to update learing rate: learning_rate=gamma*learning_rate
    'sigma': {'values':[2]},#variance of gaussian filter
    'loss' : {'values': ['NA'] },
    'net_type': {'values':['fc']},#model type. 'fc', 'linear', 'conv' (conv has not been updated since 10/10/2023
    'change': {'values' : [1]},#how many passes of labeled data before moving to unlabled
    'normalize_output': {'values':[True]},#unneeded. Normalizes no matter what
    'model_checkpoint': {'values':[None]},#value to save model at (model always saved at end)
    'optim': {'values':['adam']}, #optimizer used. 'adam', 'SGD'
    'activation': {'values':['mod_relu']},#activation: mode_relu, relu, tanh, ect
    'linear_type': {'values':['complex']},#complex or real linear layers
    'bias': {'values':[True]},#to use a bias in layers (layer=Ax+bias)
    'run_name': {'values':['Repeat']},
    'threshold_val': {'values':['NA']},#No longer used
    'offset':{'values':['NA']},#No longer used
    'data_proportion': {'values': [.50]},#what amountof the data to use
    'invert' :{'values':[False]}#to use front or back of dataset. False means use data as [x1,x2,x3,...,xn]. True is use data as [xn, xn-1, xn-2,...,x3,x2,x1]
}
 
                    
sweep_config['parameters'] = parameters_dict








if __name__ == "__main__":
    if True:
        key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa'
        parameters_dict['key']={'values':[key]}
        project="Preprocessing_final_models_seeds"
        sweep_id = wandb.sweep(sweep_config, project=project)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.agent(sweep_id, ttf.train, count=1)

    else:
        ttf.train(ttf.reformat_sweep_for_1_run(parameters_dict))