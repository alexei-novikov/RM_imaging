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
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
import Models as M
import Helpers as H
import Cleancoder as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



sweep_config = {
    'method': 'grid',
    "metric": {"goal": "minimize", "name": 'validation loss'}
}
#Coherence regime data_types 'PNAS-highcoh_regime_all_seeds','PNAS-regime_all_seeds','PNAS-lowcoh_regime_all_seeds','PNAS-regime_all_seeds]

#L1=[500, 750, 1000,950, 900, 850, 800, 400, 400, 400, 400, 800, 1000]
L1=[800,750, 700, 650, 600, 300,  300, 300, 300,  300, 300, 300, 400, 800, 1200]
#L1_old_data=[1000,950, 900, 850, 800, 400, 400, 400, 400, 800, 1000]
parameters_dict = {
    'hidden_dims': {
        'values' :[[500,500, 500, 300, 3000, 400,800]],#[1000,950, 900, 850, 800, 400, 400, 400, 400, 800, 1000]]#[[encoder], [decoder]] 
    },          
    'dropout':{'values':[.25]}, #default 0
    'num_epochs' :{ 'values': [1500] },#number of data passes
    'seed':{'values':[0]},
    'data_type': {'values': ['PNAS-regime_all_seeds']},#PNAS-highcoh_regime_all_seeds,PNAS-regime_all_seeds,PNAS-lowcoh_regime_all_seeds,PNAS-regime_all_seeds
    'net_type': {'values':['NL_L']},#model type.'fc_NC', recurrent_out' 'fc', 'linear', 'conv' (conv has not been updated since 10/10/2023
    'labeled_data': {'values': [0]},#what amountof the data to use
    'unlabeled_data' : {'values' :[5000]},#what amount of the data to use
    'Data_locat':{'values':['Low_data_GELMA']},#where the data is located
    'l1_weight' :{'values': [1.0]},
    'out_encoder':{'values':['sigmoid']},#Sigmoid
    'L1_rescaling':{'values':[False]},
    'L1_burn_time':{'values':[20]},
    'activation':{'values':['relu']},#relu sigmoidnvidi
    'G_0_intiailization':{'values':[True]},
    'Dict_first_epochs':{'values':[0]},
    'weight_decay':{'values':[0.01]},
    'sch':{'values':[False]},#StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
    'optimizer':{'values':['AdamW']},#Adam, SGD
    'Final_batch':{'values':[True]},
    'batch_size':{'values':[128]},
    'lin_type_decoder':{'values':['real']},#real, complex
    'root_MSE':{'values':[True]},
    'GELMA':{'values':[5e-7]},
    'lr':{'values':[1e-3]},
}

#N alpha/B should be constatnt











dictionary_config = {
    'method': 'grid',
    "metric": {"goal": "minimize", "name": 'validation loss'}
}

dictionary_config['parameters'] = parameters_dict




dictionary_config['project']={'values':['Lab VS ULab']}


datagraph_list=[]

if __name__ == "__main__":
    #dateTimeObj = datetime.now()
    #timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
    #parameters_dict['run_name']={'values':timestampStr}
    EXP='dictionary_final'

    if False:
        C.encoder_decoder(H.reformat_sweep_for_1_run(parameters_dict))                        
    elif EXP=='dictionary_final':
        key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa'
        #parameters_dict['key']={'values':[key]}
        wandb.login(key=key)# 89a70fbc572a495206df640bd6c9cbf2a4a0dcaa

        project='Fully unlab 2.0'
        sweep_id = wandb.sweep(dictionary_config, project=project)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.agent(sweep_id, C.encoder_decoder, count=50)
