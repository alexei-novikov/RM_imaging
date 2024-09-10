#Notebooks for small tests
import os  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
plt.rcParams['axes.facecolor']='w'
plt.rcParams['savefig.facecolor']='w'
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
#imports and plotting function
import argparse
import mat73
import logging
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
import sys
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import torch
from sklearn.preprocessing import StandardScaler
import Models as M
import Helpers as H
import time
import copy
from torch.func import vmap
from functorch.experimental import replace_all_batch_norm_modules_
from sklearn.cluster import AgglomerativeClustering


encoder_out='sigmoid'
cwd=os. getcwd()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")









data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-highcoh_regime_all_seeds/PNAS-highcoh_regime_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-lowcoh_regime_guassian_05perc_offgrid_all_seeds/PNAS-lowcoh_regime_guassian_05perc_offgrid_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-regime_all_seeds/PNAS-regime_seed0')

exp_name=f''


medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
print(medium.shape)
inners=medium.transpose().conjugate()@medium
print(inners.shape)
coherence=0
for i in inners:
    for j in i:
        if abs(j)>coherence and j<.99:
            coherence=abs(j)
print('coherence of data: ', coherence)
print('test')
#b=np.load(data_path+'/train/b.npy')
#rho=np.load(data_path+'/train/rho.npy')
#print(np.allclose(medium@rho[0],b[0]))
index_list=[]

torch.__version__ 
from sklearn.cluster import KMeans
import numpy as np




column_list=[]
index_list_list=[]

for EXP_NUM in range(1): #[33, 34, 35, 40, 41] BOTH THESE RUNNING NOW. old machine running 49, 40, 41 in tandem
    for layers in [[2048,1024],[2048,1024,512,256],[2048,1024,512,256,512,1024,2048],[256,256,256,256,2048],[1024]*5,[1024,512,256,128,128,128,4096]]:
        torch.manual_seed(EXP_NUM)
        #Raw data unlabeled 
        #SMALL NET, 5k data, 300 resets. CONIRMED THAT CONVERGES WITH TRUE INIT
        ##CONCLUSION: GELMA unscaled causes Gelma to blow up, causing L1 norm to go to 0. Scaled GELMA (i think) fixes this proble.
        starttime=time.time()
        target='raw'
        DICTIONARY_size='default'
        unlabeled_data=10000
        batchsize=128
        L1_weight_og=1e-3
        L1_weight=L1_weight_og
        L1_weight_final=L1_weight_og
        THRESH_HOLD_VAL=.001
        #layers=[4096,2048]
        #[500,500,500,300,300,400,800]#[3000,1000,500,300,300,300,300,300,300,300,300,300,300,300]#[3000,1500,1000,500,500,500,300,300,300,400,400,800]#[1000,1000,1000,500,500,500,300,300,300,400,400,400,800]
        GELMA_layers=layers#[500,500,500,300,300,400,800]
        CE=False
        KM_in=False
        INV_weight=0
        RESHUFFLE=False
        LR=1e-3 
        LR_final=LR
        GELMA_OG=1e-1
        GELMA=GELMA_OG
        GELMA_Final=GELMA_OG
        TIME_reverse=False
        E_list=0
        RESETS=0
        Epochs=10000
        index_list=[]
        GELMA_inc=0
        G_0=False
        Scheduling=False  #NEED TO CHANGE TO FALSE TO NOT SCHEDULE
        END_schedule=3000
        dateTimeObj = datetime.now()
        

        timestampStr = dateTimeObj.strftime("%H-%M-%S") 
        print('Timestamp: ', timestampStr)
        figsize,font_size=5, 25
        def soft_threh(x, th):
            return torch.sign(x)*relu(torch.abs(x)-th)
        #data_path=os.path.join(cwd,'Data/PNAS-regime_all_seeds/PNAS-regime_seed0')
        #data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
        #data_path=os.path.join(cwd,'Data/PNAS-highcoh_regime_all_seeds/PNAS-highcoh_regime_seed0')

        medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        print(medium.shape)
        inners=medium.transpose().conjugate()@medium
        print(inners.shape)
        coherence=0
        for i in inners:
            for j in i:
                if abs(j)>coherence and j<.99:
                    coherence=abs(j)
        print('coherence of data: ', coherence)


        Track_run=True
        #Enter your wanbd key below and uncomment wanbd code to track run on wandb
        if Track_run:
            key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa' #enter your own key here
            wandb.login(key=key) 
            wandb.init(project='rtt unlabeled', name=f'rtt normal grid')
            wandb.config['L1 final weight']=L1_weight_final
            wandb.config['unlabeled_data']=unlabeled_data 
            wandb.config['L1_weight']=L1_weight
            wandb.config['layers']=layers
            wandb.config['GELMA']=GELMA
            wandb.config['G_0']=G_0
            wandb.config['GELMA_layers']=GELMA_layers   
            wandb.config['inv_weight']=INV_weight
            wandb.config['CE']=CE
            wandb.config['timestamp']=timestampStr
        wand_dict={}
        #GELMA_layers.append(500)
        #layers.append(500)
        training_data=H.data_rho_loaded(data_path+'/train',unlabeled_data/80000,pixels='Gaussian')
        medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        sigmoid = nn.Sigmoid()
        softmax=nn.Softmax(dim=0)
        #training_data.b=training_data.b.to(device)
        #training_data.rho=training_data.rho.to(device)
        #rh_t, _=torch.split(rho, 400, dim=-1)


        val_data=H.data_rho_loaded(data_path+'/val', 3000/80000, pixels='Gaussian')
        b_val=val_data.b.to(device)
        rho_val=val_data.rho.to(device)
        xp=int((rho_val.shape[-1]/2)**(1/2))
        yp=xp
        indim=int(training_data.b[0].shape[0]/2)
        outdim=(training_data.rho[0].shape[0]/4)
        enc_dim=training_data.b[0].shape[0]/2
        if KM_in:
            enc_dim=enc_dim+outdim*2
        decoder_hats=[]
        if DICTIONARY_size=='default':
            encoder=M.fc_net_extra(enc_dim, layers,outdim, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None,dropout=.5)
        #encoder=M.channeled_lin_layers(enc_dim, layers, outdim*2, dropout=.5)
        
        #decoder=nn.Linear(int(training_data.rho[0].shape[0]/2), int(training_data.b[0].shape[0]), bias=False)  #R^ {N_k} -> C^{N_rec*N_freq}
            decoder=M.norm_linear(int(training_data.rho[0].shape[0]/2), int(training_data.b[0].shape[0]), normalize=False)  #R^ {N_k} -> C^{N_rec*N_freq}
        #decoder=M.fc_net_extra(outdim, layers[::-1], indim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None,dropout=.5)
        else:
            encoder=M.fc_net_extra(enc_dim, layers,DICTIONARY_size/2, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None,dropout=.5)
            decoder=M.norm_linear(DICTIONARY_size, int(training_data.b[0].shape[0]), normalize=False)  #R^ {N_k} -> C^{N_rec*N_freq}
        print(H.count_parameters(encoder))


        if G_0:
            G_0_w=(np.array(mat73.loadmat(data_path+'/G_0.mat')['A0']))
            
            #G_0_w=(np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt']))
            
            G_0_w=torch.cat((torch.tensor(G_0_w.real), torch.tensor(G_0_w.imag)), dim=0)
            G_0_w=G_0_w.float()
            decoder.weight.data=nn.parameter.Parameter(G_0_w.clone().detach().requires_grad_(True))
            Complex_eye=torch.eye(int(outdim*2)).unsqueeze(1)
            medium_hat=decoder(Complex_eye).squeeze()
            medium_hat=F.normalize(medium_hat, dim=-1)
            print(f'G_0 initialization successful: {torch.allclose(medium_hat, G_0_w.T)}')


        print(H.count_parameters(encoder))
        if Track_run:
            wandb.config['encoder params']=H.count_parameters(encoder)
        print('Data shapes:', training_data.b[0].shape[0]/2, training_data.rho[0].shape[0]/4)
        encoder.to(device)
        decoder.to(device)
        relu=nn.ReLU()
        leaky_relu=nn.LeakyReLU(THRESH_HOLD_VAL)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        optimizer_decod = torch.optim.AdamW(decoder.parameters(), lr=LR)


        tanh=nn.Tanh()
        softmax=nn.Softmax(dim=1)
        bce_loss_fn=nn.BCELoss() 
        Threshold=nn.Threshold(THRESH_HOLD_VAL, 0)  
        L2_loss_fn=nn.MSELoss()
        CE_loss_fn=nn.CrossEntropyLoss()
        #L2_loss_fn=lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
        L1_loss_fn=nn.L1Loss()
        if DICTIONARY_size=='default':
            Complex_eye=torch.eye(int(outdim*2)).unsqueeze(1).to(device)
        else:
            Complex_eye=torch.eye(DICTIONARY_size).unsqueeze(1).to(device)

        softmax=nn.Softmax(dim=1)
        if GELMA>0:
            GELMA_net=M.fc_net_extra(training_data.b[0].shape[0]/2, GELMA_layers, training_data.b[0].shape[0]/2, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None, dropout=.5)
            optimizer_GELMA = torch.optim.AdamW(GELMA_net.parameters(), lr=LR, maximize=True)
            lr_scheduler_GELMA=torch.optim.lr_scheduler.ExponentialLR(optimizer_GELMA, gamma=(LR_final/LR)**(1/Epochs), last_epoch=-1)
            GELMA_net.to(device)
            #GELMA_net=nn.DataParallel(GELMA_net)
            GELMA_net.train()
            #GELMA_net.set_params_to_zero()
            #encoder.set_params_to_zero()
        def f_col(batch):
            b=torch.stack([item[0] for item in batch])
            rho=torch.stack([item[1] for item in batch])
            num_targets=torch.stack([item[2] for item in batch])
            return b.to(device), rho.to(device), num_targets


        def rescale_rho(z):
            minimum, _=torch.min(z, -1, keepdim=True)
            maxium, _=torch.max(z, -1,   keepdim=True)
            z=(z-minimum)/(maxium-minimum)
            return z
        lr_scheduler_enc=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_final/LR)**(1/Epochs), last_epoch=-1)
        lr_scheduler_decod=torch.optim.lr_scheduler.ExponentialLR(optimizer_decod, gamma=(LR_final/LR)**(1/Epochs), last_epoch=-1)
        trainloader=DataLoader(training_data,batch_size=batchsize,shuffle=True,num_workers=0)

        for epoch in range(Epochs):
            if RESETS>0 and epoch%RESETS==0:
                encoder=M.fc_net_extra(enc_dim, layers,outdim, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None,dropout=.5)
                encoder.to(device)
                optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
                optimizer_decod = torch.optim.AdamW(decoder.parameters(), lr=LR)
            train_lossavg=0
            train_lossavg=0
            L1_loss_avg=0
            coupled_grad=0
            project_grad=0
            INV_loss_avg=0
            CEloss_avg=0


            inner_loss_term_avg=0
            for batch, (b, rho, num_targets) in enumerate(trainloader):
                if True:
                    b=b.to(device)
                    rho=rho.to(device)
                    optimizer.zero_grad()
                    optimizer_decod.zero_grad()
                    rho_hat=encoder(b)
                    rho_hat=soft_threh(rho_hat, THRESH_HOLD_VAL)
                    #rho_hat=relu(abs(rho_hat)-.005)*torch.sign(rho_hat)            
                    #rho_hat=softmax(rho_hat)
                    #max_rho, _=torch.max(abs(rho_hat), dim=-1, keepdim=True)
                    #rho_hat=abs(rho_hat)/max_rho
                    #rho_hat=sigmoid(rho_hat)
                    b_hat=decoder(rho_hat)
                    if L1_weight>0:
                        L1_loss=L1_weight*L1_loss_fn(rho_hat, rho_hat*0)
                        L1_loss.backward(retain_graph=True)  
                        L1_loss_avg+=L1_loss.item()/L1_weight/len(trainloader)
                    else:
                        L1_loss=-1
                    L2_loss=L2_loss_fn(b_hat.squeeze(), b.squeeze())
        #            L2_loss=L2_loss_fn(F.normalize(b_hat.squeeze(), dim=-1), F.normalize(b.squeeze()))
                    L2_loss.backward(retain_graph=True)
                    train_lossavg+=L2_loss.item()/len(trainloader)

                    #if GELMA==0:
                    
                    if GELMA==0:
                        optimizer.step()
                        optimizer_decod.step()
                        inner_loss_term_avg=-1


                    elif GELMA>0 and epoch>0:
                        #optimizer_GELMA.zero_grad()
                        #optimizer.zero_grad()
                        #optimizer_decod.zero_grad()
                        
                        GELMA_out=GELMA_net(b).squeeze()
                        GELMA_out=soft_threh(GELMA_out, THRESH_HOLD_VAL)
                        #GELMA_out=F.normalize(GELMA_out, dim=-1).squeeze()
                        b_hat=b_hat.squeeze()
                        GELMA_inners_coupled=(torch.inner((-b_hat).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        GELMA_inners_coupled=sum(GELMA_inners_coupled)/(torch.numel(GELMA_inners_coupled))/b.shape[-1]
                        GELMA_inners_project=(torch.inner((b).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        GELMA_inners_project=sum(GELMA_inners_project)/(torch.numel(GELMA_inners_project))/b.shape[-1]
                        GELMA_inners=GELMA*(GELMA_inners_project+GELMA_inners_coupled)
                        #inner_loss_term_avg+=(GELMA_inners).item()/len(trainloader)/GELMA

                        GELMA_inners.backward(retain_graph=True)
                
                        optimizer.step()
                        optimizer_decod.step()
                        #optimizer_GELMA.zero_grad()
                        #rho_hat=encoder(b)
        #                rho_hat=relu(abs(rho_hat)-.005)*torch.sign(rho_hat)            
                        #rho_hat=soft_threh(rho_hat, THRESH_HOLD_VAL)

                        #b_hat=decoder(rho_hat).squeeze()
                        #GELMA_out=GELMA_net(b).squeeze()
                        #GELMA_out=soft_threh(GELMA_out, THRESH_HOLD_VAL)

                        #GELMA_inners_coupled=(torch.inner((-b_hat).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        #GELMA_inners_coupled=sum(GELMA_inners_coupled)/(torch.numel(GELMA_inners_coupled))/b.shape[-1]
                        #GELMA_inners_project=(torch.inner((b).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        #GELMA_inners_project=sum(GELMA_inners_project)/(torch.numel(GELMA_inners_project))/b.shape[-1]
                        #GELMA_inners=(GELMA_inners_project+GELMA_inners_coupled)
                        #GELMA_inners.backward()
                        inner_loss_term_avg+=(GELMA_inners).item()/len(trainloader)/GELMA
        
                        

                        optimizer_GELMA.step()


                        optimizer_GELMA.zero_grad()
                        optimizer.zero_grad()
                        optimizer_decod.zero_grad()

                        #optimizer_GELMA.step()
                        #optimizer_GELMA.zero_grad()
                        #GELMA_inners_coupled=(torch.inner((-b_hat).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        #GELMA_inners_coupled=sum(GELMA_inners_coupled)/(torch.numel(GELMA_inners_coupled))
                        #GELMA_inners_project=(torch.inner((b).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1))
                        #GELMA_inners_project=sum(GELMA_inners_project)/(torch.numel(GELMA_inners_project))
                        #GELMA_inners=(GELMA_inners_project+GELMA_inners_coupled)

                        #GELMA_inners.backward()


                        #optimizer.step()
                        #optimizer_decod.step()

            
            L1_weight=min(L1_weight*(L1_weight_final/L1_weight_og)**(1/(END_schedule)), L1_weight_final)
            if GELMA>0:
                GELMA=min(GELMA*((GELMA_Final/GELMA_OG)**(1/(END_schedule))), GELMA_Final)
            if Scheduling>=0 and Epochs<END_schedule:
                lr_scheduler_enc.step()
                lr_scheduler_decod.step()
                if GELMA>0:
                    lr_scheduler_GELMA.step()
            val_lossavg=0
            #if epoch%50==49:
            #    H.plot_2_imgs(rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=20, ypix=20, font_size=25)
            if DICTIONARY_size=='default':
                wand_dict['Training image']=H.plot_2_unordered_imgs(epoch,rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=xp, ypix2=yp, font_size=25)
            else:
                wand_dict['Training image']=H.plot_2_unordered_imgs(epoch,rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=int(DICTIONARY_size**(1/2)), ypix2=int(DICTIONARY_size**(1/2)), font_size=25)

            encoder.eval()
            decoder.eval()
            #if GELMA_inc>0 and GELMA<GELMA_Final:
            #    GELMA=GELMA*GELMA_inc
            #    if GELMA>GELMA_Final:
            #        GELMA=GELMA_Final
            #        print('GELMA max reached')

            if INV_weight>0 and INV_weight<1e-1:
                INV_weight=INV_weight*(1.03)


            optimizer.zero_grad()
            optimizer_decod.zero_grad()

            font_size = 50
            if KM_in:
                medium_hat=decoder(Complex_eye).squeeze()
                medium_hat=F.normalize(medium_hat, dim=-1)
                Mhat_real, Mhat_imag=torch.split(medium_hat, int(medium_hat.shape[1]/2), dim=-1)
                Mhat_imag=-Mhat_imag
                b_real, b_imag=torch.split(b_val.squeeze(), int(b_val.shape[-1]/2), dim=-1)
                km_b_real=Mhat_real.squeeze()@b_real.squeeze().T-Mhat_imag.squeeze()@b_imag.squeeze().T
                km_b_imag=Mhat_real.squeeze()@b_imag.squeeze().T+Mhat_imag.squeeze()@b_real.squeeze().T

                km_b=torch.cat((km_b_real.T, km_b_imag.T), dim=-1)
                
                In_put=torch.cat((b_val, km_b), dim=-1)

                rho_hat=encoder(In_put)
            else:
                rho_hat=encoder(b_val)
                rho_hat=soft_threh(rho_hat, THRESH_HOLD_VAL)

                #rhos_hat=relu(abs(rho_hat)-.005)*torch.sign(rho_hat)            

            #rho_hat=sigmoid(rho_hat)
            #rho_hat=relu(rho_hat)
            #rho_hat=leaky_relu(rho_hat)
            #rho_hat=softmax(rho_hat)

            #max_rho, _=torch.max(rho_hat, dim=-1, keepdim=True)
            #rho_hat=rho_hat/max_rho
            #rho_hat=rescale_rho(rho_hat)
            #rho_hat=Threshold(abs(rho_hat))
            #max_rho, _=torch.max(rho_hat, dim=-1, keepdim=True) f
            #rho_hat=rho_hat/max_rho
            #rho_hat=tanh(rho_hat)

            b_hat=decoder(rho_hat) 
        #  L2_loss=L2_loss_fn(F.normalize(b_hat.squeeze(), dim=-1), F.normalize(b_val.squeeze()))

            L2_loss=L2_loss_fn(b_hat.squeeze(), b_val.squeeze())
            val_lossavg+=L2_loss.item()








            #if epoch%50==49:
            #    H.plot_2_imgs(rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=20, ypix=20, font_size=25)
            if DICTIONARY_size=='default':
                wand_dict['Validation image']=H.plot_2_unordered_imgs(epoch, rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=xp, ypix2=yp, font_size=25)
            else:
                wand_dict['Validation image']=H.plot_2_unordered_imgs(epoch, rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=int(DICTIONARY_size**(1/2)), ypix2=int(DICTIONARY_size**(1/2)), font_size=25)
            if epoch%1==0:
                sum_max_inner_original=0
                medium_hat=decoder(Complex_eye).squeeze()
                medium_hat=F.normalize(medium_hat, dim=-1)
                medium_hat=medium_hat.cpu().detach().numpy()
                medium_hat=H.cat2complex(medium_hat)                    
                torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()) ) #entry i j is ghat_i dot bar(g_j) 
                        #original way of computing inners
                        # mean over true max over hats
                index_list=[]
                for i in range(int(outdim*2)):
                    if max(torch_inners[:, i])>.99 and i not in index_list:
                        index_list.append(i)
                    sum_max_inner_original=sum_max_inner_original+max(torch_inners[:, i])
                max_avg_inners_original=sum_max_inner_original/(int(outdim*2))

                index_list_outs=[]
                for i in range(int(outdim*2)):
                    if max(torch_inners[:, i])<.95 and i not in index_list_outs:
                        index_list_outs.append(i)
                    

                #lr_scheduler_enc.step()
                #lr_scheduler_decod.step()
                #sum_max_inner_original=0      
                        #original way of computing inners
                        # mean over true max over hats
                maxG_inner_list=[max(torch_inners[:, i]) for i in range(int(outdim*2))]
                

                wand_dict['histogram']=H.hist_2_wandb(maxG_inner_list, figsize=5, font_size=25, epoch=epoch)
                wand_dict['Loss sum']=train_lossavg+L1_loss_avg*L1_weight+inner_loss_term_avg*GELMA
                
    




                
                #print(f'epoch: {epoch}, train loss: {train_lossavg}, L1 loss: {L1_loss_avg}, val loss: {val_lossavg}, max avg inners: {max_avg_inners_original}, GELMA loss: {inner_loss_term_avg}, num indices: {len(index_list)}')
                #print(f'coupled grad: {coupled_grad}, project grad: {project_grad}')
                wand_dict['train loss']=train_lossavg
                wand_dict['val loss']=val_lossavg
                if L1_loss>0:
                    wand_dict['L1 loss']=L1_loss_avg
                if GELMA>0:
                    wand_dict['GELMA loss']=inner_loss_term_avg
                if CE>0:
                    wand_dict['CE loss']=CEloss_avg
                if INV_weight>0:
                    wand_dict['INV loss']=INV_loss_avg

            
                wand_dict['max avg inners']=max_avg_inners_original
                wand_dict['num indices']=len(index_list)
                wand_dict['num indices <.95']=len(index_list_outs)
            if Track_run:
                wandb.log(wand_dict)

        print(time.time()-starttime)
        if Track_run:
            wandb.finish()
        index_list_list.append(index_list)
        medium_hat=decoder(Complex_eye).squeeze()
        medium_hat=F.normalize(medium_hat, dim=-1)
        medium_hat=medium_hat.cpu().detach().numpy()
        medium_hat=H.cat2complex(medium_hat)                    
        column_list.append(medium_hat)
        torch.save(encoder.state_dict(), f'/home/achristie/Codes_data/Experiment_data/rhosupport_stats/Classic_gelma/enocder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
        torch.save(decoder.state_dict(), f'/home/achristie/Codes_data/Experiment_data/rhosupport_stats/Classic_gelma/decoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
        torch.save(GELMA_net.state_dict(), f'/home/achristie/Codes_data/Experiment_data/rhosupport_stats/Classic_gelma/GELMA_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
        
    #np.save(f'/home/achristie/Codes_data/Experiment_data/rhosupport_stats/20kdata_50decoders.npy', column_list)








