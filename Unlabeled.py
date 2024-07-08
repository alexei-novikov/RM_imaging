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
#%matplotlib inline

cwd=os. getcwd()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_path=os.path.join(cwd,'Data/PNAS-regime_all_seeds/PNAS-regime_seed0')
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
#b=np.load(data_path+'/train/b.npy')
#rho=np.load(data_path+'/train/rho.npy')
#print(np.allclose(medium@rho[0],b[0]))
index_list=[]
torch.__version__




layers_list=[[2**i]*10 for i in range(6, 10)]
unlabeled_data_list=[40000]

for layers in layers_list:
    for unlabeled_data in unlabeled_data_list:
        torch.cuda.empty_cache()
        starttime=time.time()
        target='raw'
        batchsize=128
        L1_weight=5e-10
        GELMA_layers=layers
        CE=False
        KM_in=False
        INV_weight=0
        RESHUFFLE=False
        LR=1e-4
        GELMA=1e-4
        TIME_reverse=False
        E_list=0

        G_0=True
        Epochs=3000
        index_list=[]


        Track_run=True
        #Enter your wanbd key below and uncomment wanbd code to track run on wandb

        #GELMA_layers.append(500)
        #layers.append(500)
        training_data=H.data_rho_loaded(data_path+'/train',unlabeled_data/80000, normalize=True)
        medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        sigmoid = nn.Sigmoid()
        softmax=nn.Softmax(dim=0)
        #training_data.b=training_data.b.to(device)
        #training_data.rho=training_data.rho.to(device)
        #rh_t, _=torch.split(rho, 400, dim=-1)


        val_data=H.data_rho_loaded(data_path+'/val', 3000/80000, normalize=True)
        b_val=val_data.b.to(device)
        rho_val=val_data.rho.to(device)
        rh_v, _=torch.split(rho_val, 400, dim=-1)
        indim=int(training_data.b[0].shape[0]/2)
        outdim=int(training_data.rho[0].shape[0]/4)
        enc_dim=training_data.b[0].shape[0]/2
        if KM_in:
            enc_dim=enc_dim+outdim*2


        encoder=M.fc_net_extra(enc_dim, layers,outdim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None,dropout=.5)
        
        if Track_run:
            key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa' #enter your own key here
            wandb.login(key=key) 
            wandb.init(project='rtt unlabeled', name=f'Unlabeled data: {unlabeled_data}, parameters: {H.count_parameters(encoder)}')
            wandb.config['unlabeled_data']=unlabeled_data 
            wandb.config['L1_weight']=L1_weight
            wandb.config['layers']=layers
            wandb.config['GELMA']=GELMA
            wandb.config['run']=CE
        wand_dict={}
        
        #decoder=nn.Linear(int(training_data.rho[0].shape[0]/2), int(training_data.b[0].shape[0]), bias=False)  #R^ {N_k} -> C^{N_rec*N_freq}
        decoder=M.norm_linear(int(training_data.rho[0].shape[0]/2), int(training_data.b[0].shape[0]))  #R^ {N_k} -> C^{N_rec*N_freq}
        #decoder=M.fc_net_extra(outdim, layers[::-1], indim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None,dropout=.5)
        if G_0:
            G_0=(np.array(mat73.loadmat(data_path+'/G_0.mat')['A0']))
            G_0_w=torch.cat((torch.tensor(G_0.real), torch.tensor(G_0.imag)), dim=0)
            G_0_w=G_0_w.float()
            decoder.weight.data=nn.parameter.Parameter(G_0_w.clone().detach().requires_grad_(True))


        print(H.count_parameters(encoder))
        print('Data shapes:', training_data.b[0].shape[0]/2, training_data.rho[0].shape[0]/4)
        encoder.to(device)
        decoder.to(device)

        relu=nn.ReLU()
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        optimizer_decod = torch.optim.AdamW(decoder.parameters(), lr=min(LR*100, 1e-3))


        softmax=nn.Softmax(dim=1)
        bce_loss_fn=nn.BCELoss()   
        L2_loss_fn=nn.MSELoss()
        CE_loss_fn=nn.CrossEntropyLoss()
        L2_loss_fn=lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
        L1_loss_fn=nn.L1Loss()
        Complex_eye=torch.eye(int(outdim*2)).unsqueeze(1).to(device)
        softmax=nn.Softmax(dim=1)
        if GELMA>0:
            GELMA_net=M.fc_net_batch(training_data.b[0].shape[0]/2, GELMA_layers, training_data.b[0].shape[0]/2, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=.5)
            optimizer_GELMA = torch.optim.Adam(GELMA_net.parameters(), lr=LR, maximize=True)
            GELMA_net.to(device)
            #GELMA_net=nn.DataParallel(GELMA_net)
            GELMA_net.train()
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
        lr_scheduler_enc=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.995, last_epoch=-1)
        lr_scheduler_decod=torch.optim.lr_scheduler.ExponentialLR(optimizer_decod, gamma=.995, last_epoch=-1)
        trainloader=DataLoader(training_data,batch_size=batchsize,shuffle=True,num_workers=0)
        if E_list>0:
            encs=[M.fc_net_extra(enc_dim, layers,outdim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None,dropout=.5) for i in range(E_list)]
            opts=[torch.optim.AdamW(enc.parameters(), lr=LR) for enc in encs]
            for enc in encs:
                enc.to(device)
        for epoch in range(Epochs):
            train_lossavg=0
            train_lossavg=0
            for batch, (b, rho, num_targets) in enumerate(trainloader):
                if True:
                    b=b.to(device)
                    rho=rho.to(device)
                    optimizer.zero_grad()
                    optimizer_decod.zero_grad()
                    if E_list>0:
                        for enc in encs:
                            enc.zero_grad()
                    if KM_in:
                        medium_hat=decoder(Complex_eye).squeeze()
                        medium_hat=F.normalize(medium_hat, dim=-1)
                        Mhat_real, Mhat_imag=torch.split(medium_hat, int(medium_hat.shape[1]/2), dim=-1)
                        Mhat_imag=-Mhat_imag
                        b_real, b_imag=torch.split(b.squeeze(), int(b.shape[-1]/2), dim=-1)
                        km_b_real=Mhat_real.squeeze()@b_real.squeeze().T-Mhat_imag.squeeze()@b_imag.squeeze().T
                        km_b_imag=Mhat_real.squeeze()@b_imag.squeeze().T+Mhat_imag.squeeze()@b_real.squeeze().T

                        km_b=torch.cat((km_b_real.T, km_b_imag.T), dim=-1)
                        In_put=torch.cat((b, km_b), dim=-1)

                        rho_hat=encoder(In_put)
                    elif TIME_reverse:
                        rho_hat=encoder(decoder.time_reverse(b.squeeze()))
                    else:
                        rho_hat=encoder(b)

                    if E_list>0:
                        rho_hats=[enc(b) for enc in encs]
                        rho_hats=[relu(rho_hat) for rho_hat in rho_hats]
                        for rho_hat_l in rho_hats:
                            max_rho_l, _=torch.max(rho_hat_l, dim=-1, keepdim=True)
                            rho_hat_l=rho_hat_l/max_rho_l

                        b_hats=[decoder(rho_hat_l) for rho_hat_l in rho_hats]
                        L2_loss=sum([L2_loss_fn(b_hat_l.squeeze(), b.squeeze()) for b_hat_l in b_hats])/len(b_hats)
                        L2_loss.backward()
                        for opt in opts:
                            opt.step()
                    #rho_hat=sigmoid(rho_hat)
                    #rho_hat=softmax(rho_hat)
                    rho_hat=relu(rho_hat)
                    max_rho, _=torch.max(rho_hat, dim=-1, keepdim=True)
                    rho_hat=rho_hat/max_rho
                    #rho_hat=rescale_rho(rho_hat)


                    b_hat=decoder(rho_hat)
                    if INV_weight>0:
                        medium_hat=decoder(Complex_eye).squeeze()
                        medium_hat=F.normalize(medium_hat, dim=-1)
                        b_real, b_imag=torch.split(b.squeeze(), int(b.shape[-1]/2), dim=-1)
                        Mhat_real, Mhat_imag=torch.split(medium_hat, int(medium_hat.shape[1]/2), dim=-1)
                        complex_mat=Mhat_real+1j*Mhat_imag
                        #print(f'og complex mat shape: {complex_mat.shape}')
                        complex_mat=complex_mat.T #yes
                        complex_mat=torch.linalg.pinv(complex_mat.T.conj()@complex_mat, hermitian=True)@complex_mat.T.conj()  
                        complex_b=b_real+1j*b_imag
                        complex_b=complex_b.T
                        #print(complex_mat.shape, complex_b.shape)

                        rho_inv=complex_mat@complex_b
                        rho_inv=rho_inv.T
                        rho_inv_real=rho_inv.real
                        rho_inv_imag=rho_inv.imag
                        INV_loss=L2_loss_fn(rho_inv_real, rho_hat)*INV_weight
                        INV_loss.backward(retain_graph=True)
                        INV_loss=INV_loss.item()/INV_weight
                        if INV_weight<5e-3:
                            INV_weight=INV_weight*1.011


                        
                        
                        




                    if L1_weight>0:
                        L1_loss=L1_weight*L1_loss_fn(rho_hat, rho_hat*0)
                        L1_loss.backward(retain_graph=True)  
                        L1_loss=L1_loss.item()/L1_weight
                    else:
                        L1_loss=-1

                    if CE>0:
                        rho_hat_hat=encoder(b_hat)

                        CEloss=CE_loss_fn(rho_hat_hat, rho_hat)*CE
                        CEloss.backward(retain_graph=True)
                        CEloss=CEloss.item()/CE

                    
                    if GELMA>0 and epoch>0:
                        optimizer_GELMA.zero_grad()
                        GELMA_out=GELMA_net(b).squeeze()
                        GELMA_out=F.normalize(GELMA_out, dim=-1).squeeze()
                        b_hat=b_hat.squeeze()
                        GELMA_inners=torch.inner(GELMA_out.squeeze(), (b-b_hat).squeeze()).diagonal(dim1=-2, dim2=-1)
                        inner_loss_term=GELMA*sum((GELMA_inners))/(torch.numel(GELMA_inners))
                        inner_loss_term.backward(retain_graph=True)
                        inner_loss_term_avg=inner_loss_term.item()/GELMA
                        optimizer_GELMA.step()
                    
                    else:
                        inner_loss_term_avg=-1
                    L2_loss=L2_loss_fn(b_hat.squeeze(), b.squeeze())
                    L2_loss.backward()


                    
                    optimizer.step()
                    optimizer_decod.step()
                    train_lossavg+=L2_loss.item()/len(trainloader)
            #if L1_loss>5e-5:
            #    L1_weight=L1_weight*(0.9)
            val_lossavg=0
            #if epoch%50==0:
            #H.plot_2_imgs(rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=20, ypix=20, font_size=25)
            encoder.eval()
            decoder.eval()




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
                
            # print(b.shape, km_b.shape,km_b_real.shape, Mhat_real.shape)
                In_put=torch.cat((b_val, km_b), dim=-1)

                rho_hat=encoder(In_put)
            else:
                rho_hat=encoder(b_val)
            #rho_hat=sigmoid(rho_hat)
            rho_hat=relu(rho_hat)
            max_rho, _=torch.max(rho_hat, dim=-1, keepdim=True)
            rho_hat=rho_hat/max_rho
            #rho_hat=softmax(rho_hat)
            #rho_hat=rescale_rho(rho_hat)

            b_hat=decoder(rho_hat) 
            L2_loss=L2_loss_fn(b_hat.squeeze(), b_val.squeeze())
            val_lossavg+=L2_loss.item()





        
        
        
            #if epoch%50==0:
            #H.plot_2_imgs(rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=20, ypix=20, font_size=25)
            
            if epoch%1==0:
                sum_max_inner_original=0
                medium_hat=decoder(Complex_eye).squeeze()
                medium_hat=F.normalize(medium_hat, dim=-1)
                medium_hat=medium_hat.cpu().detach().numpy()
                medium_hat=H.cat2complex(medium_hat)                    
                torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()) ) #entry i j is ghat_i dot bar(g_j) 
                        #original way of computing inners
                        # mean over true max over hats
                for i in range(int(outdim*2)):
                    if max(torch_inners[:, i])>.95 and i not in index_list:
                        index_list.append(i)
                    sum_max_inner_original=sum_max_inner_original+max(torch_inners[:, i])
                max_avg_inners_original=sum_max_inner_original/(int(outdim*2))
        
                
                #lr_scheduler_enc.step()
                #lr_scheduler_decod.step()
                
                
                #print(f'epoch: {epoch}, train loss: {train_lossavg}, L1 loss: {L1_loss}, val loss: {val_lossavg}, max avg inners: {max_avg_inners_original}, GELMA loss: {inner_loss_term_avg}, num indices: {len(index_list)}')
                wand_dict['train loss']=train_lossavg
                wand_dict['val loss']=val_lossavg
                if L1_loss>0:
                    wand_dict['L1 loss']=L1_loss
                if GELMA>0:
                    wand_dict['GELMA loss']=inner_loss_term_avg

            
                wand_dict['max avg inners']=max_avg_inners_original
            if Track_run:
                wandb.log(wand_dict)

        #index_list_list.append(index_list)
        print(time.time()-starttime)
        if Track_run:
            wandb.finish()
