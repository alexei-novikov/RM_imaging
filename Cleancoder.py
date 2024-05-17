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
import mat73
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
import Models as M
import Helpers as H



                        
        
    

def encoder_decoder(config=None):
    with wandb.init(config=config):#, project=config['project'], name = config['run_name']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = wandb.config

        args=config
        in_dim=121
        target='raw'
        scaler=None
        rand_perm=None
        cwd=os. getcwd()
        #data_type=p1_1real_6e-1_5sparse_125e-1X1e-0Ydiscrization_all_seeds
        data_path=os.path.join(cwd,f'{args.data_type}/{args.data_type[:-10]}_seed{args.seed}')
        args.data_path=data_path
        avg_max_inners_list=[] #local save of max inners
        increasing=True #flag for L1 cyclic rescaling
        if 'hom' in args.data_type or 'MDS' in args.data_type:
            s=1
        else:
            s=4
        print('Sparsity: ', s)
        training_data=H.data_rho_loaded(data_path+'/train',max(args.labeled_data/80000,.0001), sparsity=s)
        training_data_unlab=H.data_rho_loaded(data_path+'/train',max(args.unlabeled_data/80000,.0001),sparsity=s)
        val_data=H.data_rho_loaded(data_path+'/val', 3000/8000,sparsity=s)
        
        in_dim=len(training_data[0][0].squeeze())  #N_rec*N_freq
        out_dim=len(training_data[0][1].squeeze())/2 #(N_K)/2
        in_dim=in_dim/2
        out_dim = out_dim/2


        if args.out_encoder=='sigmoid' and args.Final_batch or 'MDS' in args.data_type:
            encoder=M.fc_net_extra(in_dim, args.hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=args.dropout)
        else:
            encoder=M.fc_net_batch(in_dim, args.hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=args.dropout)
        if 'complex' in args.data_type:
            decoder=M.complex_linear_layer(651, 121, bias=False, activation='Linear_layer')
        elif args.lin_type_decoder=='complex':
            #encoder=M.fc_net_extra(in_dim, args.hidden_dims, out_dim*2, net_type='fc',linear_type='complex', activation='mod_relu', bias=True, out_scaling=None, dropout=args.dropout) # C^{N_rec*N_freq} -> C^ {N_k} 
            decoder=M.complex_linear_layer(int(int(out_dim*2)), in_dim, bias=False, activation='Linear_layer') #C^ {N_k} -> C^{N_rec*N_freq}
            encoder=M.fc_net_extra(int(in_dim), args.hidden_dims, int(out_dim*2), net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=args.dropout)    # C^{N_rec*N_freq} -> R^ {N_k}
        else:
            encoder=M.fc_net_extra(in_dim, args.hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=args.dropout)    # C^{N_rec*N_freq} -> R^ {N_k}
            decoder=nn.Linear(int(int(out_dim*2)), int(in_dim*2), bias=False)  #R^ {N_k} -> C^{N_rec*N_freq}
        if args.G_0_intiailization==True:
            G_0=(np.array(mat73.loadmat(data_path+'/G_0.mat')['A0']))

            if args.lin_type_decoder=='complex':
                G_0_w_real=torch.tensor(G_0.real).float()
                G_0_w_imag=torch.tensor(G_0.imag).float()
                decoder.real_layer[0].weight.data=nn.parameter.Parameter(G_0_w_real.clone().detach().requires_grad_(True))
                decoder.imag_layer[0].weight.data=nn.parameter.Parameter(G_0_w_imag.clone().detach().requires_grad_(True))    
            else:
                G_0_w=torch.cat((torch.tensor(G_0.real), torch.tensor(G_0.imag)), dim=0)
                G_0_w=G_0_w.float()
                decoder.weight.data=nn.parameter.Parameter(G_0_w.clone().detach().requires_grad_(True))
        if args.GELMA>0:
            GELMA_net=M.fc_net_batch(in_dim, args.hidden_dims, in_dim, net_type='fc',linear_type='real', activation='relu', bias=True, out_scaling=None, dropout=args.dropout)
            optimizer_GELMA = torch.optim.AdamW(GELMA_net.parameters(), lr=.001, maximize=True)
            GELMA_net.to(device)
            GELMA_net=nn.DataParallel(GELMA_net)
            GELMA_net.train()


        try:
            medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        except:
            medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['A0'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        target_1_out=30
        target_2_out=31
        train_image_ind=5
        if args.G_0_intiailization=='G_0prod':
            G_0=(np.array(mat73.loadmat(data_path+'/G_0.mat')['A0']))
            G_0_w=torch.cat((torch.tensor(G_0.real), torch.tensor(G_0.imag)), dim=0)
            G_0_w=G_0_w.float()
            G_0_w=G_0_w.to(device)
        
        torch.manual_seed(1)
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)") 
        torch.manual_seed(args.seed)
 
        L1_updates=0

        sigmoid = nn.Sigmoid()
        softmax=nn.Softmax(dim=1)
        
        #10kdata\approx 2.26GB

        Loading_batch_size=args.batch_size
        trainloader=DataLoader(training_data,batch_size=Loading_batch_size,shuffle=True,num_workers=16, pin_memory=True)
        if 'MDS' not in args.data_type:
            training_data.Check_data(medium)
            training_data_unlab.Check_data(medium)
            val_data.Check_data(medium)
        
        trainloader_unlab=DataLoader(training_data_unlab,batch_size=Loading_batch_size,shuffle=True,num_workers=16, pin_memory=True)
        valloader=DataLoader(val_data,batch_size=len(val_data),shuffle=False,num_workers=1, pin_memory=True)
        
        dummy=nn.Linear(242, 651,bias=False)
        dummy.to(device)
        
        encoder=nn.DataParallel(encoder)
        rho_support=[31*1+1,31+29, 31*10+15, 31*20+1, 31*20+29]


        decoder=nn.DataParallel(decoder)
        encoder.to(device)
        decoder.to(device)
        L1_scaling=1
        if args.L1_rescaling=='Cyclic':
            L1_scaling_burn=0
        else:
            L1_scaling_burn=args.L1_burn_time          
        
        relu=nn.ReLU()
        if args.optimizer=='Adam':
            optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=.001)
            optimizer_dec= torch.optim.Adam(decoder.parameters(),  lr=.001)
        elif args.optimizer=='AdamW':
            optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=.001, weight_decay=args.weight_decay)
            optimizer_dec= torch.optim.AdamW(decoder.parameters(),  lr=.001, weight_decay=args.weight_decay)
        else :
            optimizer_enc = torch.optim.SGD(encoder.parameters(), lr=.001)
            optimizer_dec= torch.optim.SGD(decoder.parameters(),  lr=.001)
        if args.sch=='StepLR':
            scheduler_enc=torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=100,gamma=0.1, verbose=True)
        elif args.sch=='CyclicLr':
            scheduler_enc=torch.optim.lr_scheduler.CyclicLR(optimizer_enc, base_lr=1e-6, max_lr=0.1)
        elif args.sch=='ExponentialLR':
            scheduler_enc=torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, gamma=0.99)
        BCE_loss_f=nn.BCELoss()   
        CE_loss_f=nn.CrossEntropyLoss()
        if args.root_MSE:
            l2_loss= lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
        else:
            l2_loss=nn.MSELoss()
        l1_loss_f=nn.L1Loss()
        print(encoder)
        print(decoder)
        if 'PNAS' in args.data_type:
            xpix=20
            ypix=20
        else:
            xpix=31
            ypix=21

        fake_code_optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=0.001)
        fake_code_optimizer_dec= torch.optim.Adam(decoder.parameters(), lr=0.001)
        args.run_timestamp=timestampStr 
        args.encoder_parmams=H.count_parameters(encoder)
        args.decoder_params=H.count_parameters(decoder)
        max_innnn=0

        inners=medium.transpose().conjugate()@medium
        for i in inners:
            for j in i:
                if abs(j)>max_innnn and abs(j)<.99:
                    max_innnn=abs(j)
        args.coherence=max_innnn  
    




        for epoch in range(args.num_epochs):
            inner_loss_term_avg=0
            training_accuracy=0
            validation_accuracy=0
            unlab_val_lossavg=0
            CE_train_lossavg=0
            CE_val_loss_avg=0
            train_loss=0
            l1_unlab_loss_avg=0
            lab_train_lossavg=0
            L1_grad=0
            L2_grad=0
            wand_dict={}
            encoder.train()
            decoder.train()
            unlab_val_lossavg=0
            if (args.labeled_data>0):
                if epoch==0:
                    print("YOOOOOO U R USING LABELED DATA")
                if  (args.labeled_data==1 and epoch%50==0):
                    for dummy in range(math.ceil(args.unlabeled_data/128)):
                        
                        for batch, (b,rho,num_targets) in enumerate(trainloader):
                        
                            fake_code_optimizer_dec.zero_grad()
                            fake_code_optimizer_enc.zero_grad()
                            b=b.to(device)
                            rho=rho.to(device)    

                            b=b.squeeze().unsqueeze(0).repeat(8,1).unsqueeze(1)  
                            rho=rho.squeeze().unsqueeze(0).repeat(8,1).unsqueeze(1)
                            rh, _=torch.split(rho, 651, dim=-1)

                            rho_hat=encoder(b)
                            if args.out_encoder=='sigmoid':
                                rho_hat=sigmoid(rho_hat)
                            bhat=decoder(rh).squeeze()
                            bce_loss=BCE_loss_f(rho_hat.squeeze(), rh.squeeze())
                            bce_loss.backward()
                            
                            MSE_loss=l2_loss(bhat, b.squeeze())
                            MSE_loss.backward()
                            fake_code_optimizer_enc.step()
                            fake_code_optimizer_dec.step() 
                    
                
                elif 'MDS' in args.data_type:
                    for batch, (b,rho,num_targets) in enumerate(trainloader):
                        optimizer_enc.zero_grad()
                        #optimizer_dec.zero_grad()
                        b=b.to(device)
                        rho=rho.to(device)    
                        rh, _=torch.split(rho, int(rho.shape[-1]/2), dim=-1)
                        rho_hat=encoder(b)
                        ce_loss=CE_loss_f(rho_hat.squeeze(), rh.squeeze())
                        ce_loss.backward() 
                        optimizer_enc.step()
                        CE_train_lossavg+=ce_loss.item()
                        training_accuracy+=H.accuracy(torch.round(softmax(rho_hat.squeeze())), rh.squeeze()).item()/len(trainloader)
                        if batch==1:
                            rho_hat=softmax(rho_hat.squeeze())
                            rho_hat=torch.cat((rho_hat, torch.zeros_like(rho_hat)), -1)                 
                            rho_for_training_output=torch.clone(rho)
                            rho_hat_for_training_output=torch.clone(rho_hat)
                #print(bce_lossavg)

                else:

                    for batch, (b,rho,num_targets) in enumerate(trainloader):
                        optimizer_enc.zero_grad()
                        optimizer_dec.zero_grad()
                        b=b.to(device)
                        rho=rho.to(device)    
                        font_size = 50
                        rho_hat=encoder(b)
                        if args.out_encoder=='sigmoid':
                            rho_hat=sigmoid(rho_hat)
                        #rho_hat=relu(rho_hat.squeeze())
                        #H.plot_2_imgs(rho, rho_hat,ind=5, figsize=5)    
                        bce_loss=BCE_loss_f(rho_hat.squeeze(), rh.squeeze())
                        bce_loss.backward() 
                        optimizer_dec.step()
                        optimizer_enc.step()
                        lab_train_lossavg+=bce_loss.item()
                    #print(bce_lossavg)
                #H.plot_2_imgs(rho, rho_hat,ind=5, figsize=5)

            if args.unlabeled_data>0:
                if args.Dict_first_epochs>0:

                    encoder_loss_avg_list=[]
                    decoder_loss_avg_list=[]

                    for dummy in range(args.Dict_first_epochs):                        
                        
                        for param in encoder.parameters():
                            param.requires_grad = True
                    
                        encoder_loss_avg=0
                        for batch, (b,rho,num_targets) in enumerate(trainloader_unlab):
                            b=b.to(device)
                            b=b.squeeze()
                            rho=rho.to(device)    
                            optimizer_enc.zero_grad()
                            optimizer_dec.zero_grad()
                            rho_hat=encoder(b)
                            if args.out_encoder=='sigmoid':
                                rho_hat=sigmoid(rho_hat)
                            rho_hat=rho_hat.squeeze()
                            if args.l1_weight>0:
                                l1_loss=l1_loss_f(rho_hat, rho_hat*0)*args.l1_weight/L1_scaling
                                if args.L1_rescaling=='Geom':
                                    l1_loss=l1_loss*((1.015)**epoch)
                                    l1_loss.backward(retain_graph=True)
                                    l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling/((1.015)**epoch)
                                elif L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:
                                    l1_loss.backward(retain_graph=True)
                                    l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling/args.Dict_first_epochs
                                    L1_grad=L1_grad+sum([torch.norm(p.grad) for p in encoder.parameters()])
                                    optimizer_enc.step()
                                    optimizer_enc.zero_grad()
                                else:
                                    L1_grad=0
                                    l1_loss.backward(retain_graph=True)
                                    l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling/args.Dict_first_epochs
                                    

                            b_hat=decoder(rho_hat)
                            b_hat=b_hat.squeeze()
                            MSE_loss=l2_loss(b_hat, b)
                            MSE_loss.backward()
#                            if args.l1_weight>0 and L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:
#                                L2_grad=L2_grad+sum([torch.norm(p.grad) for p in encoder.parameters()])
                            #bce_loss=loss(rho_hat.squeeze(), rh.squeeze())
                            #bce_loss.backward() 
                            optimizer_enc.step()
                            train_loss+=MSE_loss.item()/len(trainloader_unlab)/args.Dict_first_epochs
                            encoder_loss_avg+=MSE_loss.item()/len(trainloader_unlab)
                            if batch==0:
                                rho_hat=torch.cat((rho_hat, torch.zeros_like(rho_hat)), -1)                 
                                rho_for_training_output=torch.clone(rho)
                                rho_hat_for_training_output=torch.clone(rho_hat)
                        encoder_loss_avg_list.append(encoder_loss_avg)

                    

                        if L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:


                            L1_scaling=((L1_grad/args.l1_weight*L1_scaling)/L2_grad)
                            
                            L1_scaling_burn=0
                        else:
                            L1_scaling_burn+=1




                    for param in encoder.parameters():
                        param.requires_grad = False
                    
                    for dummy in range(args.Dict_first_epochs):
                        decoder_avg_loss=0
                        for batch, (b,rho,num_targets) in enumerate(trainloader_unlab):
                            optimizer_enc.zero_grad()
                            optimizer_dec.zero_grad()
                            b=b.to(device)
                            b=b.squeeze()
                            rho=rho.to(device)    
                            rho_hat=encoder(b)
                            if args.out_encoder=='sigmoid':
                                rho_hat=sigmoid(rho_hat)
                            rho_hat=rho_hat.squeeze()
                            b_hat=decoder(rho_hat)
                            b_hat=b_hat.squeeze()
                            MSE_loss=l2_loss(b_hat, b)
                            
                            MSE_loss.backward()
                            optimizer_dec.step()
                            decoder_avg_loss+=MSE_loss.item()/len(trainloader_unlab)
                        decoder_loss_avg_list.append(decoder_avg_loss)
                    for i, j in zip(encoder_loss_avg_list, decoder_loss_avg_list):
                        wandb.log({'encoder_avg_loss':i, 'decoder_loss_avg':j})



                else:
                    for batch, (b,rho,num_targets) in enumerate(trainloader_unlab):
                        b=b.to(device)
                        b=b.squeeze()
                        rho=rho.to(device)    
                    
                        if args.GELMA>0:
                            optimizer_GELMA.zero_grad()
                        optimizer_enc.zero_grad()
                        optimizer_dec.zero_grad()
                        b=b.to(device)
                        b=b.squeeze()
                        rho=rho.to(device)    
                        rho_hat=encoder(b)
                        if args.out_encoder=='sigmoid':
                            rho_hat=sigmoid(rho_hat)
                        rho_hat=rho_hat.squeeze()
                        if args.l1_weight>0:
                            l1_loss=l1_loss_f(rho_hat, rho_hat*0)*args.l1_weight/L1_scaling
                            if args.L1_rescaling=='Geom':
                                l1_loss=l1_loss*((1.015)**epoch)
                                l1_loss.backward(retain_graph=True)
                                l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling/((1.015)**epoch)
                            elif L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:
                                l1_loss.backward(retain_graph=True)
                                l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling
                                L1_grad=L1_grad+sum([torch.norm(p.grad) for p in encoder.parameters()])
                                optimizer_enc.step()
                                optimizer_enc.zero_grad()
                            else:
                                L1_grad=0
                                l1_loss.backward(retain_graph=True)
                                l1_unlab_loss_avg+=l1_loss.item()/len(trainloader_unlab)*L1_scaling
                                
                    


                        b_hat=decoder(rho_hat)
                        b_hat=b_hat.squeeze()
                        MSE_loss=l2_loss(b_hat, b)
                        if args.GELMA>0:
                            GELMA_out=GELMA_net(b)
                            GELMA_inners=torch.inner(GELMA_out.squeeze(), (b-b_hat).squeeze()).diagonal(dim1=-2, dim2=-1)
                            inner_loss_term=args.GELMA*sum((GELMA_inners))/len(GELMA_inners)
                            inner_loss_term.backward(retain_graph=True)
                            inner_loss_term_avg+=inner_loss_term.item()/len(trainloader_unlab)/args.GELMA
                            optimizer_GELMA.step()

                        MSE_loss.backward()
                        if args.l1_weight>0 and L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:
                            L2_grad=L2_grad+sum([torch.norm(p.grad) for p in encoder.parameters()])
                        optimizer_enc.step()
                        optimizer_dec.step()
                        train_loss+=MSE_loss.item()/len(trainloader_unlab)

                        if batch==1:
                            if  'complex' not in args.data_type and args.lin_type_decoder!='complex':

                                rho_hat=torch.cat((rho_hat, torch.zeros_like(rho_hat)), -1)                 
                            rho_for_training_output=torch.clone(rho)
                            rho_hat_for_training_output=torch.clone(rho_hat)
                    
                    

                if L1_scaling_burn>args.L1_burn_time and args.L1_rescaling==True:


                    L1_scaling=((L1_grad/args.l1_weight*L1_scaling)/L2_grad)
                    
                    L1_scaling_burn=0
                else:
                    L1_scaling_burn+=1



                if args.L1_rescaling=='Cyclic':
                    #rate=(args.Max_l1_weight/args.l1_weight)**(1/args.L1_burn_time)
                    if increasing==True:
                        L1_scaling=L1_scaling/((args.Max_l1_weight/args.l1_weight)**(1/args.L1_burn_time))
                    else:
                        L1_scaling=L1_scaling*((args.Max_l1_weight/args.l1_weight)**(1/args.L1_burn_time))
                    
                    L1_updates+=1
                    if L1_updates%args.L1_burn_time==0: 
                        increasing=not increasing

            if 'complex' in args.data_type or args.lin_type_decoder=='complex':
                Complex_eye=complex_eye=torch.cat((torch.eye(int(out_dim*2)), torch.zeros((int(out_dim*2),int(out_dim*2)))), dim=1).unsqueeze(1).to(device)
            else:
                Complex_eye=torch.eye(int(out_dim*2)).unsqueeze(1).to(device)

            medium_hat=decoder(Complex_eye).squeeze()
            medium_hat=F.normalize(medium_hat, dim=-1)
            medium_hat=medium_hat.cpu().detach().numpy()
            medium_hat=H.cat2complex(medium_hat)













                    
            torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()) ) #entry i j is ghat_i dot bar(g_j) 

            sum_max_inners=0
            sum_max_noreplace=0
            

            sum_max_inner_original=0
            sum_max_noreplace_original=0
            if 'diffref' not in args.data_type:


                #torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()) ) #entry i j is ghat_i dot bar(g_j) 
                #mean over hats, max over true 
                for i in range(int(out_dim*2)):
                    sum_max_inners=sum_max_inners+max(torch_inners[i, :])
                    index_of_max=np.argmax(torch_inners[i,:])
                    torch_inners[:,index_of_max]=0*torch_inners[:,index_of_max]
                max_avg_inners=sum_max_inners/(int(out_dim*2))
                torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()))
                for i in range(int(out_dim*2)):
                    sum_max_noreplace=sum_max_noreplace+max(torch_inners[i,:]) #Fix ghat_i, find max over g

                avg_max_inners=sum_max_noreplace/(int(out_dim*2))





                #original way of computing inners
                # mean over true max over hats
                for i in range(int(out_dim*2)):
                    sum_max_inner_original=sum_max_inner_original+max(torch_inners[:, i])
                    index_of_max=np.argmax(torch_inners[:,i])
                    torch_inners[:,index_of_max]=0*torch_inners[index_of_max,:]
                max_avg_inners_original=sum_max_inner_original/(int(out_dim*2))
                torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()))
                for i in range(int(out_dim*2)):
                    sum_max_noreplace_original=sum_max_noreplace_original+max(torch_inners[:,i]) #Fix g, find max over ghat

                avg_max_inners_original=sum_max_noreplace_original/(int(out_dim*2))







                
            else:
                for i in rho_support:
                    sum_max_inners=sum_max_inners+max(torch_inners[:,i])
                    index_of_max=np.argmax(torch_inners[:,i])
                    torch_inners[index_of_max, :]=0*torch_inners[index_of_max,:]
                max_avg_inners_original=sum_max_inners/5
                torch_inners=np.inner(medium_hat,medium.T.conj())


                for i in rho_support:
                    sum_max_noreplace=sum_max_noreplace+max(torch_inners[:,i]) #Fix g_i, find max over j

                avg_max_inners_original=sum_max_noreplace/5
            encoder.eval()
            decoder.eval()
            for batch, (b,rho,num_targets) in enumerate(valloader):
                b=b.to(device)
                rho=rho.to(device)    
                rho_hat=encoder(b)
                if 'MDS' in args.data_type:
                    rh, _=torch.split(rho, int(rho.shape[-1]/2), dim=-1)
                    rh=rh.squeeze() 
                    CE_val_loss=CE_loss_f(rho_hat.squeeze(), rh.squeeze())
                    CE_val_loss_avg+=CE_val_loss.item()
                elif args.out_encoder=='sigmoid':

                    rho_hat=sigmoid(rho_hat)

                b_hat=decoder(rho_hat)
                val_loss=l2_loss(b_hat.squeeze(), b.squeeze())
                
                unlab_val_lossavg+=val_loss.item()
                if batch==0:
                    if  'complex' not in args.data_type and args.lin_type_decoder!='complex':
                        if 'MDS' in args.data_type:
                            rho_hat=softmax(rho_hat.squeeze())
                            validation_accuracy+=H.accuracy(torch.round(rho_hat.squeeze()), rh.squeeze()).item()/len(valloader)

                        rho_hat=torch.cat((rho_hat, torch.zeros_like(rho_hat)), -1)     
                    rho_for_fixed_output=torch.clone(rho)
                    rho_hat_for_fixed_output=torch.clone(rho_hat)

            wand_dict['Train Loss']=train_loss/(Loading_batch_size/args.batch_size)
            wand_dict['val loss unlabeled full model']=unlab_val_lossavg 
            wand_dict['L1_weight']=args.l1_weight/L1_scaling

            if 'MDS' in args.data_type:
                wand_dict['CE val loss']=CE_val_loss_avg/len(valloader)
                wand_dict['CE train loss']=CE_train_lossavg/len(trainloader)/(Loading_batch_size/args.batch_size)
            elif args.labeled_data>0:
                wand_dict['labeled train loss']=lab_train_lossavg /(Loading_batch_size/args.batch_size)
            wand_dict['Train Loss']=train_loss
            if 'diffref' not in args.data_type:
                wand_dict['Mean_j Max_i |<hat g_j, g_i>| 1-1']=max_avg_inners
                wand_dict['Mean_j Max_i |<hat g_j, g_i>|']=avg_max_inners


            wand_dict['Mean_i Max_j |<hat g_j, g_i>|']=avg_max_inners_original 
            wand_dict['Mean_i Max_j |<hat g_j, g_i>|']=max_avg_inners_original


            if args.GELMA>0:
                wand_dict['GELMA term']=inner_loss_term_avg
            if 'MDS' in args.data_type:
                wand_dict['validation accuracy']=validation_accuracy
                wand_dict['training accuracy']=training_accuracy

            if args.l1_weight>0:
                wand_dict['l1 loss']=l1_unlab_loss_avg/args.l1_weight/(Loading_batch_size/args.batch_size)
                if args.sch:
                    scheduler_enc.step()
            wand_dict['One target'] = H.output2wandb(rho_for_fixed_output.squeeze(), rho_hat_for_fixed_output.squeeze(), target_1_out, epoch, xpix=xpix, ypix=ypix)
            wand_dict['Two targets'] = H.output2wandb(rho_for_fixed_output.squeeze(), rho_hat_for_fixed_output.squeeze(), target_2_out, epoch, xpix=xpix, ypix=ypix)
            wand_dict['training image'] = H.output2wandb(rho_for_training_output.squeeze(), rho_hat_for_training_output.squeeze(), train_image_ind, epoch, xpix=xpix, ypix=ypix)   
            #if epoch==args.num_epochs-1:
                #H.save_2_imgs(rho_for_fixed_output, rho_hat_for_fixed_output,ind=target_1_out,figsize=8,scaling='Linf',font_size=50,  xpix=20, ypix=20, file_name=f'{timestampStr}_validation')
                #H.save_2_imgs(rho_for_training_output, rho_hat_for_training_output,ind=train_image_ind,figsize=8,scaling='Linf',font_size=20,  xpix=20, ypix=20, file_name=f'{timestampStr}_training')

            wandb.log(wand_dict)
            if epoch==0:
                print('first epoch passed')
            try:
                os.mkdir(f'/home/achristie/Codes_data/Experiment_data/{args.Data_locat}')
            except:
                pass

            avg_max_inners_list.append(avg_max_inners_original)
            #save_dict={'ghats_max':max_avg_inners, 'Train_Loss':train_loss, 'ghats_no_repl':avg_max_inners, 'val_loss_unlabeled':unlab_val_lossavg}                
            #if args.l1_weight>0:
            #    save_dict['l1_loss']=l1_unlab_loss_avg/args.l1_weight
            args.save_lcation=f'/home/achristie/Codes_data/Experiment_data/{args.Data_locat}/{args.labeled_data}L_{args.unlabeled_data}U_{args.hidden_dims}_{timestampStr}'
            np.save(f'/home/achristie/Codes_data/Experiment_data/{args.Data_locat}/{args.net_type}_{args.labeled_data}L_{args.unlabeled_data}U_{args.hidden_dims}_{args.l1_weight}L1w_{timestampStr}', avg_max_inners_list)   
            torch.save(encoder.module.state_dict(),f'/home/achristie/Codes_data/Experiment_data/{args.Data_locat}/{args.labeled_data}L_{args.unlabeled_data}U_{args.hidden_dims}_{timestampStr}encoder.pt')
            torch.save(decoder.module.state_dict(),f'/home/achristie/Codes_data/Experiment_data/{args.Data_locat}/{args.labeled_data}L_{args.unlabeled_data}U_{args.hidden_dims}_{timestampStr}decoder.pt')
        