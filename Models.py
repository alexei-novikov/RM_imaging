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
import Helpers as H



#Mode reul activation function.
#ReLu(b-input), b updated during grad descent
class mod_relu(torch.nn.Module):
    def __init__(self, in_features,rand_weights=True):
        super(mod_relu,self).__init__()
        if rand_weights:
            self.b = nn.Parameter((2*(torch.rand(in_features)-.5))/(in_features**(1/2)))
        else:
            self.b = nn.Parameter(0*(2*(torch.rand(in_features)-.5))/(in_features**(1/2)))            
        self.b.requiresGrad = True
        self.Relu=nn.ReLU()

    def forward(self, x):
        return self.Relu(torch.abs(x) + self.b)           

#Linear layer with normalized columns
class norm_linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(norm_linear, self).__init__()
        self.weight=nn.Parameter((torch.rand(out_dim, in_dim)-.5)/(in_dim**(1/2)))
    def forward(self,x):
        weights=F.normalize(self.weight,dim=0)
        out=torch.matmul(x,weights.t())
        out=out.squeeze()
        out=F.normalize(out,dim=-1)
        return out
    def time_reverse(self, x):
        weights=F.normalize(self.weight,dim=1)
        weights_real, weights_imag=torch.split(weights.t(), int(weights.t().shape[1]/2), dim=-1)
        weights_imag=-weights_imag
        x_real, x_imag=torch.split(x, int(x.shape[1]/2), dim=-1)
        out_real=torch.matmul(x_real,weights_real.t())-torch.matmul(x_imag,weights_imag.t())
        out_imag=torch.matmul(x_real,weights_imag.t())+torch.matmul(x_imag,weights_real.t())
        out=torch.cat((out_real,out_imag),-1)
        out=out.squeeze()
        return out
    
class norm_linear_complex(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(norm_linear_complex, self).__init__()
        self.weight=nn.Parameter((torch.rand(out_dim, in_dim)-.5)/(in_dim**(1/2)))
    
    def forward(self,x):
        x=x.squeeze()
        weights=F.normalize(self.weight,dim=0)
        weights=weights.t()
        weights_real, weights_imag=torch.split(weights, int(weights.shape[1]/2), dim=-1)
        x_real, x_imag=torch.split(x.squeeze(), int(x.shape[1]/2), dim=-1)
        out_real=torch.matmul(x_real,weights_real)-torch.matmul(x_imag,weights_imag)
        out_imag=torch.matmul(x_real,weights_imag)+torch.matmul(x_imag,weights_real)
        out=torch.cat((out_real,out_imag),-1)
        out=F.normalize(out,dim=-1)
        return out



#linear layer wrapper for same signatrue as complex
class linear_layer_wrapper(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation='relu',threshold_val=False, offset=False,dropout=0.0, batch_normalization=True):
        super(linear_layer_wrapper, self).__init__()
        in_dim=in_dim*2
        out_dim=out_dim*2
        if activation=='relu' and batch_normalization:
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias),nn.BatchNorm1d(int(out_dim)), nn.ReLU() ,nn.Dropout(dropout))
        elif activation=='leaky' and batch_normalization:
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias),nn.BatchNorm1d(int(out_dim)), nn.LeakyReLU() ,nn.Dropout(dropout))
        elif activation=='relu' and not batch_normalization:
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias), nn.ReLU() ,nn.Dropout(dropout))
        elif activation=='sigmoid' and batch_normalization:
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias),nn.BatchNorm1d(int(out_dim)), nn.Sigmoid() ,nn.Dropout(dropout))
        elif activation=='mod_relu' and batch_normalization:
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias),nn.BatchNorm1d(int(out_dim)), mod_relu(int(out_dim)) ,nn.Dropout(dropout))

        elif activation=='Linear_layer':
            self.layer=nn.Sequential(nn.Linear(int(in_dim),int(out_dim),bias=bias))                                                   
    def forward(self,x):
        out=self.layer(x)
        return out



#complex thresholding activation
class C_Threshold(nn.Module):
    def __init__(self,threshold_val, offset):
        super(C_Threshold, self).__init__()
        self.threshold_val=threshold_val
        self.offset=offset
        if offset:
            self.acti=nn.Threshold(threshold_val, -threshold_val)
        else:
            self.acti=nn.Threshold(threshold_val, 0)
        self.tan=torch.atan2
        
    def forward(self, modulus):
        modulus=self.acti(modulus)+self.threshold_val*self.offset
        return modulus


        
#complex lienar model class
class complex_linear_layer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation='relu', threshold_val=1e-3, offset=True,dropout=0.0, batch_normalization=True):
        super(complex_linear_layer, self).__init__()
        p=dropout
        self.in_dim=int(in_dim)
        out_dim=int(out_dim)
        self.dropout = nn.Dropout(p=p)
        if activation!='relu' and activation!='Linear_layer' and activation!='linear_threshold':
            self.activation_str='complex'
        else:
            self.activation_str=activation

        if activation=='relu':
            self.activation=nn.ReLU()
        if activation=='sigmoid':
            self.activation=nn.Sigmoid()
        if activation=='tanh':
            self.activation=nn.Tanh()
        if activation=='leaky':
            self.activation=nn.LeakyReLU()
        if activation=='hardshrink':
            self.activation=nn.Hardshrink(5e-4)
        if activation=='Complex threshhold':
            self.activation=C_Threshold(threshold_val,offset)
        if activation=='mod_relu':
            self.activation=mod_relu(out_dim)
        elif activation=='linear_threshold':
            self.activation=mod_relu(out_dim,rand_weights=False)            
        if activation=='Linear_layer' or activation=='linear_threshold':
            self.real_layer=nn.Sequential(nn.Linear(self.in_dim,out_dim,bias=bias))
            self.imag_layer=nn.Sequential(nn.Linear(self.in_dim,out_dim,bias=bias)) 
        else:
            self.real_layer=nn.Linear(self.in_dim,out_dim,bias=bias)
            self.real_batch=nn.BatchNorm1d(out_dim)
            self.imag_layer=nn.Linear(self.in_dim,out_dim,bias=bias)  
            self.imag_batch=nn.BatchNorm1d(out_dim)
    def forward(self,x):
        out=x
        real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)

        if self.activation_str=='Linear_layer' or self.activation_str=='linear_threshold':
            real_out=((self.real_layer(real)-self.imag_layer(imag)))
            imag_out=((self.imag_layer(real)+self.real_layer(imag)))
            if self.activation_str=='linear_threshold':
                real=real_out
                imag=imag_out
                modulus=torch.abs(real+1j*imag)
                theta=torch.atan2(imag, real)
                modulus=self.activation(modulus)
                real_out=modulus*torch.cos(theta)
                imag_out=modulus*torch.sin(theta) 
        elif self.activation_str=='relu':
            real_out=self.activation(self.real_batch(self.real_layer(real)-self.imag_layer(imag)))
            imag_out=self.activation(self.imag_batch(self.imag_layer(real)+self.real_layer(imag)))
        else:
            real_out=self.real_batch(self.real_layer(real)-self.imag_layer(imag))
            imag_out=self.imag_batch(self.imag_layer(real)+self.real_layer(imag))
            real=real_out
            imag=imag_out
            modulus=torch.abs(real+1j*imag)
            theta=torch.atan2(imag, real)
            modulus=self.activation(modulus)
            real_out=modulus*torch.cos(theta)
            imag_out=modulus*torch.sin(theta) 
            real_out=self.dropout(real_out)
            imag_out=self.dropout(imag_out)

                               
        return torch.cat((real_out,imag_out),-1) 

    
class variational_enc(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True,threshold_val=1e-3, offset=True,dropout=0, out_scaling='L2', batch_normalization=True):
        super(variational_enc,self).__init__()
        fc_net_extra
        self.shared=fc_net_batch(in_dim, [hidden_dims[0]], hidden_dims[1], net_type, linear_type, activation, bias, threshold_val, offset, dropout, out_scaling, batch_normalization)
        self.enc_mean=fc_net_batch(hidden_dims[1], hidden_dims[2:], out_dim, net_type, linear_type, activation, bias, threshold_val, offset, dropout, out_scaling, batch_normalization)
        self.enc_logvar=fc_net_batch(hidden_dims[1], hidden_dims[:2], out_dim, net_type, linear_type, activation, bias, threshold_val, offset, dropout, out_scaling, batch_normalization)
    def forward(self, x):
        shared=self.shared(x.squeeze())
        mean=self.enc_mean(shared.squeeze())
        logvar=self.enc_logvar(shared.squeeze())
        return mean, logvar



#Fc=fully connected. 
#Takes in complex in_dim, hidden_dims, and out_dim. Will return a fully connected network  with Relu activation, and batchnormalization
class fc_net_batch(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True,threshold_val=1e-3, offset=True,dropout=0, out_scaling='L2', batch_normalization=True):
        super(fc_net_batch, self).__init__()
        p=dropout
        self.dropout = nn.Dropout(p=p)
        self.activation=activation

        if linear_type=='complex':
            linear_layer=complex_linear_layer
        else:
            linear_layer=linear_layer_wrapper
        #Makes network linear if flagged
        if net_type=='linear' or hidden_dims[0]==0:
            self.net_type='linear'
            self.num_layers=1
            if activation=='linear_threshold':
                self.layers = nn.ModuleList([(linear_layer(in_dim,out_dim, bias=False,activation='linear_threshold'))])
                self.layers[0].activation
            else:
                self.layers = nn.ModuleList([(linear_layer(in_dim,out_dim, bias=False,activation='Linear_layer'))])
        else:
            self.net_type=net_type   
            self.num_layers=len(hidden_dims)+1
            self.layers = nn.ModuleList([linear_layer(in_dim, hidden_dims[0], bias=bias, activation=activation,threshold_val=threshold_val, offset=offset, dropout=dropout, batch_normalization=batch_normalization)])
            self.layers.extend([linear_layer(hidden_dims[i-1],hidden_dims[i],bias=bias, activation=activation,threshold_val=threshold_val, offset=offset,dropout=dropout, batch_normalization=batch_normalization) for i in range(1, self.num_layers-1)])   



        
            if net_type=='fc':
                self.layers.extend([linear_layer(hidden_dims[-1],out_dim, bias=True, activation='Linear_layer',batch_normalization=batch_normalization)])  
            # reurrent_out are two components. one that is a fully connected, one that is linear. rueccurent_out will return 
            #Out_scaling(FC(x)+Lin(x))
            elif self.net_type=='residual_out':
                self.layers.extend([linear_layer(hidden_dims[-1]+in_dim,out_dim, bias=False, activation='Linear_layer', batch_normalization=batch_normalization)])  
            

        if self.net_type=='linear' or self.net_type=='stacking'or out_scaling==None:
           self.out_scaling=None
        else:
            self.out_scaling=out_scaling
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(-1)



    def forward(self, x):
        out=x.squeeze()
        #iterates through layer list and evaluates each layer sequentially
        for i in range(self.num_layers):
            if self.net_type=='residual_out' and i==self.num_layers-1:
                out=torch.cat((x.squeeze(), out.squeeze()),-1) 
            out = self.dropout(self.layers[i](out))
        out=out.unsqueeze(1)

        #Scales output according to out_scaling flag
        if self.out_scaling==None:
            return out
        elif self.out_scaling=='L2':
            out = F.normalize(out,dim=-1)
        elif self.out_scaling=='L1':
            out = F.normalize(out,p=1,dim=-1)
        elif self.out_scaling=='softmax':
            out= self.softmax(out)        

        elif self.out_scaling=='min_max':
            out=out.squeeze()
            out=(out-torch.amin(out, 1).reshape((out.shape[0],1)))/(torch.amax(out,1).reshape((out.shape[0],1))-torch.amin(out,1).reshape((out.shape[0],1)))
            out=out.unsqueeze(1)
        elif self.out_scaling=='sigmoid':
            out=self.sigmoid(out)
        elif self.out_scaling=='max':
            out=out.squeeze()
            out=out/(torch.amax(out,1).reshape((out.shape[0],1)))
        elif self.out_scaling=='L2_max':
            out=out.squeeze()
            out=F.normalize(out,dim=-1)
        elif self.out_scaling=='sigmoid_modulus':
            out=out.squeeze()
            real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)
            modulus=torch.abs(real+1j*imag)
            theta=torch.atan2(imag, real)
            modulus=self.sigmoid(modulus)
            real=modulus*torch.cos(theta)
            imag=modulus*torch.sin(theta)
            out=torch.cat((real,imag),-1) 
            out=out.unsqueeze(1)
        elif self.out_scaling=='max':
            out=out.squeeze()
            real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)
            modulus=torch.abs(real+1j*imag)
            theta=torch.atan2(imag, real)
            modulus=(modulus)/(torch.amax(modulus,1).reshape((modulus.shape[0],1)))
            real=modulus*torch.cos(theta)
            imag=modulus*torch.sin(theta)
            out=torch.cat((real,imag),-1) 
            out=out.unsqueeze(1)
        elif self.out_scaling=='linear_error':
            Linear, error=torch.split(out, int(out.shape[-1]/2),dim=-1)
        elif self.out_scaling=='min_max_sigmoid':
            out=self.sigmoid(out)
            out=(out-torch.amin(out, 0))/(torch.amax(out,0)-torch.amin(out,0))
        elif self.out_scaling=='real_softmax':
            real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)
            real=self.softmax(real)
            out=torch.cat((real,imag),-1) 
        elif self.out_scaling=='sigmoid_arg':
            out=(self.sigmoid(out)-.5)*2*math.pi
        elif self.out_scaling=='L1':
            out = F.normalize(out,p=1.0, dim=-1)
        return out






#Same as fc_net_batch but includes a batch normalization layer in the output
class fc_net_extra(fc_net_batch):
    def __init__(self, in_dim, hidden_dims, out_dim, net_type='fc',linear_type='real', activation='relu', bias=True,threshold_val=1e-3, offset=True,dropout=0, out_scaling='L2', batch_normalization=True):
        super().__init__(in_dim, hidden_dims, out_dim, net_type='fc',linear_type=linear_type, activation=activation, bias=True,threshold_val=1e-3, offset=True,dropout=0, out_scaling=out_scaling, batch_normalization=True)
        self.extra_batch=nn.BatchNorm1d(int(out_dim*2))

        if net_type=='conv':
            self.num_conv_layers=3
            self.conv_layers=nn.ModuleList([nn.Sequential(                
            nn.Conv2d(3,3,7,padding='same'),
            nn.BatchNorm2d(3),
            nn.ReLU()) for i in range(self.num_conv_layers-1)])  
            self.conv_layers.append(nn.Conv2d(3,1,7,padding='same')) 
        else:
            self.num_conv_layers=0 
    def forward(self, x):
        x=x.squeeze()
        if self.num_conv_layers>0:
            x=x.view(-1, 10,int(len(x[1])/10))
            x=x.unsqueeze(1)
            x=x.repeat(1,3,1,1)
            for i in range(self.num_conv_layers):
                x=self.conv_layers[i](x)
            x=x.squeeze()
            x=x.view(-1, x.shape[1]*x.shape[2])
        out=super().forward(x)
        out=self.extra_batch(out.squeeze())
        return out

class E_C_repeated(torch.nn.Module):
    def __init__(self,  decoder, num_repeats, *encoder_params, **kwargs):
        super(E_C_repeated,self).__init__()
        self.decoder=decoder
        self.encoders = nn.ModuleList([fc_net_extra(*encoder_params,**kwargs) for i in range(num_repeats)])
        self.leaky=nn.LeakyReLU()
    def forward(self, x):
        for i in range(len(self.encoders)):
            x=self.encoders[i](x)
            if i!=len(self.encoders)-1:
                x=self.leaky(x)
                x_max, _=torch.max(abs(x), dim=-1, keepdim=True)
                x=abs(x)/x_max
                x=self.decoder(x)
        return x
        


#Essentially just a wrapper for the fc_net_batch class. returns #FC_net_batch(x), Linear_layer(x)
class residual_L_E(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, net_type='fc', linear_type='real', activation='relu', bias=True, threshold_val=0.001, offset=True, dropout=0, out_scaling='L2',batch_normalization=True):
        super(residual_L_E, self).__init__()
        self.fc=fc_net_batch(in_dim, hidden_dims, out_dim, net_type, linear_type, activation, bias, threshold_val, offset, dropout, out_scaling, batch_normalization=batch_normalization)
        self.linear_layer=linear_layer_wrapper(in_dim, out_dim, bias=False, activation='Linear_layer', batch_normalization=False)
    def forward(self, x):
        fc_part=self.fc(x.squeeze())
        lin_part=self.linear_layer(x.squeeze())
        lin_part=lin_part.unsqueeze(1)
        return fc_part, lin_part
    
#another wrapper for fc_net_batch    
class residual_out(fc_net_batch):
    def __init__(self, in_dim, hidden_dims, out_dim, net_type='fc', linear_type='real', activation='relu', bias=True, threshold_val=0.001, offset=True, dropout=0, out_scaling='L2'):
        super().__init__(in_dim, hidden_dims, out_dim, net_type, linear_type, activation, bias, threshold_val, offset, dropout, out_scaling)
        
    def forward(self, x):
        out=x.squeeze()
        for i in range(self.num_layers):
            print(out.shape)
            if self.net_type=='residual_out' and i==self.num_layer-1:
                #print(x.shape, out.shape)
                out=torch.cat((x.squeeze(), out.squeeze()),-1) 
            out = self.dropout(self.layers[i](out))

        
        out=out.unsqueeze(1)
        if self.out_scaling==None:
            return out
        elif self.out_scaling=='L2':
            out = F.normalize(out,dim=-1)
        elif self.out_scaling=='softmax':
            out= self.out_scaler(out)        
        elif 'sigmoid_' in self.out_scaling:
            out=self.out_scaler(out)
            out=self.s*F.normalize(out, p=2,dim=-1)
        elif self.out_scaling=='min_max':
            out=(out-torch.amin(out, 0))/(torch.amax(out,0)-torch.amin(out,0))
        elif self.out_scaling=='min_max_sigmoid':
            out=self.out_scaler(out)
            out=(out-torch.amin(out, 0))/(torch.amax(out,0)-torch.amin(out,0))
        elif self.out_scaling=='max':
            out=self.out_scaler(out)
            out=out/(torch.amax(out, 0))
        elif self.out_scaling=='real_softmax':
            real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)
            real=self.softmax(real)
            out=torch.cat((real,imag),-1) 
        elif self.out_scaling=='real_sigmoid':
            real, imag=torch.split(out, int(out.shape[-1]/2),dim=-1)
            real=self.sigmoid(real)
            out=torch.cat((real,imag),-1) 
        elif self.out_scaling!=None:
            out= self.out_scaler(out)        
        return out

    
 
    
    





    

#Softmax with ||.||_1= s, for some sparsity parameter s
class Softmax_s(nn.Module):
    def __init__(self,s) -> None:
        super(Softmax_s, self).__init__()
        self.sparsity=s
        self.softmax=nn.Softmax(-1)
    def forward(self, x):
        return self.sparsity*self.softmax(x)




    

