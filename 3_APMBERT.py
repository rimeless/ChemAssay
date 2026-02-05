import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5,7"
class APMLMEncoder(nn.Module):
    def __init__(self,num_features=550,d_model=256,nhead=8,num_layers=4,dim_feedforward=512,dropout=0.1,mask_prob=0.15):
        super().__init__();self.num_features=num_features;self.d_model=d_model;self.mask_prob=mask_prob
        self.idx_emb=nn.Embedding(num_features,d_model);self.val_proj=nn.Linear(1,d_model);self.mask_token=nn.Parameter(torch.zeros(d_model))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=num_layers);self.head=nn.Linear(d_model,1)
        nn.init.normal_(self.idx_emb.weight,std=0.02);nn.init.normal_(self.val_proj.weight,std=0.02);nn.init.zeros_(self.val_proj.bias)
        nn.init.normal_(self.head.weight,std=0.02);nn.init.zeros_(self.head.bias);nn.init.normal_(self.mask_token,std=0.02)
    def forward(self,ap_vec,mask=None):
        B,F=ap_vec.shape;device=ap_vec.device
        if F!=self.num_features:raise ValueError(f"Expected {self.num_features}, got {F}")
        if mask is None:mask=(torch.rand(B,F,device=device)<self.mask_prob)
        idx=torch.arange(F,device=device).unsqueeze(0).expand(B,F);idx_emb=self.idx_emb(idx)
        v=ap_vec.unsqueeze(-1);val_emb=self.val_proj(v);x=idx_emb+val_emb
        masked_token=idx_emb+self.mask_token.view(1,1,-1);x=torch.where(mask.unsqueeze(-1),masked_token,x)
        h=self.encoder(x);pred=self.head(h).squeeze(-1);return pred,mask


class APMDataset(Dataset):
    def __init__(self,apm_df):self.data=torch.from_numpy(apm_df.values.astype("float32"))
    def __len__(self):return self.data.shape[0]
    def __getitem__(self,idx):return self.data[idx]


def create_dataloader(apm_df,batch_size=1024,num_workers=4,shuffle=True):
    return DataLoader(APMDataset(apm_df),batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

def train_apmlm(model,dataloader,epochs=5,lr=1e-4,device_ids=(0,1),grad_clip=1.0):
    if torch.cuda.is_available():
        main_device=torch.device(f"cuda:{device_ids[0]}");model.to(main_device)
        if len(device_ids)>1:
            dp_model=nn.DataParallel(model,device_ids=list(device_ids))
        else:
            dp_model=model
        device=main_device
    else:
        device=torch.device("cpu");model.to(device);dp_model=model
    opt=torch.optim.AdamW(dp_model.parameters(),lr=lr)
    for ep in range(1,epochs+1):
        dp_model.train();tot=0.0;n=0;pbar=tqdm(dataloader,desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            batch=batch.to(device);opt.zero_grad()
            pred,mask=dp_model(batch);mf=mask.float();den=mf.sum()
            if den.item()==0:continue
            loss=((pred-batch)**2*mf).sum()/den;loss.backward()
            if grad_clip is not None:nn.utils.clip_grad_norm_(dp_model.parameters(),grad_clip)
            opt.step();tot+=loss.item();n+=1;pbar.set_postfix({"loss":tot/max(1,n)})
        print(f"[Epoch {ep}] mean masked MSE: {tot/max(1,n):.6f}")
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, f"{adir}/apmlm_encoder_epoch{ep}_opt.pt")
    return dp_model


adir = '/spstorage/USERS/gina/Project/FD/assay/'

# apm_df = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_ap_opt.pkl')

apm_df = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_ap_opt_random.pkl')

# # # apm_df: (N,550) DataFrame
# mean.to_pickle(f"{adir}/apm_mean.pkl")
# std.to_pickle(f"{adir}/apm_std.pkl")

mean = pd.read_pickle(f"{adir}/apm_mean.pkl")
std = pd.read_pickle(f"{adir}/apm_std.pkl")

# mean=apm_df.mean(axis=0);std=apm_df.std(axis=0).replace(0,1.0)
apm_df=(apm_df-mean)/std
loader=create_dataloader(apm_df,batch_size=512,num_workers=4,shuffle=True)
model=APMLMEncoder(num_features=apm_df.shape[1],d_model=256,nhead=8,num_layers=4)
train_apmlm(model,loader,epochs=20,lr=1e-4, device_ids=(0,1))


