import os,argparse,random
import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from transformers import AutoModel,AutoTokenizer
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm

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
    
    
class APMCompoundEncoder(nn.Module):
    def __init__(self,apmlm_ckpt_path,num_features=550,proj_dim=256,hidden_dim=512,freeze_backbone=True,device="cpu"):
        super().__init__()
        self.backbone=APMLMEncoder(num_features=num_features)
        ckpt=torch.load(apmlm_ckpt_path,map_location=device)
        if "model_state" in ckpt:
            state=ckpt["model_state"]
        elif "model" in ckpt:  # 혹시 옛날 형식이면
            state=ckpt["model"]
        else:
            state=ckpt
        self.backbone.load_state_dict(state)
        self.freeze_backbone=freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        self.proj=nn.Sequential(
            nn.Linear(num_features,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,proj_dim)
        )
        self.norm=nn.LayerNorm(proj_dim)
    def forward(self,ap_vec):
        mask_zero=torch.zeros_like(ap_vec,dtype=torch.bool,device=ap_vec.device)
        if self.freeze_backbone:
            with torch.no_grad():
                pred,_=self.backbone(ap_vec,mask=mask_zero)
        else:
            pred,_=self.backbone(ap_vec,mask=mask_zero)
        z=self.proj(pred)
        z=self.norm(z)
        return F.normalize(z,dim=-1)




class AssayEncoder(nn.Module):
    def __init__(self,assay_model,prot_model,proj_dim=256):
        super().__init__()
        self.assay_backbone=AutoModel.from_pretrained(assay_model)
        self.prot_backbone=AutoModel.from_pretrained(prot_model)
        self.prot_tok=AutoTokenizer.from_pretrained(prot_model,do_lower_case=False)
        in_dim=self.assay_backbone.config.hidden_size+self.prot_backbone.config.hidden_size
        self.proj=nn.Sequential(nn.Linear(in_dim,proj_dim),nn.ReLU(),nn.Linear(proj_dim,proj_dim))
        self.norm=nn.LayerNorm(proj_dim)
    def mean_pool(self,h,m):
        m=m.unsqueeze(-1).float()
        return (h*m).sum(1)/m.sum(1).clamp(min=1e-6)
    def encode_protein(self,target_seqs,device):
        B=len(target_seqs);H=self.prot_backbone.config.hidden_size
        has=[(s is not None and s!="") for s in target_seqs]
        if not any(has):return torch.zeros(B,H,device=device)
        idx=[i for i,v in enumerate(has) if v]
        seqs=[" ".join(list(target_seqs[i])) for i in idx]
        enc=self.prot_tok(seqs,padding=True,truncation=True,max_length=1024,return_tensors="pt").to(device)
        out=self.prot_backbone(**enc)
        h=self.mean_pool(out.last_hidden_state,enc["attention_mask"])
        z=torch.zeros(B,H,device=device);z[idx]=h
        return z
    def forward(self,assay_inputs,target_seqs):
        device=next(self.parameters()).device
        out=self.assay_backbone(**assay_inputs)
        h_text=out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:,0]
        h_prot=self.encode_protein(target_seqs,device)
        h=torch.cat([h_text,h_prot],dim=-1)
        z=self.norm(self.proj(h))
        return F.normalize(z,dim=-1)

def info_nce_loss(zc,za,labels,temp):
    sim=zc@za.T/temp
    pos_mask=labels>0.5
    pos_idx=pos_mask.nonzero(as_tuple=False).squeeze(-1)
    if pos_idx.numel()==0:return None
    sim_pos=sim[pos_idx]
    target=pos_idx
    return F.cross_entropy(sim_pos,target)

class Chem2AssayModelAPM(nn.Module):
    def __init__(self,apmlm_ckpt_path,assay_model,prot_model,temp=0.1,proj_dim=256,lambda_bce=1.0,use_infonce=True,freeze_apmlm=True,device="cpu"):
        super().__init__()
        self.comp=APMCompoundEncoder(apmlm_ckpt_path,num_features=550,proj_dim=proj_dim,hidden_dim=512,freeze_backbone=freeze_apmlm,device=device)
        self.assay=AssayEncoder(assay_model,prot_model,proj_dim=proj_dim)
        self.temp=temp
        self.lambda_bce=lambda_bce
        self.use_infonce=use_infonce
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self,ap_vec,assay_inputs,target_seqs,labels):
        zc=self.comp(ap_vec)
        za=self.assay(assay_inputs,target_seqs)
        logits=(zc*za).sum(-1)/self.temp
        loss_bce=self.bce(logits,labels)
        loss=self.lambda_bce*loss_bce
        loss_infonce=None
        if self.use_infonce:
            linf=info_nce_loss(zc,za,labels,self.temp)
            if linf is not None:
                loss_infonce=linf;loss=loss+loss_infonce
        return {"loss":loss,"loss_bce":loss_bce,"loss_infonce":loss_infonce,"logits":logits}

class Chem2AssayDataset(Dataset):
    def __init__(self,biodf2,aid2seq,apdf,smidf=None):
        self.df=biodf2.reset_index(drop=True)
        self.aid2seq=aid2seq.set_index("Protein Accession") if "Protein Accession" in aid2seq.columns else aid2seq
        self.apdf=apdf;self.smidf=smidf
    def __len__(self):return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        cid=r["CID"];aid=r["AID"];prot=r["Protein Accession"]
        ap_vec=self.apdf.loc[cid].values.astype(np.float32)
        assay_text=f"Assay {aid}"
        if prot in self.aid2seq.index:target_seq=self.aid2seq.loc[prot,"SEQ"]
        else:target_seq=None
        label=float(r["Activity Outcome"])
        return {"cid":cid,"aid":aid,"prot":prot,"assay_text":assay_text,"target_seq":target_seq,"ap_vec":ap_vec,"label":label}

def make_collate_fn(assay_tok):
    def collate(batch):
        assay_text=[b["assay_text"] for b in batch]
        target_seqs=[b["target_seq"] for b in batch]
        ap_vec=torch.tensor([b["ap_vec"] for b in batch],dtype=torch.float32)
        labels=torch.tensor([b["label"] for b in batch],dtype=torch.float32)
        as_inputs=assay_tok(assay_text,padding=True,truncation=True,return_tensors="pt")
        return {"assay_inputs":as_inputs,"target_seqs":target_seqs,"ap_vec":ap_vec,"labels":labels}
    return collate

def setup_ddp():
    dist.init_process_group("nccl")
    r=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(r)
    return r

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def train(args):
    rank=setup_ddp()
    device=torch.device(f"cuda:{rank}")
    set_seed(args.seed+rank)
    biodf2=pd.read_pickle(args.biodf2)
    aid2seq=pd.read_pickle(args.aid2seq)
    apdf=pd.read_pickle(args.apdf)
    smidf=pd.read_pickle(args.smidf) if args.smidf is not None else None
    if smidf is not None:
        valid=apdf.index.intersection(smidf.index)
        biodf2=biodf2[biodf2["CID"].isin(valid)].reset_index(drop=True)
    else:
        biodf2=biodf2[biodf2["CID"].isin(apdf.index)].reset_index(drop=True)
    dataset=Chem2AssayDataset(biodf2,aid2seq,apdf,smidf)
    sampler=DistributedSampler(dataset,shuffle=True)
    assay_tok=AutoTokenizer.from_pretrained(args.assay_model)
    loader=DataLoader(dataset,batch_size=args.bs,sampler=sampler,num_workers=4,pin_memory=True,collate_fn=make_collate_fn(assay_tok))
    model=Chem2AssayModelAPM(args.apmlm_ckpt,args.assay_model,args.prot_model,args.temp,args.proj_dim,args.lambda_bce,args.use_infonce,args.freeze_apmlm,device).to(device)
    model=DDP(model,device_ids=[rank],find_unused_parameters=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=GradScaler()
    log_f=None;csv_path=None
    if dist.get_rank()==0:
        os.makedirs(args.out_dir,exist_ok=True)
        log_path=os.path.join(args.out_dir,"train.log")
        csv_path=os.path.join(args.out_dir,"train.csv")
        log_f=open(log_path,"a",buffering=1)
        if not os.path.exists(csv_path):
            with open(csv_path,"w") as cf:cf.write("epoch,loss,bce,infonce\n")
    for e in range(1,args.epochs+1):
        sampler.set_epoch(e);model.train()
        tot_loss=tot_bce=tot_infonce=0.0;cnt=infon_steps=0
        if dist.get_rank()==0:pbar=tqdm(loader,desc=f"Epoch {e}",ncols=100)
        else:pbar=loader
        for b in pbar:
            b["assay_inputs"]={k:v.to(device) for k,v in b["assay_inputs"].items()}
            b["ap_vec"]=b["ap_vec"].to(device)
            b["labels"]=b["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out=model(b["ap_vec"],b["assay_inputs"],b["target_seqs"],b["labels"])
                loss=out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt);scaler.update()
            tot_loss+=loss.item();tot_bce+=out["loss_bce"].item()
            if out["loss_infonce"] is not None:
                tot_infonce+=out["loss_infonce"].item();infon_steps+=1
            cnt+=1
            if dist.get_rank()==0:pbar.set_postfix(loss=f"{loss.item():.4f}")
        if dist.get_rank()==0:
            avg_loss=tot_loss/max(1,cnt);avg_bce=tot_bce/max(1,cnt)
            avg_infonce=tot_infonce/max(1,infon_steps) if infon_steps>0 else 0.0
            msg=f"epoch {e} loss {avg_loss:.4f} bce {avg_bce:.4f} infonce {avg_infonce:.4f}"
            print(msg);log_f.write(msg+"\n")
            with open(csv_path,"a") as cf:cf.write(f"{e},{avg_loss},{avg_bce},{avg_infonce}\n")
            ckpt_path=os.path.join(args.out_dir,f"ckpt_epoch{e}.pt")
            torch.save({"epoch":e,"model":model.module.state_dict(),"optimizer":opt.state_dict()},ckpt_path)
    if dist.get_rank()==0 and log_f is not None:log_f.close()
    dist.destroy_process_group()

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--biodf2",type=str,required=True)
    p.add_argument("--aid2seq",type=str,required=True)
    p.add_argument("--apdf",type=str,required=True)
    p.add_argument("--smidf",type=str,default=None)
    p.add_argument("--apmlm_ckpt",type=str,required=True)
    p.add_argument("--out_dir",type=str,default="./ckpts_apm")
    p.add_argument("--assay_model",type=str,default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--prot_model",type=str,default="Rostlab/prot_bert")
    p.add_argument("--bs",type=int,default=8)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--temp",type=float,default=0.1)
    p.add_argument("--proj_dim",type=int,default=256)
    p.add_argument("--lambda_bce",type=float,default=1.0)
    p.add_argument("--use_infonce",action="store_true")
    p.add_argument("--freeze_apmlm",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

if __name__=="__main__":
    args=parse()
    train(args)


CUDA_VISIBLE_DEVICES=0,1,3,6 torchrun --nproc_per_node=2 --master_port=29501 train_apm.py \
  --biodf2 sub_biodf2_crop.pkl \
  --aid2seq alldf_pivot.pkl \
  --apdf pubchem_ap_opt.pkl \
  --smidf pubchem_smiles_opt.pkl \
  --apmlm_ckpt ckpts_exp1/ckpt_epoch5.pt \
  --out_dir ./ckpts_apm_contrastive \
  --use_infonce \
  --freeze_apmlm
