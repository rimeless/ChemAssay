import os,argparse,random
import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from transformers import AutoModel,AutoTokenizer
from torch.cuda.amp import autocast,GradScaler

class Chem2AssayDataset(Dataset):
    def __init__(self,biodf2,aid2seq,apdf,smidf):
        self.df=biodf2.reset_index(drop=True)
        self.aid2seq=aid2seq.set_index("Protein Accession") if "Protein Accession" in aid2seq.columns else aid2seq
        self.apdf=apdf
        self.smidf=smidf
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        cid=r["CID"]
        aid=r["AID"]
        prot=r["Protein Accession"]
        smiles=self.smidf.loc[cid,"smiles"]
        ap_vec=self.apdf.loc[cid].values.astype(np.float32)
        assay_text=f"Assay {aid}"
        target_seq=self.aid2seq.loc[prot,"SEQ"] if prot in self.aid2seq.index else None
        return {"smiles":smiles,"assay_text":assay_text,"target_seq":target_seq,"ap_vec":ap_vec,"assay_cont":None,"label":float(r["Activity Outcome"]),"aux_sim":None}

def make_collate_fn(chem_tok,assay_tok):
    def collate(batch):
        smiles=[b["smiles"] for b in batch]
        assay_text=[b["assay_text"] for b in batch]
        target_seqs=[b["target_seq"] for b in batch]
        ap_vec=torch.tensor([b["ap_vec"] for b in batch],dtype=torch.float32)
        labels=torch.tensor([b["label"] for b in batch],dtype=torch.float32)
        sm_inputs=chem_tok(smiles,padding=True,truncation=True,return_tensors="pt")
        as_inputs=assay_tok(assay_text,padding=True,truncation=True,return_tensors="pt")
        return {"smiles_inputs":sm_inputs,"assay_inputs":as_inputs,"target_seqs":target_seqs,"ap_vec":ap_vec,"assay_cont_vec":None,"labels":labels,"aux_similarity_targets":None}
    return collate

class CompoundEncoder(nn.Module):
    def __init__(self,chemberta_name,ap_dim=550,hidden_dim=768,proj_dim=256):
        super().__init__()
        self.chemberta=AutoModel.from_pretrained(chemberta_name)
        self.ap_mlp=nn.Sequential(nn.Linear(ap_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU())
        in_dim=self.chemberta.config.hidden_size+hidden_dim
        self.proj=nn.Sequential(nn.Linear(in_dim,proj_dim),nn.ReLU(),nn.Linear(proj_dim,proj_dim))
        self.norm=nn.LayerNorm(proj_dim)
    def forward(self,smiles_inputs,ap_vec):
        out=self.chemberta(**smiles_inputs)
        h_cb=out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:,0]
        h_ap=self.ap_mlp(ap_vec)
        h=torch.cat([h_cb,h_ap],dim=-1)
        z=self.norm(self.proj(h))
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
        B=len(target_seqs)
        H=self.prot_backbone.config.hidden_size
        has=[s is not None and s!="" for s in target_seqs]
        if not any(has):
            return torch.zeros(B,H,device=device)
        idx=[i for i,v in enumerate(has) if v]
        seqs=[" ".join(list(target_seqs[i])) for i in idx]
        enc=self.prot_tok(seqs,padding=True,truncation=True,max_length=1024,return_tensors="pt").to(device)
        out=self.prot_backbone(**enc)
        h=self.mean_pool(out.last_hidden_state,enc["attention_mask"])
        z=torch.zeros(B,H,device=device)
        z[idx]=h
        return z
    def forward(self,assay_inputs,target_seqs):
        device=next(self.parameters()).device
        out=self.assay_backbone(**assay_inputs)
        h_text=out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:,0]
        h_prot=self.encode_protein(target_seqs,device)
        h=torch.cat([h_text,h_prot],dim=-1)
        z=self.norm(self.proj(h))
        return F.normalize(z,dim=-1)
    

class Chem2AssayModel(nn.Module):
    def __init__(self,chemberta,assay_model,prot_model,temp=0.1,lambda_bce=1.0,use_infonce=True):
        super().__init__()
        self.comp=CompoundEncoder(chemberta)
        self.assay=AssayEncoder(assay_model,prot_model)
        self.temp=temp
        self.lambda_bce=lambda_bce
        self.use_infonce=use_infonce
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self,smiles_inputs,ap_vec,assay_inputs,target_seqs,labels):
        zc=self.comp(smiles_inputs,ap_vec)
        za=self.assay(assay_inputs,target_seqs)
        logits=(zc*za).sum(-1)/self.temp
        loss_bce=self.bce(logits,labels)
        loss=self.lambda_bce*loss_bce
        loss_infonce=None
        if self.use_infonce:
            linf=info_nce_loss(zc,za,labels,self.temp)
            if linf is not None:
                loss_infonce=linf
                loss=loss+loss_infonce
        return {"loss":loss,"loss_bce":loss_bce,"loss_infonce":loss_infonce,"logits":logits}


# class Chem2AssayModel(nn.Module):
#     def __init__(self,chemberta,assay_model,prot_model,temp=0.1,lambda_bce=1.0,use_bce=True,use_infonce=True):
#         super().__init__()
#         self.comp=CompoundEncoder(chemberta)
#         self.assay=AssayEncoder(assay_model,prot_model)
#         self.temp=temp
#         self.bce=nn.BCEWithLogitsLoss()
#         self.lambda_bce=lambda_bce
#         self.use_bce=use_bce
#         self.use_infonce=use_infonce
#     def forward(self,smiles_inputs,ap_vec,assay_inputs,target_seqs,labels):
#         zc=self.comp(smiles_inputs,ap_vec)      # (B, d)
#         za=self.assay(assay_inputs,target_seqs) # (B, d)
#         logits=(zc*za).sum(-1)/self.temp        # (B,)
#         total_loss=0.0
#         loss_bce=None
#         loss_infonce=None
#         if self.use_bce:
#             loss_bce=self.bce(logits,labels)
#             total_loss=total_loss+self.lambda_bce*loss_bce
#         if self.use_infonce:
#             loss_infonce=info_nce_loss(zc,za,self.temp)
#             total_loss=total_loss+loss_infonce
#         return {"loss":total_loss,"loss_bce":loss_bce,"loss_infonce":loss_infonce,"logits":logits}

# def info_nce_loss(zc,za,temp):
#     sim = zc @ za.T / temp        # (B, B)
#     labels = torch.arange(sim.size(0), device=sim.device)
#     return F.cross_entropy(sim, labels)

def info_nce_loss(zc,za,labels,temp):
    sim=zc@za.T/temp
    pos_mask=labels>0.5
    pos_idx=pos_mask.nonzero(as_tuple=False).squeeze(-1)
    if pos_idx.numel()==0:
        return None
    sim_pos=sim[pos_idx]
    target=pos_idx
    return F.cross_entropy(sim_pos,target)

# class Chem2AssayModel(nn.Module):
#     def __init__(self,chemberta,assay_model,prot_model,temp=0.1):
#         super().__init__()
#         self.comp=CompoundEncoder(chemberta)
#         self.assay=AssayEncoder(assay_model,prot_model)
#         self.temp=temp
#         self.bce=nn.BCEWithLogitsLoss()
#     def forward(self,smiles_inputs,ap_vec,assay_inputs,target_seqs,labels):
#         zc=self.comp(smiles_inputs,ap_vec)
#         za=self.assay(assay_inputs,target_seqs)
#         logits=(zc*za).sum(-1)/self.temp
#         loss=self.bce(logits,labels)
#         return loss,logits

def setup_ddp():
    dist.init_process_group("nccl")
    r=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(r)
    return r

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def train(args):
    rank=setup_ddp()
    device=torch.device(f"cuda:{rank}")
    set_seed(args.seed+rank)
    biodf2=pd.read_pickle(args.biodf2)
    aid2seq=pd.read_pickle(args.aid2seq)
    apdf=pd.read_pickle(args.apdf)
    smidf=pd.read_pickle(args.smidf)
    valid=apdf.index.intersection(smidf.index)
    biodf2=biodf2[biodf2["CID"].isin(valid)].reset_index(drop=True)
    dataset=Chem2AssayDataset(biodf2,aid2seq,apdf,smidf)
    sampler=DistributedSampler(dataset,shuffle=True)
    chem_tok=AutoTokenizer.from_pretrained(args.chemberta)
    assay_tok=AutoTokenizer.from_pretrained(args.assay_model)
    loader=DataLoader(dataset,batch_size=args.bs,sampler=sampler,num_workers=4,pin_memory=True,collate_fn=make_collate_fn(chem_tok,assay_tok))
    model=Chem2AssayModel(args.chemberta,args.assay_model,args.prot_model,args.temp).to(device)
    model=DDP(model,device_ids=[rank],find_unused_parameters=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=GradScaler()
    for e in range(1,args.epochs+1):
        sampler.set_epoch(e)
        model.train()
        tot=0;cnt=0
        for b in loader:
            b["smiles_inputs"]={k:v.to(device) for k,v in b["smiles_inputs"].items()}
            b["assay_inputs"]={k:v.to(device) for k,v in b["assay_inputs"].items()}
            b["ap_vec"]=b["ap_vec"].to(device)
            b["labels"]=b["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out=model(b["smiles_inputs"],b["ap_vec"],b["assay_inputs"],b["target_seqs"],b["labels"])
                loss=out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot+=loss.item();cnt+=1
        if dist.get_rank()==0:
            print(f"epoch {e} loss {tot/cnt:.4f}")
    dist.destroy_process_group()

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--biodf2",type=str,required=True)
    p.add_argument("--aid2seq",type=str,required=True)
    p.add_argument("--apdf",type=str,required=True)
    p.add_argument("--smidf",type=str,required=True)
    p.add_argument("--chemberta",type=str,default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--assay_model",type=str,default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--prot_model",type=str,default="Rostlab/prot_bert")
    p.add_argument("--bs",type=int,default=8)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--temp",type=float,default=0.1)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

if __name__=="__main__":
    args=parse()
    train(args)


# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train.py --biodf2 sub_biodf2.pkl --aid2seq alldf_pivot.pkl --apdf pubchem_ap_opt.pkl --smidf pubchem_smiles_opt.pkl

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --biodf2 sub_biodf2_crop.pkl --aid2seq alldf_pivot.pkl --apdf pubchem_ap_opt.pkl --smidf pubchem_smiles_opt.pkl


# aaa = pd.read_pickle(f'{adir}/sub_biodf2.pkl')
# aaa = aaa.sample(frac=1, random_state=7)


# testdf = pd.read_pickle(f'{adir}/test_biodf.pkl')

# bbb = aaa[~aaa.CID.isin(testdf.CID)]
# bbb = bbb[bbb['Activity Outcome'].isin(['Active','Inactive'])]
# bbb= bbb.iloc[0:100000,:]

# bbb['Activity Outcome'] = [1 if x=='Active' else 0 for x in bbb['Activity Outcome']]

# bbb.to_pickle(f'{adir}/sub_biodf2_crop.pkl')