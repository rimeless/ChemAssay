import os,argparse
import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,average_precision_score

class Chem2AssayDataset(Dataset):
    def __init__(self,biodf2,aid2seq,apdf,smidf=None):
        self.df=biodf2.reset_index(drop=True)
        self.aid2seq=aid2seq.set_index("Protein Accession") if "Protein Accession" in aid2seq.columns else aid2seq
        self.apdf=apdf
        self.smidf=smidf
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
        cids=[b["cid"] for b in batch]
        aids=[b["aid"] for b in batch]
        prots=[b["prot"] for b in batch]
        assay_text=[b["assay_text"] for b in batch]
        target_seqs=[b["target_seq"] for b in batch]
        ap_vec=torch.tensor([b["ap_vec"] for b in batch],dtype=torch.float32)
        labels=torch.tensor([b["label"] for b in batch],dtype=torch.float32)
        as_inputs=assay_tok(assay_text,padding=True,truncation=True,return_tensors="pt")
        return {"cids":cids,"aids":aids,"prots":prots,"assay_inputs":as_inputs,"target_seqs":target_seqs,"ap_vec":ap_vec,"labels":labels}
    return collate

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
    def __init__(self,num_features=550,proj_dim=256,hidden_dim=512):
        super().__init__()
        self.backbone=APMLMEncoder(num_features=num_features)
        self.proj=nn.Sequential(nn.Linear(num_features,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,proj_dim))
        self.norm=nn.LayerNorm(proj_dim)
    def forward(self,ap_vec):
        mask_zero=torch.zeros_like(ap_vec,dtype=torch.bool,device=ap_vec.device)
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

class Chem2AssayModelAPM(nn.Module):
    def __init__(self,assay_model,prot_model,temp=0.1,proj_dim=256):
        super().__init__()
        self.comp=APMCompoundEncoder(num_features=550,proj_dim=proj_dim,hidden_dim=512)
        self.assay=AssayEncoder(assay_model,prot_model,proj_dim=proj_dim)
        self.temp=temp

def eval_main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    biodf2_test=pd.read_pickle(args.test_biodf2)
    aid2seq=pd.read_pickle(args.aid2seq)
    apdf=pd.read_pickle(args.apdf)
    smidf=pd.read_pickle(args.smidf) if args.smidf is not None else None
    if smidf is not None:
        valid=apdf.index.intersection(smidf.index)
        biodf2_test=biodf2_test[biodf2_test["CID"].isin(valid)].reset_index(drop=True)
    else:
        biodf2_test=biodf2_test[biodf2_test["CID"].isin(apdf.index)].reset_index(drop=True)
    dataset=Chem2AssayDataset(biodf2_test,aid2seq,apdf,smidf)
    assay_tok=AutoTokenizer.from_pretrained(args.assay_model)
    collate_fn=make_collate_fn(assay_tok)
    loader=DataLoader(dataset,batch_size=args.bs,shuffle=False,num_workers=4,collate_fn=collate_fn)
    model=Chem2AssayModelAPM(args.assay_model,args.prot_model,args.temp,args.proj_dim).to(device)
    ckpt=torch.load(args.ckpt,map_location=device)
    if "model" in ckpt:
        state=ckpt["model"]
    elif "model_state" in ckpt:
        state=ckpt["model_state"]
    else:
        state=ckpt
    model.load_state_dict(state)
    model.eval()
    all_labels=[];all_logits=[];all_probs=[]
    all_cids=[];all_aids=[];all_prots=[]
    comp_emb_list=[];assay_emb_list=[]
    with torch.no_grad():
        for b in tqdm(loader,desc="Eval",ncols=100):
            b["assay_inputs"]={k:v.to(device) for k,v in b["assay_inputs"].items()}
            ap_vec=b["ap_vec"].to(device)
            labels=b["labels"].to(device)
            zc=model.comp(ap_vec)
            za=model.assay(b["assay_inputs"],b["target_seqs"])
            logits=(zc*za).sum(-1)/model.temp
            probs=torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_cids.extend(b["cids"])
            all_aids.extend(b["aids"])
            all_prots.extend(b["prots"])
            comp_emb_list.append(zc.cpu().numpy())
            assay_emb_list.append(za.cpu().numpy())
    labels_np=np.concatenate(all_labels,axis=0)
    logits_np=np.concatenate(all_logits,axis=0)
    probs_np=np.concatenate(all_probs,axis=0)
    comp_emb_np=np.concatenate(comp_emb_list,axis=0)
    assay_emb_np=np.concatenate(assay_emb_list,axis=0)
    try:
        auroc=roc_auc_score(labels_np,probs_np)
        auprc=average_precision_score(labels_np,probs_np)
    except Exception as e:
        print("AUROC/AUPRC 계산 중 오류:",e);auroc=auprc=None
    print(f"Test AUROC: {auroc:.4f}" if auroc is not None else "Test AUROC: None")
    print(f"Test AUPRC: {auprc:.4f}" if auprc is not None else "Test AUPRC: None")
    out_dir=args.out_dir;os.makedirs(out_dir,exist_ok=True)
    pair_df=pd.DataFrame({"CID":all_cids,"AID":all_aids,"Protein":all_prots,"label":labels_np,"logit":logits_np,"prob":probs_np})
    pair_path=os.path.join(out_dir,"test_pairs_with_scores.pkl")
    pair_df.to_pickle(pair_path)
    comp_df=pd.DataFrame(comp_emb_np);comp_df.insert(0,"CID",all_cids)
    assay_df=pd.DataFrame(assay_emb_np);assay_df.insert(0,"AID",all_aids)
    comp_agg=comp_df.groupby("CID").mean()
    assay_agg=assay_df.groupby("AID").mean()
    comp_agg_path=os.path.join(out_dir,"compound_embeddings_by_CID.pkl")
    assay_agg_path=os.path.join(out_dir,"assay_embeddings_by_AID.pkl")
    comp_agg.to_pickle(comp_agg_path);assay_agg.to_pickle(assay_agg_path)
    print("Saved:");print(" pair-level:",pair_path);print(" compound embeddings:",comp_agg_path);print(" assay embeddings:",assay_agg_path)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--test_biodf2",type=str,required=True)
    p.add_argument("--aid2seq",type=str,required=True)
    p.add_argument("--apdf",type=str,required=True)
    p.add_argument("--smidf",type=str,default=None)
    p.add_argument("--ckpt",type=str,required=True)
    p.add_argument("--out_dir",type=str,default="./eval_out_apm")
    p.add_argument("--assay_model",type=str,default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--prot_model",type=str,default="Rostlab/prot_bert")
    p.add_argument("--bs",type=int,default=16)
    p.add_argument("--temp",type=float,default=0.1)
    p.add_argument("--proj_dim",type=int,default=256)
    return p.parse_args()

if __name__=="__main__":
    args=parse()
    eval_main(args)


CUDA_VISIBLE_DEVICES=6 python eval_apm.py \
  --test_biodf2 test_with_Ptn_sample.pkl \
  --aid2seq alldf_pivot.pkl \
  --apdf pubchem_ap_opt.pkl \
  --smidf pubchem_smiles_opt.pkl \
  --ckpt ckpts_apm_contrastive/ckpt_epoch1.pt \
  --out_dir eval_apm_epoch1


CUDA_VISIBLE_DEVICES=6 python eval_apm.py \
  --test_biodf2 test_with_Ptn_sample.pkl \
  --aid2seq alldf_pivot.pkl \
  --apdf pubchem_ap_opt.pkl \
  --smidf pubchem_smiles_opt.pkl \
  --ckpt ckpts_apm_contrastive/ckpt_epoch1.pt \
  --out_dir eval_apm_epoch1
