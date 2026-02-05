import os, argparse, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

# ----------------- 모델 정의 -----------------
class CompoundEncoder(nn.Module):
    def __init__(self,chemberta_model_name=None,ap_dim=550,hidden_dim=768,proj_dim=256,mode="chemberta"):
        super().__init__()
        self.mode=mode
        self.use_chemberta=chemberta_model_name is not None and mode in ["chemberta","both"]
        self.use_ap=mode in ["ap550","both"]
        if self.use_chemberta:
            self.chemberta=AutoModel.from_pretrained(chemberta_model_name)
            cb_hidden=self.chemberta.config.hidden_size
        else:
            self.chemberta=None
            cb_hidden=0
        if self.use_ap:
            self.ap_mlp=nn.Sequential(
                nn.Linear(ap_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU()
            )
            ap_hidden=hidden_dim
        else:
            self.ap_mlp=None
            ap_hidden=0
        in_dim=cb_hidden+ap_hidden
        if in_dim==0:
            raise ValueError("CompoundEncoder: at least one of ChemBERTa or AP must be used.")
        self.proj=nn.Sequential(
            nn.Linear(in_dim,proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim,proj_dim)
        )
        self.norm=nn.LayerNorm(proj_dim)
    def forward(self,smiles_inputs=None,ap_vec=None):
        feats=[]
        if self.use_chemberta:
            if smiles_inputs is None:
                raise ValueError("ChemBERTa mode requires smiles_inputs")
            out=self.chemberta(**smiles_inputs)
            if hasattr(out,"pooler_output") and out.pooler_output is not None:
                cb_emb=out.pooler_output
            else:
                cb_emb=out.last_hidden_state[:,0]
            feats.append(cb_emb)
        if self.use_ap:
            if ap_vec is None:
                raise ValueError("AP mode requires ap_vec")
            ap_emb=self.ap_mlp(ap_vec)
            feats.append(ap_emb)
        h=torch.cat(feats,dim=-1) if len(feats)>1 else feats[0]
        z=self.proj(h)
        z=self.norm(z)
        z=F.normalize(z,dim=-1)
        return z

class AssayEncoder(nn.Module):
    """
    BioBERT(assay text) + ProtBERT/ProtT5(target sequence) + optional continuous features
    """
    def __init__(self,
                 assay_model_name,
                 prot_model_name=None,
                 cont_dim=None,
                 cont_hidden_dim=256,
                 proj_dim=256,
                 use_cont=True,
                 use_prot=True):
        super().__init__()
        self.assay_backbone=AutoModel.from_pretrained(assay_model_name)
        text_hidden=self.assay_backbone.config.hidden_size
        self.use_prot=use_prot and (prot_model_name is not None)
        if self.use_prot:
            self.prot_backbone=AutoModel.from_pretrained(prot_model_name)
            self.prot_tokenizer=AutoTokenizer.from_pretrained(prot_model_name,do_lower_case=False)
            prot_hidden=self.prot_backbone.config.hidden_size
        else:
            self.prot_backbone=None
            self.prot_tokenizer=None
            prot_hidden=0
        self.use_cont=use_cont and (cont_dim is not None)
        if self.use_cont:
            self.cont_mlp=nn.Sequential(
                nn.Linear(cont_dim,cont_hidden_dim),
                nn.ReLU(),
                nn.Linear(cont_hidden_dim,cont_hidden_dim),
                nn.ReLU()
            )
            cont_hidden=cont_hidden_dim
        else:
            self.cont_mlp=None
            cont_hidden=0
        in_dim=text_hidden+prot_hidden+cont_hidden
        self.proj=nn.Sequential(
            nn.Linear(in_dim,proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim,proj_dim)
        )
        self.norm=nn.LayerNorm(proj_dim)
    @staticmethod
    def mean_pool(last_hidden_state,attention_mask):
        mask=attention_mask.unsqueeze(-1).float()
        summed=(last_hidden_state*mask).sum(dim=1)
        counts=mask.sum(dim=1).clamp(min=1e-6)
        return summed/counts
    def encode_assay_text(self,assay_inputs):
        out=self.assay_backbone(**assay_inputs)
        if hasattr(out,"pooler_output") and out.pooler_output is not None:
            h_text=out.pooler_output
        else:
            h_text=out.last_hidden_state[:,0]
        return h_text
    def encode_protein(self,target_seqs,device):
        B=len(target_seqs)
        if not self.use_prot or target_seqs is None:
            return torch.zeros(B,self.prot_backbone.config.hidden_size,device=device)
        spaced=[" ".join(list(s)) for s in target_seqs]
        enc=self.prot_tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)
        out=self.prot_backbone(**enc)
        h_seq=self.mean_pool(out.last_hidden_state,enc["attention_mask"])
        return h_seq
    def forward(self,assay_inputs,target_seqs=None,cont_vec=None):
        device=next(self.parameters()).device
        h_text=self.encode_assay_text(assay_inputs)
        B=h_text.size(0)
        feats=[h_text]
        if self.use_prot:
            if target_seqs is None:
                h_prot=torch.zeros(B,self.prot_backbone.config.hidden_size,device=device)
            else:
                h_prot=self.encode_protein(target_seqs,device)
            feats.append(h_prot)
        if self.use_cont:
            if cont_vec is None:
                raise ValueError("cont_vec is required when use_cont=True")
            h_cont=self.cont_mlp(cont_vec)
            feats.append(h_cont)
        h=torch.cat(feats,dim=-1) if len(feats)>1 else feats[0]
        z=self.proj(h)
        z=self.norm(z)
        z=F.normalize(z,dim=-1)
        return z

class CompoundAssayContrastiveModel(nn.Module):
    def __init__(self,
                 chemberta_model_name,
                 assay_model_name,
                 prot_model_name=None,
                 ap_dim=550,
                 compound_hidden_dim=768,
                 assay_cont_dim=None,
                 assay_cont_hidden_dim=256,
                 proj_dim=256,
                 compound_mode="chemberta",
                 temperature=0.1,
                 use_assay_cont=True,
                 use_prot=True,
                 use_aux_similarity_head=False):
        super().__init__()
        self.comp_enc=CompoundEncoder(
            chemberta_model_name=chemberta_model_name,
            ap_dim=ap_dim,
            hidden_dim=compound_hidden_dim,
            proj_dim=proj_dim,
            mode=compound_mode
        )
        self.assay_enc=AssayEncoder(
            assay_model_name=assay_model_name,
            prot_model_name=prot_model_name,
            cont_dim=assay_cont_dim,
            cont_hidden_dim=assay_cont_hidden_dim,
            proj_dim=proj_dim,
            use_cont=use_assay_cont,
            use_prot=use_prot
        )
        self.temperature=temperature
        self.bce=nn.BCEWithLogitsLoss()
        self.use_aux_similarity_head=use_aux_similarity_head
        if use_aux_similarity_head:
            self.sim_head=nn.Sequential(
                nn.Linear(proj_dim*2,proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim,2)
            )
            self.mse=nn.MSELoss()
    def forward(self,
                smiles_inputs=None,
                ap_vec=None,
                assay_inputs=None,
                target_seqs=None,
                assay_cont_vec=None,
                labels=None,
                aux_similarity_targets=None,
                return_embeddings=False):
        z_c=self.comp_enc(smiles_inputs=smiles_inputs,ap_vec=ap_vec)
        z_a=self.assay_enc(assay_inputs,target_seqs=target_seqs,cont_vec=assay_cont_vec)
        sim=(z_c*z_a).sum(dim=-1)
        logits=sim/self.temperature
        out={"logits":logits}
        if return_embeddings:
            out["compound_emb"]=z_c
            out["assay_emb"]=z_a
        loss=None
        if labels is not None:
            labels=labels.float().to(logits.device)
            loss_contrastive=self.bce(logits,labels)
            loss=loss_contrastive
            out["loss_contrastive"]=loss_contrastive
        if self.use_aux_similarity_head and aux_similarity_targets is not None:
            pair_repr=torch.cat([z_c,z_a],dim=-1)
            pred_sim=self.sim_head(pair_repr)
            target=aux_similarity_targets.to(pred_sim.device).float()
            loss_aux=self.mse(pred_sim,target)
            out["aux_pred_similarity"]=pred_sim
            out["loss_aux_similarity"]=loss_aux
            loss=loss+loss_aux if loss is not None else loss_aux
        if loss is not None:
            out["loss"]=loss
        return out

# ----------------- 데이터셋/콜레이트 -----------------
# class Chem2AssayDataset(Dataset):
#     """
#     예시 DF 컬럼 가정:
#       smiles: str
#       assay_text: str
#       target_seq: str (AA sequence) 또는 None
#       ap_vec: np.array shape (550,)
#       assay_cont: np.array shape (C,) 또는 None
#       label: 0/1
#       aux_sim0, aux_sim1: float (optional)
#     """
#     def __init__(self,df):
#         self.df=df.reset_index(drop=True)
#     def __len__(self):
#         return len(self.df)
#     def __getitem__(self,idx):
#         row=self.df.iloc[idx]
#         item={
#             "smiles":row["smiles"],
#             "assay_text":row["assay_text"],
#             "target_seq":row.get("SEQ",None),
#             "ap_vec":np.array(row["ap_vec"],dtype=np.float32),
#             "label":float(row["Activity Outcome"])
#         }
#         if "assay_cont" in row and row["assay_cont"] is not None:
#             item["assay_cont"]=np.array(row["assay_cont"],dtype=np.float32)
#         else:
#             item["assay_cont"]=None
#         if "aux_sim0" in row and "aux_sim1" in row:
#             item["aux_sim"]=np.array([row["aux_sim0"],row["aux_sim1"]],dtype=np.float32)
#         else:
#             item["aux_sim"]=None
#         return item

# def make_collate_fn(chem_tokenizer,assay_tokenizer,use_assay_cont=True,use_aux_sim=True):
#     def collate(batch):
#         smiles=[b["smiles"] for b in batch]
#         assay_text=[b["assay_text"] for b in batch]
#         target_seqs=[b["target_seq"] if b["target_seq"] is not None else "" for b in batch]
#         ap_vec=torch.tensor([b["ap_vec"] for b in batch],dtype=torch.float32)
#         labels=torch.tensor([b["label"] for b in batch],dtype=torch.float32)
#         sm_inputs=chem_tokenizer(smiles,padding=True,truncation=True,return_tensors="pt")
#         as_inputs=assay_tokenizer(assay_text,padding=True,truncation=True,return_tensors="pt")
#         batch_dict={
#             "smiles_inputs":sm_inputs,
#             "assay_inputs":as_inputs,
#             "ap_vec":ap_vec,
#             "target_seqs":target_seqs,
#             "labels":labels
#         }
#         if use_assay_cont and any(b["assay_cont"] is not None for b in batch):
#             cont_filled=[(b["assay_cont"] if b["assay_cont"] is not None else np.zeros_like(batch[0]["assay_cont"])) for b in batch]
#             assay_cont=torch.tensor(cont_filled,dtype=torch.float32)
#             batch_dict["assay_cont_vec"]=assay_cont
#         else:
#             batch_dict["assay_cont_vec"]=None
#         if use_aux_sim and any(b["aux_sim"] is not None for b in batch):
#             aux=[(b["aux_sim"] if b["aux_sim"] is not None else np.zeros(2,dtype=np.float32)) for b in batch]
#             aux_sim=torch.tensor(aux,dtype=torch.float32)
#             batch_dict["aux_similarity_targets"]=aux_sim
#         else:
#             batch_dict["aux_similarity_targets"]=None
#         return batch_dict
#     return collate

class Chem2AssayDataset(Dataset):
    """
    정규화된 세 개의 객체를 받아서 on-the-fly로 조합하는 Dataset
    
    - pair_df: interaction table
        columns:
            compound_id
            assay_id
            Activity Outcome (label)
            (optional) aux_sim0, aux_sim1
    - compound_table: DataFrame 혹은 dict-like
        index or key: compound_id
        fields:
            smiles
            ap_vec (np.ndarray, shape (ap_dim,))
    - assay_table: DataFrame 혹은 dict-like
        index or key: assay_id
        fields:
            assay_text
            SEQ (optional)
            assay_cont (optional, np.ndarray, shape (C,))
    """
    def __init__(self, pair_df, compound_table, assay_table):
        self.pair_df = pair_df.reset_index(drop=True)
        # DataFrame이라고 가정하고 index를 id로 맞춰 놓는 걸 추천
        self.compound_table = compound_table
        self.assay_table = assay_table
    def __len__(self):
        return len(self.pair_df)
    def __getitem__(self, idx):
        row = self.pair_df.iloc[idx]
        c_id = row["compound_id"]
        a_id = row["assay_id"]
        # compound 정보 가져오기
        c_row = self.compound_table.loc[c_id]
        smiles = c_row["smiles"]
        ap_vec = np.array(c_row["ap_vec"], dtype=np.float32)
        # assay 정보 가져오기
        a_row = self.assay_table.loc[a_id]
        assay_text = a_row["assay_text"]
        target_seq = a_row.get("SEQ", None)
        # continuous assay features (optional)
        if "assay_cont" in a_row and a_row["assay_cont"] is not None:
            assay_cont = np.array(a_row["assay_cont"], dtype=np.float32)
        else:
            assay_cont = None
        # label
        label = float(row["Activity Outcome"])
        # optional auxiliary similarity
        if "aux_sim0" in row and "aux_sim1" in row:
            aux_sim = np.array([row["aux_sim0"], row["aux_sim1"]], dtype=np.float32)
        else:
            aux_sim = None
        item = {
            "smiles": smiles,
            "assay_text": assay_text,
            "target_seq": target_seq,
            "ap_vec": ap_vec,
            "assay_cont": assay_cont,
            "label": label,
            "aux_sim": aux_sim
        }
        return item

def make_collate_fn(chem_tokenizer,assay_tokenizer,use_assay_cont=True,use_aux_sim=True):
    def collate(batch):
        smiles=[b["smiles"] for b in batch]
        assay_text=[b["assay_text"] for b in batch]
        target_seqs=[b["target_seq"] if b["target_seq"] is not None else "" for b in batch]
        ap_vec=torch.tensor([b["ap_vec"] for b in batch],dtype=torch.float32)
        labels=torch.tensor([b["label"] for b in batch],dtype=torch.float32)
        sm_inputs=chem_tokenizer(smiles,padding=True,truncation=True,return_tensors="pt")
        as_inputs=assay_tokenizer(assay_text,padding=True,truncation=True,return_tensors="pt")
        batch_dict={
            "smiles_inputs":sm_inputs,
            "assay_inputs":as_inputs,
            "ap_vec":ap_vec,
            "target_seqs":target_seqs,
            "labels":labels
        }
        if use_assay_cont and any(b["assay_cont"] is not None for b in batch):
            cont_filled=[(b["assay_cont"] if b["assay_cont"] is not None else np.zeros_like(batch[0]["assay_cont"])) for b in batch]
            assay_cont=torch.tensor(cont_filled,dtype=torch.float32)
            batch_dict["assay_cont_vec"]=assay_cont
        else:
            batch_dict["assay_cont_vec"]=None
        if use_aux_sim and any(b["aux_sim"] is not None for b in batch):
            aux=[(b["aux_sim"] if b["aux_sim"] is not None else np.zeros(2,dtype=np.float32)) for b in batch]
            aux_sim=torch.tensor(aux,dtype=torch.float32)
            batch_dict["aux_similarity_targets"]=aux_sim
        else:
            batch_dict["aux_similarity_targets"]=None
        return batch_dict
    return collate


# ----------------- DDP 셋업/학습 루프 -----------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(rank,args):
    local_rank=setup_ddp()
    device=torch.device(f"cuda:{local_rank}")
    set_seed(args.seed+local_rank)
    # 데이터 로드 (예: 피클/파켓 등)
    df=pd.read_pickle(args.data_path)
    dataset=Chem2AssayDataset(df)
    sampler=DistributedSampler(dataset,shuffle=True)
    chem_tok=AutoTokenizer.from_pretrained(args.chemberta_name)
    assay_tok=AutoTokenizer.from_pretrained(args.assay_model_name)
    collate_fn=make_collate_fn(chem_tok,assay_tok,use_assay_cont=(args.assay_cont_dim>0),use_aux_sim=args.use_aux_sim_head)
    loader=DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    model=CompoundAssayContrastiveModel(
        chemberta_model_name=args.chemberta_name,
        assay_model_name=args.assay_model_name,
        prot_model_name=args.prot_model_name,
        ap_dim=args.ap_dim,
        compound_hidden_dim=args.compound_hidden_dim,
        assay_cont_dim=(args.assay_cont_dim if args.assay_cont_dim>0 else None),
        assay_cont_hidden_dim=args.assay_cont_hidden_dim,
        proj_dim=args.proj_dim,
        compound_mode=args.compound_mode,
        temperature=args.temperature,
        use_assay_cont=(args.assay_cont_dim>0),
        use_prot=True,
        use_aux_similarity_head=args.use_aux_sim_head
    ).to(device)
    model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=GradScaler()
    for epoch in range(1,args.epochs+1):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss=0.0
        n_steps=0
        for batch in loader:
            # 텐서들만 device로
            batch["smiles_inputs"]= {k:v.to(device) for k,v in batch["smiles_inputs"].items()}
            batch["assay_inputs"]= {k:v.to(device) for k,v in batch["assay_inputs"].items()}
            batch["ap_vec"]=batch["ap_vec"].to(device)
            batch["labels"]=batch["labels"].to(device)
            if batch["assay_cont_vec"] is not None:
                batch["assay_cont_vec"]=batch["assay_cont_vec"].to(device)
            if batch["aux_similarity_targets"] is not None:
                batch["aux_similarity_targets"]=batch["aux_similarity_targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out=model(
                    smiles_inputs=batch["smiles_inputs"],
                    ap_vec=batch["ap_vec"],
                    assay_inputs=batch["assay_inputs"],
                    target_seqs=batch["target_seqs"],
                    assay_cont_vec=batch["assay_cont_vec"],
                    labels=batch["labels"],
                    aux_similarity_targets=batch["aux_similarity_targets"],
                    return_embeddings=False
                )
                loss=out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss+=loss.item()
            n_steps+=1
        avg_loss=epoch_loss/max(1,n_steps)
        if dist.get_rank()==0:
            print(f"[Epoch {epoch}] loss={avg_loss:.6f}")
            os.makedirs(args.out_dir,exist_ok=True)
            ckpt_path=os.path.join(args.out_dir,f"contrastive_epoch{epoch}.pt")
            torch.save({"epoch":epoch,"model_state":model.module.state_dict(),"optimizer_state":optimizer.state_dict()},ckpt_path)
    dist.destroy_process_group()

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data_path",type=str,required=True)
    p.add_argument("--out_dir",type=str,default=f"./ckpts")
    p.add_argument("--chemberta_name",type=str,default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--assay_model_name",type=str,default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--prot_model_name",type=str,default="Rostlab/prot_bert")
    p.add_argument("--ap_dim",type=int,default=550)
    p.add_argument("--compound_hidden_dim",type=int,default=768)
    p.add_argument("--assay_cont_dim",type=int,default=0)
    p.add_argument("--assay_cont_hidden_dim",type=int,default=256)
    p.add_argument("--proj_dim",type=int,default=256)
    p.add_argument("--compound_mode",type=str,default="both",choices=["chemberta","ap550","both"])
    p.add_argument("--temperature",type=float,default=0.1)
    p.add_argument("--use_aux_sim_head",dest="use_aux_sim_head",action="store_true")
    p.add_argument("--batch_size",type=int,default=8)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    train(0,args)


#CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_contrastive_ddp.py --data_path your_data.pkl
#CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_contrastive_ddp.py --data_path your_data.pkl --use_aux_sim_head
