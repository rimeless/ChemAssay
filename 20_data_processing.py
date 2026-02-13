import pandas as pd
import numpy as np
import pickle

LABEL_MAP={"active":1,"positive":1,"inactive":0,"negative":0,"inconclusive":2,"ambiguous":2,"unspecified":3,"unknown":3}
ASSAY_TYPE_MAP={"screening":0,"confirmatory":1,"summary":2,"other":2}


smidf = pd.read_pickle(f'{adir}/pubchem_smiles.pkl')
smidf['CID'] = list(smidf.index)
smidf['SMILES'] = smidf['smiles'].astype(str)
comp_df = smidf
assay_df = bio2.copy()

# aid2seq.columns = ['Protein Accession','SEQ']
# aid2seq.to_pickle(f'{adir}/aid2seq.pkl')


# def preprocess_from_three_files(assay_info_path,compound_info_path,interaction_path,cid_col="CID",smiles_col="SMILES"):
#     assay_df=pd.read_pickle(assay_info_path)
#     comp_df=pd.read_pickle(compound_info_path)
#     inter_df=pd.read_pickle(interaction_path)
#     for col in ["AID","BioAssay Name","Outcome Type"]:
#         if col not in assay_df.columns:
#             raise ValueError(f"assay_info must contain column '{col}'")
#     for col in [cid_col,smiles_col]:
#         if col not in comp_df.columns:
#             raise ValueError(f"compound_info must contain column '{col}'")
#     for col in ["AID",cid_col,"Activity Outcome"]:
#         if col not in inter_df.columns:
#             raise ValueError(f"interaction file must contain column '{col}'")
#     inter_df["Activity Outcome"]=inter_df["Activity Outcome"].astype(str).str.lower().str.strip()
#     assay_df["Outcome Type"]=assay_df["Outcome Type"].astype(str).str.lower().str.strip()
#     merged=inter_df.merge(assay_df[["AID","BioAssay Name","Outcome Type"]],on="AID",how="left")
#     merged=merged.merge(comp_df[[cid_col,smiles_col]],on=cid_col,how="left")
#     merged=merged.dropna(subset=[smiles_col,"BioAssay Name"])
#     def map_label(x):
#         return LABEL_MAP.get(str(x).lower().strip(),3)
#     def map_assay_type(x):
#         return ASSAY_TYPE_MAP.get(str(x).lower().strip(),2)
#     merged["label_int"]=merged["Activity Outcome"].apply(map_label)
#     merged["assay_type_int"]=merged["Outcome Type"].apply(map_assay_type)
#     unique_aids=sorted(merged["AID"].unique().tolist())
#     aid_to_assay_id={aid:i for i,aid in enumerate(unique_aids)}
#     merged["assay_id_int"]=merged["AID"].map(aid_to_assay_id)
#     compound_smiles=merged[smiles_col].astype(str).tolist()
#     assay_descriptions=merged["BioAssay Name"].astype(str).tolist()
#     assay_ids=merged["assay_id_int"].astype(int).tolist()
#     labels=merged["label_int"].astype(int).tolist()
#     assay_types=merged["assay_type_int"].astype(int).tolist()
#     N=len(merged)
#     activity_values=[None]*N
#     target_sequences=[None]*N
#     multi_dose_info=[False]*N
#     print(f"Total usable samples: {N}")
#     print(f"Num assays: {len(unique_aids)}")
#     return {"compound_smiles":compound_smiles,"assay_descriptions":assay_descriptions,"assay_ids":assay_ids,"labels":labels,"assay_types":assay_types,"activity_values":activity_values,"target_sequences":target_sequences,"multi_dose_info":multi_dose_info,"aid_to_assay_id":aid_to_assay_id}



seq_df = pd.read_pickle(f'{adir}/aid2seq.pkl')

adir = '/spstorage/USERS/gina/Project/FD/assay/'


bio2 = pd.read_csv('/spstorage/DB/PUBCHEM/assay/2026/bioassays.tsv.gz', sep='\t', compression='gzip')

biodf2 = pd.read_csv('/spstorage/DB/PUBCHEM/assay/2026/bioactivities.tsv.gz', sep='\t', compression='gzip')

assay_info_path = f'{adir}/data/assay_info.pkl'

interaction_path = f'{adir}/data/sample_interaction.pkl'
compound_info_path = f'{adir}/data/compound_info.pkl'
cid_col = "CID"
smiles_col = "SMILES"
aid2seq_path = f'{adir}/aid2seq.pkl'

bio2.loc[:,['AID','BioAssay Name','Outcome Type']].to_pickle(f'{adir}/data/assay_info.pkl')
biodf2.loc[:,['AID','CID','Activity Outcome','Activity Value','Protein Accession']].to_pickle(f'{adir}/data/interaction.pkl')
inter_df.to_pickle(f'{adir}/data/sample_interaction.pkl')
# comp_df.to_pickle(f'{adir}/data/compound_info.pkl')

abb = preprocess_with_values_and_seq(assay_info_path,compound_info_path,interaction_path,aid2seq_path,cid_col="CID",smiles_col="SMILES",value_col="Activity Value",acc_col="Protein Accession",seq_col="SEQ")
def preprocess_with_values_and_seq(assay_info_path,compound_info_path,interaction_path,aid2seq_path,cid_col="CID",smiles_col="SMILES",value_col="Activity Value",acc_col="Protein Accession",seq_col="SEQ"):
    assay_df=pd.read_pickle(assay_info_path)
    comp_df=pd.read_pickle(compound_info_path)
    inter_df=pd.read_pickle(interaction_path)
    seq_df=pd.read_pickle(aid2seq_path)
    for col in ["AID","BioAssay Name","Outcome Type"]:
        if col not in assay_df.columns:
            raise ValueError(f"assay_info must contain column '{col}'")
    for col in [cid_col,smiles_col]:
        if col not in comp_df.columns:
            raise ValueError(f"compound_info must contain column '{col}'")
    for col in ["AID",cid_col,"Activity Outcome"]:
        if col not in inter_df.columns:
            raise ValueError(f"interaction file must contain column '{col}'")
    if acc_col not in inter_df.columns:
        raise ValueError(f"interaction file must contain column '{acc_col}' for Protein Accession")
    if acc_col not in seq_df.columns or seq_col not in seq_df.columns:
        raise ValueError(f"aid2seq file must contain columns '{acc_col}' and '{seq_col}'")
    inter_df["Activity Outcome"]=inter_df["Activity Outcome"].astype(str).str.lower().str.strip()
    assay_df["Outcome Type"]=assay_df["Outcome Type"].astype(str).str.lower().str.strip()
    if value_col in inter_df.columns:
        inter_df[value_col]=pd.to_numeric(inter_df[value_col],errors="coerce")
    else:
        inter_df[value_col]=np.nan
    group_cols=["AID",cid_col]
    agg_dict={"Activity Outcome":"first",value_col:"mean",acc_col:"first"}
    inter_grp=inter_df.groupby(group_cols,as_index=False).agg(agg_dict)
    merged=inter_grp.merge(assay_df[["AID","BioAssay Name","Outcome Type"]],on="AID",how="left")
    merged=merged.merge(comp_df[[cid_col,smiles_col]],on=cid_col,how="left")
    merged=merged.merge(seq_df[[acc_col,seq_col]],on=acc_col,how="left")
    merged=merged.dropna(subset=[smiles_col,"BioAssay Name"])
    def map_label(x):
        return LABEL_MAP.get(str(x).lower().strip(),3)
    def map_assay_type(x):
        return ASSAY_TYPE_MAP.get(str(x).lower().strip(),2)
    merged["label_int"]=merged["Activity Outcome"].apply(map_label)
    merged["assay_type_int"]=merged["Outcome Type"].apply(map_assay_type)
    unique_aids=sorted(merged["AID"].unique().tolist())
    aid_to_assay_id={aid:i for i,aid in enumerate(unique_aids)}
    merged["assay_id_int"]=merged["AID"].map(aid_to_assay_id)
    def _parse_value(v):
        if pd.isna(v):
            return None
        try:
            v=float(v)
        except Exception:
            return None
        if v<=0:
            return None
        return v
    activity_values=[_parse_value(v) for v in merged[value_col].values]
    def _parse_seq(s):
        if isinstance(s,str) and len(s.strip())>0:
            return s.strip()
        return None
    target_sequences=[_parse_seq(s) for s in merged[seq_col].values]
    compound_smiles=merged[smiles_col].astype(str).tolist()
    assay_descriptions=merged["BioAssay Name"].astype(str).tolist()
    assay_ids=merged["assay_id_int"].astype(int).tolist()
    labels=merged["label_int"].astype(int).tolist()
    assay_types=merged["assay_type_int"].astype(int).tolist()
    N=len(merged)
    multi_dose_info=[False]*N
    print(f"Total usable samples: {N}")
    print(f"Num assays: {len(unique_aids)}")
    print(f"Activity values non-null: {sum(v is not None for v in activity_values)} ({100.0*sum(v is not None for v in activity_values)/max(N,1):.2f}%)")
    print(f"Target sequences non-null: {sum(s is not None for s in target_sequences)} ({100.0*sum(s is not None for s in target_sequences)/max(N,1):.2f}%)")
    return {"compound_smiles":compound_smiles,"assay_descriptions":assay_descriptions,"assay_ids":assay_ids,"labels":labels,"assay_types":assay_types,"activity_values":activity_values,"target_sequences":target_sequences,"multi_dose_info":multi_dose_info,"aid_to_assay_id":aid_to_assay_id}