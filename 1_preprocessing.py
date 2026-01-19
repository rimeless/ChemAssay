
import pandas as pd
import glob
zincs = glob.glob('/spstorage/DB/ZINC/drug-like-clean-substance/*')
zincdir = '/spstorage/DB/ZINC/drug-like-clean-substance'
maindir = '/spstorage/USERS/gina/Project/Chemcriptome'
zincdf = pd.read_csv('/spstorage/DB/ZINC/ZINC-downloader-2D-txt.wget', sep=' ', header=None)
zincs = list(zincdf.iloc[:,7])

zsmis = []
zids=  []
zinchis = []
for zz in zincs:
    zdf = pd.read_csv(f'{zincdir}/{zz}',sep='\t')
    zsmis = zsmis + list(zdf['smiles'])
    zids = zids + list(zdf['zinc_id'])
    zinchis = zinchis + list(zdf['inchikey'])


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rdkit.Chem import Recap, BRICS

# Example SMILES
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
text = "[CLS]CC(=O)Oc1ccccc1C(=O)O"
text = '[CLS]Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1'
lines = ['Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1',"CC(=O)Oc1ccccc1C(=O)O"]



with open(f'{maindir}/property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)



#### example

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from transformers import BertTokenizer, BertModel, BertConfig

# Example SMILES
smiles_list = ["CCO", "CC(N)C(=O)O", "C1=CC=CC=C1"]

# Create a simple character-level tokenizer for SMILES
class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]  # Special tokens for Transformers
        self.vocab = self.special_tokens + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=()[]+-#/\\")
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}
    def tokenize(self, smiles):
        return [char for char in smiles]
    def convert_tokens_to_ids(self, tokens):
        return [self.token2id[token] for token in tokens if token in self.token2id]
    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        token_ids = self.convert_tokens_to_ids(tokens)
        return [self.token2id["[CLS]"]] + token_ids + [self.token2id["[SEP]"]]  # Add special tokens


# Initialize tokenizer
tokenizer = SMILESTokenizer()

# Tokenize and encode
tokenized_smiles = [tokenizer.encode(smiles) for smiles in smiles_list]
print("Tokenized SMILES:", tokenized_smiles)

import torch
import torch.nn as nn
import torch.optim as optim

class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_heads=4, num_layers=2):
        super(SMILESTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer
    def forward(self, x):
        x = self.embedding(x)  # Convert tokens to embeddings
        x = self.transformer(x)  # Pass through Transformer encoder
        x = self.fc(x)  # Output logits
        return x

# Initialize model
vocab_size = len(tokenizer.vocab)
model = SMILESTransformer(vocab_size)

# Convert tokenized SMILES to a tensor
max_len = max(len(seq) for seq in tokenized_smiles)
padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in tokenized_smiles]
input_tensor = torch.tensor(padded_sequences)

# Forward pass through the Transformer
output = model(input_tensor)
print("Transformer Output Shape:", output.shape)  # Expected: [batch_size, seq_length, vocab_size]

# Dummy labels (for self-supervised learning like masked token prediction)
labels = input_tensor.clone()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_tensor)  # Forward pass
    loss = criterion(output.view(-1, vocab_size), labels.view(-1))  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")





mol = Chem.MolFromSmiles(smiles)

# RECAP Fragmentation (breaks at ester, amide, etc.)
recap_frags = Recap.RecapDecompose(mol)
print("RECAP Fragments:", recap_frags.GetAllChildren().keys())

# BRICS Fragmentation (breaks at synthetic points)
brics_frags = list(BRICS.BRICSDecompose(mol, minFragmentSize=4))
print("BRICS Fragments:", brics_frags)

vocab_filename = f'{maindir}/vocab_bpe_300.txt'

tokenizer = BertTokenizer(vocab_file = vocab_filename, do_lower_case = False, do_basic_tokenize = False)
tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)


# foundation model with property prediction
# token 별 embedding을 구하고, 이를 transformer에 넣어서 property를 예측하는 모델을 만들어보자
# 이를 위해서는, token을 embedding으로 바꾸는 과정이 필요하다.
# 이를 위해서는, BPE를 사용할 수 있을 것이다.
# BPE를 사용하기 위해서는, vocab을 만들어야 한다.
# 이를 위해서는, token을 먼저 만들어야 한다.
# 이를 위해서는, SMILES를 tokenize하는 방법을 알아야 한다.

# 1
# 이미 되어있는 SMILES tokenizer를 사용해서 MLM 그냥 고
# 1-1: 마지막 layer에 compound property 예측
# 1-1-2: target 예측 추가
# 1-2: 마지막 layer에 target 예측

# 2 functional group의 property 함께 가는 모델 

# down: 해당 property 가지는 compound 생성

dtip = '/spstorage/USERS/gina/Project/AP/DB/InterpretableDTIP'
pert2etid = pd.read_csv('/spstorage/USERS/gina/Project/old/OA2/pert2etid.csv',index_col=0)
cid2ecid = pd.read_csv(dtip+'/cid2ecid.csv', index_col = 0)
write.csv(paste0(cmps,collapse= '\n'), file = '/spstorage/USERS/gina/cmps.csv', row.names = F)



p2z = pd.read_csv(f'{maindir}/pert2zinc.txt',sep='\t',header=None)
p2z = p2z.dropna()

for z in list(set(p2z.iloc[:,1])):
    if 'ZINC' in z:
        zz = z.split('ZINC')[1]

zzs = [int(z.split('ZINC')[1]) for z in list(set(p2z.iloc[:,1]))]

zsmis = []
zids=  []
zinchis = []
for zz in zincs:
    zdf = pd.read_csv(f'{zincdir}/{zz}',sep='\t')
    zdf = zdf[zdf.zinc_id.isin(zzs)]
    if len(zdf)!=0:
        zsmis = zsmis + list(zdf['smiles'])
        zids = zids + list(zdf['zinc_id'])
        zinchis = zinchis + list(zdf['inchikey'])


zsmis2 = []
zids2=  []
zinchis2 = []
cds = list(set(adff.cid))
cdss = '\n'.join([str(int(x)) for x in cds])
with open(f'{maindir}/cds.txt', 'w') as f:
    f.write(cdss)

with open(f'{maindir}/smis.txt', 'w') as f:
    f.write('\n'.join(smis))



ssmidf= smidf[smidf.iloc[:,0].isin(ap.index)].sample(3*10**5, random_state=42)


ssmis =list(ssmidf.iloc[:,1])


with open(f'{maindir}/SPMM/data/pretrain_300K.txt', 'w') as f:
    f.write('\n'.join(ssmis))


ssmis = list(set(smidf[smidf.iloc[:,0].isin(ap1a.index)].iloc[:,1]) - set(ssmis))


with open(f'{maindir}/SPMM/data/test_20K.txt', 'w') as f:
    f.write('\n'.join(ssmis))

# '\n'.join(ssmis)
# write.csv(paste0(cmps,collapse= '\n'), file = f'{maindir}/SPMM/data/pretrain_320K', row.names = F)




    
for zz in zincs:
    zdf = pd.read_csv(f'{zincdir}/{zz}',sep='\t')
    zdf = zdf[zdf.zinc_id.isin(adff.cid)]
    if len(zdf)!=0:
        zsmis2 = zsmis2 + list(zdf['smiles'])
        zids2 = zids2 + list(zdf['zinc_id'])
        zinchis2 = zinchis2 + list(zdf['inchikey'])


ap1a = ap1.T[ap1.T.index.isin(ssmm11[ssmm11>=300].index)].T



ap1a.values()


cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(ap1a.columns)]
cvv = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + ['##'+v for v in cvv]
with open(f'{maindir}/SPMM/vocab_ap_368.txt', 'w') as f:
    f.write('\n'.join(cvv))

aptcks = []
for vv in ap1a.values:
    aptcks.append(''.join(np.array(cvv)[vv==1]))


cid2aptck = dict(zip(ap1a.index, aptcks))
train_aptcks = [cid2aptck[c] for c in list(ssmidf.iloc[:,0])]
smi2aptck = dict(zip(ssmidf.iloc[:,1], train_aptcks))



all_aptcks = [cid2aptck[c] for c in list(smidf[smidf.iloc[:,0].isin(ap1a.index)].iloc[:,0])]


smi2aptck = dict(zip(smidf[smidf.iloc[:,0].isin(ap1a.index)].iloc[:,1], all_aptcks))


# with open(f'{maindir}/SPMM/data/pretrain_ap_300K.txt', 'w') as f:
#     f.write('\n'.join(train_aptcks))

with open(f'{maindir}/SPMM/data/smi2aptck.pkl', 'wb') as f:
    pickle.dump(smi2aptck, f)


iap1a = iapdf.iloc[:,list(ap1a.columns)]
iap1a[iap1a>=1] = 1
iap1a[iap1a<1] = 0

for cd in list(iap1a.index):
    if cid2smi2[cd] not in smi2aptck:
        smi2aptck[cid2smi2[cd]] = ''.join(np.array(cvv)[iap1a.iloc[0,:].values==1])


aps = pd.concat([ap, iapdf[~iapdf.index.isin(ap.index)]])
aps = aps.loc[:,list(ap1a.columns)]


smi2aptck368 = dict()
for cd in list(aps.index)[50901:]:
    vv= aps.loc[cd,:].values
    tdf = pd.DataFrame({'cvv':cvv,'vv':vv})
    tdf = tdf.sort_values('vv',ascending=False)
    tdf = tdf[tdf.vv>0]
    if cid2smi2[cd] not in smi2aptck368:
        smi2aptck368[cid2smi2[cd]] = ''.join(tdf.cvv.values)


for cd in list(aps.index)[200451:]:
    vv= aps.loc[cd,:].values
    tdf = pd.DataFrame({'cvv':cvv,'vv':vv})
    tdf = tdf.sort_values('vv',ascending=False)
    tdf = tdf[tdf.vv>0]
    if cid2smi2[cd] not in smi2aptck368:
        smi2aptck368[cid2smi2[cd]] = ''.join(tdf.cvv.values)


for i,vv in enumerate(list(smi2aptck368.values())):
    if len(vv)==0:
        print(i)
        print(list(smi2aptck368.keys())[i])
        # smi2aptck368[list(smi2aptck368.keys())[i]] = '##[UNK]'


with open(f'{maindir}/SPMM/data/smi2aptck368.pkl', 'wb') as f:
    pickle.dump(smi2aptck368, f)


aaps = pd.concat([ap, iapdf[~iapdf.index.isin(ap.index)]])

aaps = aaps.loc[:,list(apmap.columns)]



cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(aaps.columns)]
cvv = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + ['##'+v for v in cvv]
with open(f'{maindir}/SPMM/vocab_ap_59.txt', 'w') as f:
    f.write('\n'.join(cvv))


python SPMM_smi2ap_cl_55_sorting.py --data_path './for_CL_smiles.txt'

python SPMM_ap2ap_cl_55_sorting.py --data_path './for_CL_smiles.txt'

smi2aptck55 = dict()
for cd in list(aaps.index)[200451:]:
    vv= aaps.loc[cd,:].values
    tdf = pd.DataFrame({'cvv':cvv,'vv':vv})
    tdf = tdf.sort_values('vv',ascending=False)
    tdf = tdf[tdf.vv>0]
    smi2aptck55[cid2smi2[cd]] = ''.join(tdf.cvv.values)


with open(f'{maindir}/SPMM/data/smi2aptck55.pkl', 'wb') as f:
    pickle.dump(smi2aptck55, f)




import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


ap = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/AP_320K.pkl')

def get_atc(cd):
    data = requests.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/xml".format(str(cd)))
    html = BeautifulSoup(data.content, "xml")
    uu=html.findAll('URL')
    atc = [str(ui).split('=')[-1].split('<')[0] for ui in uu if 'atc_ddd_index' in str(ui)]
    if len(atc)==0:atc = [0]*7
    return tuple(atc)


cdcd = list(ap.index)


atcs = []
for i,cd in enumerate(cdcd[50:]):
    if i%1000==0:
        print(i)
    data = requests.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/xml".format(str(cd)))
    html = BeautifulSoup(data.content, "xml")
    uu=html.findAll('URL')
    atc = [str(ui).split('=')[-1].split('<')[0] for ui in uu if 'atc_ddd_index' in str(ui)]
    if len(atc)!=0:
        atcs.append([cd] + atc[0:-1])


kks = []
cdir = '/spstorage/USERS/gina/Project/Chemcriptome'
for i in range(1,7):
    with open (f'{cdir}/atc_code{i}.json','r') as file:
        atcc=json.load(file)
    for kk in atcc['Annotations']['Annotation']:
        nm = kk['Name']
        src = kk['SourceID']
        if 'LinkedRecords' in kk.keys():
            for cid in kk['LinkedRecords']['CID']:
                kks.append([nm, src, src[0:3], src[0:4], src[0:5], cid, len(kk['LinkedRecords']['CID'])])
        else:
            kks.append([nm, src, src[0:3], src[0:4], src[0:5], '-',0])


kkdf = pd.DataFrame(kks, columns = ['Name', 'SourceID', 'ATC1', 'ATC2', 'ATC3', 'CID', 'nCID'])
            
kkdf[kkdf.CID!='-']
    


atcap = pd.read_pickle(f'{cdir}/atc_AP.pkl')

atcdf = pd.read_csv(f'{fddir}/Data/atc_df.csv')    
atcdf = pd.merge(atcdf, atclass[atclass.class_level==2], on='class_id')

atcdf = atcdf.loc[:,['cid','code']]
atcdf.columns = ['CID','ATC1']
atcdf = pd.concat([atcdf, kkdf.loc[:,['CID','ATC1']]])
atcdf= atcdf.drop_duplicates()
atcdf = atcdf[atcdf.CID!='-']
atcdf = atcdf[atcdf.CID.isin(atcap.index)]

with open(f'{cdir}/atc_cds.txt','w') as f:
    f.write('\n'.join([str(cd) for cd in set(atcdf[atcdf.CID!='-'].CID)]))


atcinfo = pd.read_csv(f'{cdir}/atccid_info.csv')
atcinfo = atcinfo.loc[:,[' cid','smiles']]
atcinfo.columns = ['CID','smiles']

atcdf = pd.merge(atcdf, atcinfo, on='CID')

trcids = list(set(atcdf.sample(3300, random_state=42).CID))

tratcdf = atcdf[atcdf.CID.isin(trcids)]

ap1a = pd.read_pickle(f'{cdir}/ap1a.pkl')
# cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
# cvv = [cvv[v] for i,v in enumerate(ap1a.columns)]
# cvv = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + ['##'+v for v in cvv]
# with open(f'{maindir}/SPMM/vocab_ap_368.txt', 'w') as f:
#     f.write('\n'.join(cvv))
tratcdf = tratcdf.sample(frac=1, random_state=42)

atc_ap1a = atcap.copy()
atc_ap1a[atc_ap1a>=1] = 1
atc_ap1a[atc_ap1a<1] = 0

# atc_ap1a = atc_ap1a[atc_ap1a.T.sum()!=0]
ssmm11 = atc_ap1a.sum()
atc_ap1a = atc_ap1a.T[atc_ap1a.T.index.isin(ssmm11[ssmm11>=10].index)].T




cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(atc_ap1a.columns)]
cvv = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + ['##'+v for v in cvv]
with open(f'{cdir}/SPMM/vocab_ap_380.txt', 'w') as f:
    f.write('\n'.join(cvv))



cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(atc_ap1a[atc_ap1a.index.isin(trcids)].columns)]
aptcks = []
for vv in atc_ap1a[atc_ap1a.index.isin(trcids)].values:
    aptcks.append(''.join(np.array(cvv)[vv==1]))


cid2aptck = dict(zip(atc_ap1a[atc_ap1a.index.isin(trcids)].index, aptcks))
train_aptcks = [cid2aptck[c] for c in trcids]

trcid2smi = dict(zip(tratcdf.CID, tratcdf.smiles))
smi2aptck = dict(zip([trcid2smi[cd] for cd in trcids], train_aptcks))


with open(f'{cdir}/SPMM/data/pretrain_3K.txt', 'w') as f:
    f.write('\n'.join([trcid2smi[cd] for cd in trcids]))


with open(f'{cdir}/SPMM/data/smi2aptck_atc.pkl', 'wb') as f:
    pickle.dump(smi2aptck, f)


aptcks_test = []
for vv in atc_ap1a[~atc_ap1a.index.isin(trcids)].values:
    aptcks_test.append(''.join(np.array(cvv)[vv==1]))

cid2aptck_test = dict(zip(atc_ap1a[~atc_ap1a.index.isin(trcids)].index, aptcks_test))
tecids = list(set(atc_ap1a[atc_ap1a.index.isin(atcdf.CID)].index)-set(trcids))
train_aptcks_test = [cid2aptck_test[c] for c in tecids]

all_cid2smi = dict(zip(atcdf.CID, atcdf.smiles))
test_aptcks = [cid2aptck_test[c] for c in tecids]
# smi2aptck = dict(zip([all_cid2smi[cd] for cd in tecids], train_aptcks))

with open(f'{cdir}/SPMM/data/test_300.txt', 'w') as f:
    f.write('\n'.join([all_cid2smi[cd] for cd in tecids]))


aptcks_all = []
for vv in atc_ap1a[atc_ap1a.index.isin(atcdf.CID)].values:
    aptcks_all.append(''.join(np.array(cvv)[vv==1]))



smi2aptck = dict(zip([all_cid2smi[cd] for cd in list(atc_ap1a[atc_ap1a.index.isin(atcdf.CID)].index)], aptcks_all))


with open(f'{cdir}/SPMM/data/smi2aptck_atc.pkl', 'wb') as f:
    pickle.dump(smi2aptck, f)




#######################
# label
#######################

from sklearn.preprocessing import MultiLabelBinarizer

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()


cds = list(set(tratcdf.CID))
tds = []
for cd in cds:
  tds.append(list(tratcdf[tratcdf.CID==cd].ATC1))


cid2tids = pd.DataFrame([cds,tds]).T
cid2tids.columns = ['cid','tids']
# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Transform labels into binary format
binary_labels = mlb.fit_transform(cid2tids["tids"])

c2l = dict(zip(cds, binary_labels))
# with open(f'{maindir}/c2l.pkl', 'wb') as f:
#   pickle.dump(c2l, f)

cid2smi = dict(zip(tratcdf.CID, tratcdf.smiles))
smi2target = dict()
for cd in c2l.keys():
  if cd in cid2smi.keys():
    smi2target[cid2smi[cd]] = [float(a) for a in c2l[cd]]



smi2target = dict()
for cd in c2l.keys():
  if cd in cid2smi.keys():
    smi2target[cid2smi[cd]] = c2l[cd].astype(np.float32)


with open(f'{cdir}/SPMM/smi2target_atc.pkl', 'wb') as f:
  pickle.dump(smi2target2, f)




cvv = [f'{a}{b}' for a in list("abcdefghij") for b in range(10)][0:85]
smi2target2 = dict()
for cd in c2l.keys():
  if cd in cid2smi.keys():
    smi2target2[cid2smi[cd]] = ''.join(np.array(cvv)[c2l[cd]==1])


with open(f'{cdir}/SPMM/smi2target2_atc.pkl', 'wb') as f:
  pickle.dump(smi2target2, f)


cvv = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + ['##'+v for v in cvv]
with open(f'{cdir}/SPMM/vocab_target_89.txt', 'w') as f:
    f.write('\n'.join(cvv))


python d_smiles2pv_types.py --checkpoint './Pretrain/checkpoint_epoch=28.ckpt' --input_file './data/test_20K.txt'
python d_smiles2pv_types.py --checkpoint './Pretrain_ap2property/checkpoint_epoch=28.ckpt' --input_file './data/test_20K.txt' --itype 'ap2pv' --vocab_filename './vocab_ap_368.txt'


python d_smiles2pv_types.py --input_file ./data/test_300.txt --output ./ap2atc_output_51.pkl --itype ap2atc --checkpoint ./Pretrain_ap2target_atc_more_epoch/checkpoint_epoch_51.ckpt --text_config ./config_bert_ap_atc.json --k 2 --vocab_filename ./vocab_ap_380.txt --vocab_filename2 ./vocab_target_89.txt

python d_smiles2pv_types.py --input_file ./data/test_300.txt --output ./smi2atc_output_78.pkl --itype smi2atc --checkpoint ./Pretrain_smi2target_atc_more_epoch/checkpoint_epoch_78.ckpt --k 2 --vocab_filename2 ./vocab_target_89.txt


python d_smiles2pv_types.py --input_file ./data/test_20K.txt --output ./ap2pv_output_65.pkl --itype ap2pv --checkpoint ./Pretrain_ap2target_more_epoch/checkpoint_epoch_65.ckpt --text_config ./config_bert_ap.json --k 2 --vocab_filename ./vocab_ap_368.txt

with open(f'{maindir}/SPMM/ap2pv_pred.pkl', 'rb') as f:
    ap2pv_pred = pickle.load(f)

with open(f'{maindir}/SPMM/smi2pv_pred.pkl', 'rb') as f:
    smi2pv_pred = pickle.load(f)
    

from scipy.stats import spearmanr

cc = [spearmanr(ap2pv_pred[0][i], ap2pv_pred[1][i])[0] for i in range(ap2pv_pred[0].shape[0])]
dd = [spearmanr(smi2pv_pred[0][i], smi2pv_pred[1][i])[0] for i in range(smi2pv_pred[0].shape[0])]

# plt.plot(aa,bb)
# ##

# spearmanr(aa,bb)

import matplotlib.pyplot as plt


# Create scatter plot
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)

plt.scatter(aa, bb, color='coral', alpha=0.7)

# Set x and y axis limits
plt.xlim(0.75, 1.0)
plt.ylim(0.75, 1.0)

# Labels and title
plt.xlabel("AP")
plt.ylabel("SMILES")
plt.title('pearson')
# plt.text(0.86, 0.99, f"Correlation: {spearmanr(cc,bb)[0]:.2f}", fontsize=12)
#plt.title("Scatter Plot of A vs B (0.9 - 1.0 Range)")

# Show plot
plt.subplot(1,2,2)

plt.scatter(cc, dd, color='coral', alpha=0.7)

# Set x and y axis limits
plt.xlim(0.75, 1.0)
plt.ylim(0.75, 1.0)

# Labels and title
plt.xlabel("AP")
plt.ylabel("SMILES")
plt.title('spearman')
# plt.text(0.86, 0.99, f"Correlation: {spearmanr(cc,bb)[0]:.2f}", fontsize=12)
#plt.title("Scatter Plot of A vs B (0.9 - 1.0 Range)")

# Show plot
plt.show()



from sklearn.preprocessing import MultiLabelBinarizer

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()


cds = list(set(atcdf.CID))
tds = []
for cd in cds:
  tds.append(list(atcdf[atcdf.CID==cd].ATC1))


cid2tids2 = pd.DataFrame([cds,tds]).T
cid2tids2.columns = ['cid','tids']
# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Transform labels into binary format
binary_labels2 = mlb.fit_transform(cid2tids2["tids"])

c2l2 = dict(zip(cds, binary_labels2))
# with open(f'{maindir}/c2l.pkl', 'wb') as f:
#   pickle.dump(c2l, f)
cvv = [f'{a}{b}' for a in list("abcdefghij") for b in range(10)][0:85]
smi2target2 = dict()
for cd in c2l2.keys():
  if cd in all_cid2smi.keys():
    smi2target2[all_cid2smi[cd]] = ''.join(np.array(cvv)[c2l2[cd]==1])


export LD_LIBRARY_PATH=/share/apps/anaconda3/envs/DeepPocket/lib:$LD_LIBRARY_PATH

#!/bin/bash

# Number of cores to use
num_cores=30

# Directory containing the PDB files
alphafold_pdb_directory=/spstorage/DB/AlphaFoldDB/9606_Human_v4
file_pattern="*pdb*"
output_pocket_path=/spstorage/USERS/sung/projects/APM_ver2/Pocket


# Function to process each line
process_line() {
    input_pdb_path=$1
    output_pocket_path=/spstorage/USERS/sung/projects/APM_ver2/Pocket

    python /spstorage/DB/DeepPocket/Github/predict.py \
    -p "${input_pdb_path}" \
    -c /spstorage/DB/DeepPocket/Models/classification_models/first_model_coach420_best_test_auc_91001.pth.tar \
    -s /spstorage/DB/DeepPocket/Models/segmentation_models/seg0_best_test_IOU_91.pth.tar -r 10 \
    -o "${output_pocket_path}" 2>/dev/null  # Redirect stderr to /dev/null

    cd ${output_pocket_path}
    dir_name=$(basename "${input_pdb_path}" .pdb)
    mkdir -p "${dir_name}"
    mv "${dir_name}"* "${dir_name}/"
}


export -f process_line

find "${alphafold_pdb_directory}" -type f -name "${file_pattern}" | parallel -j "${num_cores}" process_line

# process_line /spstorage/DB/AlphaFoldDB/9606_Human_v4/AF-Q03001-F15-model_v4.pdb



### 

ccs = ap.sum()
ccsi = list(ccs)

ccssii = np.argsort(ccsi)[-135:-35][::-1]

ap_type = []
apidx = []
for v in np.argsort(ccsi)[::-1]:
    if (v//10 not in ap_type)&(v%10!=0):
        ap_type.append(v//10)
        apidx.append(v)


apidx.sort()

apmap = ap.loc[:,apidx]

cid2apm = dict(zip(apmap.index, apmap.values))

with open(f{maindir}/cid2apm.pkl, 'wb') as f:
    pickle.dump(cid2apm, f)


rmse = [0.58197711448, 0.407901763916][::-1]
r2 = [0.4604787,0.6035627][::-1]

smi_res = [0.407901763916, 0.6035627]
ap_res = [0.58197711448, 0.4604787]

import matplotlib.pyplot as plt
import numpy as np

# RMSE, R2 values for two results
metrics = ["RMSE", "R2"]
smi_res = [0.407901763916, 0.6035627]
ap_res = [0.58197711448, 0.4604787]

x = np.arange(len(metrics))  # x축 위치
width = 0.3  # 막대 너비

# Plot Bar Chart
plt.figure(figsize=(6, 5))
plt.bar(x - width/2, smi_res, width, label="SMILES", color="#003092", alpha=0.7)
plt.bar(x + width/2, ap_res, width, label="AP", color="#D84040", alpha=0.7)

# Labels and Title
plt.ylabel("Values")
plt.xlabel("Metrics")
plt.title("RMSE and R2 Comparison")
plt.xticks(x, metrics)  # X축 레이블 설정
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()

metrics = ["RMSE", "R2"]
sa18 = [0.5851826071739197,  0.5188935625495116]
sa49 = [0.4770876467227936, 0.6628293467491571]

x = np.arange(len(metrics))  # x축 위치
width = 0.3  # 막대 너비

# Plot Bar Chart
plt.figure(figsize=(6, 5))
plt.bar(x - width/2, sa18, width, label="epoch 18", color="#003092", alpha=0.7)
plt.bar(x + width/2, sa49, width, label="epoch 49", color="#D84040", alpha=0.7)

# Labels and Title
plt.ylabel("Values")
plt.xlabel("Metrics")
plt.title("RMSE and R2 Comparison")
plt.xticks(x, metrics)  # X축 레이블 설정
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()


python d_smiles2pv.py --input_file ./data/test_3K.txt --output ./smi2ap_output_9.pkl --itype smi2ap --checkpoint ./Pretrain_smi2ap/checkpoint_epoch_9.ckpt


import glob
smifiles = glob.glob('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/data/*/*csv')

maindir = '/spstorage/USERS/gina/Project/Chemcriptome/SPMM'
with open('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/all_smis.txt', 'w') as f:
    f.write('\n'.join(allsmi))

with open('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/all_smis.txt', 'r') as f:
    allsmi = f.readlines()

as2c= pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/smi2cids.txt',sep='\t',header=None)
left_smis = list(set(allsmi)-set(as2c.dropna().iloc[:,0]))
Chem.MolFromSmiles(left_smis[0])

for i, smi in enumerate(left_smis):
    with open(f"{maindir}/downstream_smis/{i}.smi", 'w') as fout:
        fout.write(smi+'\n')


%%bash
for smi in $(ls *.smi)
do
 echo $smi
 echo "obabel -ismi $smi -osdf -O ./drug_pdbqt/$smi.sdf --gen3d --fast -p 7.4 --canonical"
 obabel -ismi $smi -osdf -O $smi.sdf --gen3d --fast -p 7.4 --canonical
done


#!/bin/bash

for smi in *.smi
do

  if [ -f "$smi.sdf" ]; then
    echo "🚫 Skipping $smi → $sdf_file already exists"
    continue
  fi
  timeout 300s obabel -ismi "$smi" -osdf -O $smi.sdf --gen3d --fast -p 7.4 --canonical

  if [ $? -eq 124 ]; then
    echo "⚠️ Timeout on $smi"
  elif [ $? -ne 0 ]; then
    echo "❌ Failed on $smi"
  else
    echo "✅ Finished $smi"
  fi
done




lltss2[lltss2<1]=0
lltss2[lltss2>=1]=1
ap = lltss2.loc[:,list(ap1a.columns)]

cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(ap.columns)]
aptcks = []
for vv in ap.values:
    aptcks.append(''.join(np.array(cvv)[vv==1]))


cid2aptcks = dict(zip(list(ap.index), aptcks))
s2c = dict(zip(list(as2c[as2c.iloc[:,1].isin(ap.index)].iloc[:,0]), list(as2c[as2c.iloc[:,1].isin(ap.index)].iloc[:,1])))
s2ap = dict(zip(list(as2c[as2c.iloc[:,1].isin(ap.index)].iloc[:,0]), [cid2aptcks[c] for c in list(as2c[as2c.iloc[:,1].isin(ap.index)].iloc[:,1])]))

import glob
smifiles = glob.glob('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/data/*/*csv')


for smif in smifiles:
    smidf = pd.read_csv(smif)
    smic = list(smidf.columns)[0]
    smidf = smidf[smidf.iloc[:,0].isin(list(s2ap))]
    smidf[smic] = [s2ap[c] for c in list(smidf.iloc[:,0])]
    nmm = smif.split('/')
    smidf.to_csv('/'.join(nmm[0:-1]+[f'AP_{nmm[-1]}']), index=False)




iadf =iidf[iidf.tid.isin(adff.tid)]

trdf = pd.concat([adff[adff.cid.isin(ssmidf.iloc[:,0])], iadf[iadf.cid.isin(ssmidf.iloc[:,0])]])
itid_top20 = list(iadf[iadf.cid.isin(ssmidf.iloc[:,0])].value_counts('tid')[0:20].index)



ixdf = iadf[(iadf.tid.isin(itid_top20))]



trixdf  = ixdf[ixdf.cid.isin(ssmidf.iloc[:,0])]
trdf = pd.concat([adff[(adff.cid.isin(ssmidf.iloc[:,0])&(adff.tid.isin(itid_top20)))], trixdf])

trcds = list(set(trdf.cid))


trrdf = trdf[trdf.cid.isin(trcds[0:int(len(trcds)*0.9)])]
valrdf = trdf.drop(trrdf.index)

iixdf = ixdf.drop(trixdf.index)
tedf = pd.concat([adff[(adff.cid.isin(set(adff.cid)-set(ssmidf.iloc[:,0])))&(adff.tid.isin(itid_top20))], iixdf[iixdf.cid.isin(set(adff.cid)-set(ssmidf.iloc[:,0]))]])
trrdf['act'] =['i' if a=='i' else 'a' for a in trrdf.act]
valrdf['act'] =['i' if a=='i' else 'a' for a in valrdf.act]
tedf['act'] =['i' if a=='i' else 'a' for a in tedf.act]




cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(apmap.columns)]

daps = pd.read_pickle(f'{maindir}/SPMM/downstream_APs.pkl')
# daps = daps.loc[:,list(ap1a.columns)]
daps = daps.loc[:,list(apmap.columns)]

cid2ap_sort55 = dict()
for cd in list(daps.index):
    xx=pd.DataFrame({'cvv':cvv,'ap':list(daps.loc[cd,:])}).sort_values('ap',ascending=False)
    xx = xx[xx.ap>0]
    cid2ap_sort55[cd] = ''.join(list(xx.cvv))


dsmi2cid = pd.read_csv(f'{maindir}/SPMM/smi2cids.txt', sep='\t', header=None).dropna()
dsmi2cid.columns = ['smiles','cid']
dsmi2cid = dsmi2cid[dsmi2cid.cid.isin(daps.index)]
dff = glob.glob(f'{maindir}/SPMM/data/*/*csv')
for f in dff:
    if 'AP_' not in f:
        ffs = f.split('/')
        ffs[-1] = 'AP_55_sorting_'+ffs[-1]
        dfv = pd.read_csv(f)
        wnt = dfv.columns
        dfv = pd.merge(dfv, dsmi2cid, left_on= list(dfv.columns)[0], right_on='smiles')
        dfv[list(dfv.columns)[0]] = [cid2ap_sort[cd] for cd in list(dfv.cid)]
        dfv= dfv.loc[:,wnt]
        dfv.to_csv('/'.join(ffs), index=False)

# mlb2 = MultiLabelBinarizer()

# smi2
pd.read_


cid2tids[cid2tids.cid.isin(trrdf[trrdf.act=='a'].cid)]

bidf = pd.DataFrame(binary_labels)
bidf.index = cid2tids.cid

cid2tids['size'] = [len(a) for a in cid2tids.tids]
ssdf = cid2tids[cid2tids['size']==1]
ssdf['tid'] = [a[0] for a in ssdf.tids]
bbidf = bidf.loc[ssdf[~ssdf.tid.duplicated()].cid,:]
bidf.columns =list(pd.DataFrame([(list(bbidf.values[i]).index(1),a) for i,a in enumerate(list(ssdf[~ssdf.tid.duplicated()].tid))]).sort_values(0).iloc[:,1])

bidf = bidf.astype(float)

trbidf = pd.concat([pd.DataFrame([cid2smi[cd] for cd in list(bidf[bidf.index.isin(trrdf[trrdf.act=='a'].cid)].index)]),bidf[bidf.index.isin(trrdf[trrdf.act=='a'].cid)].reset_index()], axis=1)
trbidf = trbidf.drop('cid', axis=1)
trbidf.columns = ['smiles'] + list(bidf.columns)
trbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_train.csv', index=False)
valbidf = pd.concat([pd.DataFrame([cid2smi[cd] for cd in list(bidf[bidf.index.isin(valrdf[valrdf.act=='a'].cid)].index)]),bidf[bidf.index.isin(valrdf[valrdf.act=='a'].cid)].reset_index()], axis=1)
valbidf = valbidf.drop('cid', axis=1)
valbidf.columns = ['smiles'] + list(bidf.columns)
valbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_valid.csv', index=False)

tbidf = pd.concat([pd.DataFrame([cid2smi[cd] for cd in list(bidf[bidf.index.isin(tedf[tedf.act=='a'].cid)].index)]),bidf[bidf.index.isin(tedf[tedf.act=='a'].cid)].reset_index()], axis=1)
tbidf = tbidf.drop('cid', axis=1)
tbidf.columns = ['smiles'] + list(bidf.columns)
tbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_test.csv', index=False)

trbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_train_top20.csv', index=False)
valbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_valid_top20.csv', index=False)
tbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_test_top20.csv', index=False)

tr10 = list(trrdf.value_counts('tid').index)[8:18]

trbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_train_10.csv', index=False)
valbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_valid_10.csv', index=False)
tbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_test_10.csv', index=False)



cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(ap1a.columns)]
aptcks2 = []
for vv in ap1a.values:
    aptcks2.append(''.join(np.array(cvv)[vv==1]))


cid2aptck2 = dict(zip(ap1a.index, aptcks2))
trbidf = pd.concat([pd.DataFrame([cid2aptck2[cd] for cd in list(bidf[bidf.index.isin(trrdf[trrdf.act=='a'].cid)].index)]),bidf[bidf.index.isin(trrdf[trrdf.act=='a'].cid)].reset_index()], axis=1)
trbidf = trbidf.drop('cid', axis=1)
trbidf.columns = ['smiles'] + list(bidf.columns)
trbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_train.csv', index=False)
valbidf = pd.concat([pd.DataFrame([cid2aptck2[cd] for cd in list(bidf[bidf.index.isin(valrdf[valrdf.act=='a'].cid)].index)]),bidf[bidf.index.isin(valrdf[valrdf.act=='a'].cid)].reset_index()], axis=1)
valbidf = valbidf.drop('cid', axis=1)
valbidf.columns = ['smiles'] + list(bidf.columns)
valbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_valid.csv', index=False)


tt = bidf[bidf.index.isin(tedf[tedf.act=='a'].cid)]
tt = tt[tt.index.isin(ap1a.index)]
tbidf = pd.concat([pd.DataFrame([cid2aptck2[cd] for cd in tt.index]),tt.reset_index()], axis=1)
tbidf = tbidf.drop('cid', axis=1)
tbidf.columns = ['smiles'] + list(bidf.columns)
tbidf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_test.csv', index=False)

trbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_train_top20.csv', index=False)
valbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_valid_top20.csv', index=False)
tbidf.loc[:,['smiles']+itid_top20].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_test_top20.csv', index=False)

tr10 = list(trrdf.value_counts('tid').index)[8:18]

trbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_train_10.csv', index=False)
valbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_valid_10.csv', index=False)
tbidf.loc[:,['smiles']+tr10].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_test_10.csv', index=False)

for tid in itid_top20:
    for i in range(3):
        sdf = [trrdf, valrdf, tedf][i]
        tsk = ['train','valid','test'][i]
        tdf = sdf[sdf.tid==tid]
        tdf['label'] = [1.0 if a=='a' else 0.0 for a in tdf.act]
        tdf = tdf.drop('act', axis=1)
        tdf['smiles'] = [cid2smi[cd] for cd in tdf.cid]
        tdf['ap'] = [cid2aptck2[cd] if cd in cid2aptck2 else '-' for cd in tdf.cid]
        tdf.loc[:,['smiles','label']].to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_{tid}_{tsk}.csv', index=False)
        tdf = tdf[tdf.ap!='-']
        tdf = tdf.loc[:,['ap','label']]
        tdf.columns = ['smiles','label']
        tdf.to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/AP_TARGET_{tid}_{tsk}.csv', index=False)



use_feats = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'Halogen', 'Aromatic', 'Hydrophobe', 'DA', 'Hydrophilic', 'etc']
    
    
    
    .to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/TARGET_train_{tid}.csv', index=False)


bidf[bidf.index.isin(valrdf[valrdf.act=='a'].cid)]

bidf[bidf.index.isin(tedf[tedf.act=='a'].cid)]

bidf20 = bidf.loc[:,itid_top20]

# # cds = list(set(trrdf[trrdf.act=='a'].cid))
# # tds = []
# # for cd in cds:
# #   tds.append(list(trrdf[trrdf.act=='a'][trrdf[trrdf.act=='a'].cid==cd].tid))


# cid2tidsx = pd.DataFrame([cds,tds]).T
# cid2tidsx.columns = ['cid','tids']

# # Transform labels into binary format




# cds2 = list(set(valrdf[valrdf.act=='a'].cid))
# tds2 = []
# for cd in cds2:
#   tds2.append(list(valrdf[valrdf.act=='a'][valrdf[valrdf.act=='a'].cid==cd].tid))


# cid2tidsv = pd.DataFrame([cds2,tds2]).T
# cid2tidsv.columns = ['cid','tids']
# val_labels = mlb2.transform(cid2tidsv["tids"])



# cds3 = list(set(tedf[tedf.act=='a'].cid))
# tds3 = []
# for cd in cds3:
#   tds3.append(list(tedf[tedf.act=='a'][tedf[tedf.act=='a'].cid==cd].tid))


# cid2tidst = pd.DataFrame([cds3,tds3]).T
# cid2tidst.columns = ['cid','tids']

# te_labels = mlb2.transform(cid2tidst["tids"])

class smi2ap_Dataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()
        self.data = [l.strip() for l in lines]
        with open('./normalize_ap.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm
        with open('./smi2apm.pkl', 'rb') as w:
            self.s2a = pickle.load(w)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        properties = (torch.tensor(self.s2a[self.data[index]]) - self.property_mean) / self.property_std

        return properties.to(dtype=torch.float), '[CLS]' + smiles



ccssii = np.argsort(ccsi)[-100:][::-1]

ccs = ap.sum()

ccsi = list(ccs)



ap_type = []
apidx = []
for v in np.argsort(ccsi)[::-1]:
    if (v//10 not in ap_type)&(v%10!=0):
        ap_type.append(v//10)
        apidx.append(v)

apidx.sort()

apmap = ap.loc[:,apidx]


smi2apm = dict(zip([cid2smi[cd] for cd in apmap[apmap.index.isin(cid2smi)].index],apmap[apmap.index.isin(cid2smi)].values))


with open(f'{maindir}/SPMM/normalize_ap.pkl', 'wb') as f:
    pickle.dump(torch.tensor(list(apmap.mean())), torch.tensor(list(apmap.std())), f)

with open(f'{maindir}/SPMM/smi2apm.pkl', 'wb') as f:
    pickle.dump(smi2apm, f)

ap_type2 = []
apidx2 = []
for v in np.argsort(ccsi)[::-1]:
    if (v//10 not in ap_type2)&(v%10!=0):
        ap_type2.append(v//10)
        apidx2.append(v)
        apidx2.append(v+1)

apidx2.sort()

apmap2 = ap.loc[:,apidx2]


smi2apm2 = dict(zip([cid2smi[cd] for cd in apmap2[apmap2.index.isin(cid2smi)].index],apmap2[apmap2.index.isin(cid2smi)].values))

smi2apm3 = smiapm.copy()


apmap3 = pd.concat([apmap,iap]).drop_duplicates()

smi2apm3 = dict(zip([cid2smi2[cd] for cd in apmap3[apmap3.index.isin(cid2smi2)].index],apmap3[apmap3.index.isin(cid2smi2)].values))

dd2=dict(zip(list(iap.index), iap.values.tolist()))

for k,v in dd2.items():
    if k not in smi2apm.keys():
        smi2apm3[cid2smi2[k]] = v


with open(f'{maindir}/SPMM/normalize_ap2.pkl', 'wb') as f:
    pickle.dump(tuple([torch.tensor(list(apmap2.mean())), torch.tensor(list(apmap2.std()))]), f)


with open(f'{maindir}/SPMM/normalize_ap2.pkl', 'rb') as f:
    norm = pickle.load(f)


with open(f'{maindir}/SPMM/smi2apm2.pkl', 'wb') as f:
    pickle.dump(smi2apm2, f)


with open(f'{maindir}/SPMM/smi2apm3.pkl', 'wb') as f:
    pickle.dump(smi2apm3, f)




with open(f'{maindir}/SPMM/ap2pv_pred.pkl', 'rb') as f:
    ap2pv_pred = pickle.load(f)

with open(f'{maindir}/SPMM/smi2pv_pred.pkl', 'rb') as f:
    smi2pv_pred = pickle.load(f)
    


with open(f'{maindir}/SPMM/smi2ap_output_18.pkl', 'rb') as f:
    smi2ap_pred_18 = pickle.load(f)


smi2ap_pred_18 = [torch.stack(list(smi2ap_pred_18.keys())), torch.stack(list(smi2ap_pred_18.values()))]
smi2ap_pred_49 = [torch.stack(list(smi2ap_pred_49.keys())), torch.stack(list(smi2ap_pred_49.values()))]



with open(f'{maindir}/SPMM/smi2ap_output_49.pkl', 'rb') as f:
    smi2ap_pred_49 = pickle.load(f)
    

with open(f'{maindir}/SPMM/normalize_ap.pkl', 'rb') as w:
    norm = pickle.load(w)


with open(f'{maindir}/SPMM/normalize.pkl', 'rb') as w:
    norm = pickle.load(w)




rn = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'Halogen', 'Aromatic', 'Hydrophobe', 'DA', 'Hydrophilic', 'etc']
# rns = [str(x)+str(y) for x,y in combinations_with_replacement(rn,2)]
rns = list(itertools.combinations_with_replacement(rn, 2))
# Sample values (Replace with your actual list B)
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "white", "blue"])


plt.figure(figsize=(20, 10))
xy = ['smi2ap_18', 'smi2ap_49']

for ii,xx in enumerate([smi2ap_pred_18, smi2ap_pred_49]):
    ref, cand = xx
    mean, std = norm
    mse = []
    n_mse = []
    rs, cs = [], []
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        mse.append((r - c) ** 2)
        n_mse.append((ref[i] - cand[i]) ** 2)
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0)).squeeze()
    n_mse = torch.stack(n_mse, dim=0)
    n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    print('mean of 53 properties\' normalized RMSE:', n_rmse.mean().item())
    rs = torch.stack(rs)
    cs = torch.stack(cs).squeeze()
    r2 = []
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:, i], cs[:, i]))
    r2 = np.array(r2)
    print('mean r^2 coefficient of determination:', r2.mean().item())
    # List A (feature names)
    # B =r2.tolist()
    B = rmse.tolist()
    # Create an empty DataFrame (size = len(A) x len(A))
    df = pd.DataFrame(index=rn, columns=rn, dtype=float)
    # Fill DataFrame with values from B
    for (i, j), value in zip(rns, B):
        df.loc[i, j] = value
        df.loc[j, i] = value  # Ensure symmetry
    df[df<0] = 0
    # Create mask to show only lower triangle
    # mask = np.triu(np.ones_like(df, dtype=bool))
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)  # k=1 keeps diagonal visible
    # Plot heatmap
    plt.subplot(1,2,ii+1)
    sns.heatmap(df, mask=mask, cmap=custom_cmap, annot=True, linewidths=0.5, square=True, fmt=".2f", cbar=True)
    plt.title(f"{xy[ii]} R2")


plt.savefig(f'{maindir}/SPMM/figs/SMILES_AP_rmse.png')




plt.figure(figsize=(10, 20))
xy = ['smi2pv', 'ap2pv_pred']

for ii,xx in enumerate([smi2pv_pred, ap2pv_pred]):
    ref, cand = xx
    mean, std = norm
    mse = []
    n_mse = []
    rs, cs = [], []
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        mse.append((r - c) ** 2)
        n_mse.append((ref[i] - cand[i]) ** 2)
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0)).squeeze()
    n_mse = torch.stack(n_mse, dim=0)
    n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    print('mean of 53 properties\' normalized RMSE:', n_rmse.mean().item())
    rs = torch.stack(rs)
    cs = torch.stack(cs).squeeze()
    r2 = []
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:, i], cs[:, i]))
    r2 = np.array(r2)
    print('mean r^2 coefficient of determination:', r2.mean().item())
    # List A (feature names)
    B =r2.tolist()
    # B = n_rmse.tolist()
    # Create an empty DataFrame (size = len(A) x len(A))
    df = pd.DataFrame(B, dtype=float)
    df.index = names
    # Fill DataFrame with values from B
    df[df<0] = 0
    df[df>2] =2
    # Create mask to show only lower triangle
    # Plot heatmap
    plt.subplot(1,2,ii+1)
    sns.heatmap(df, cmap='coolwarm', annot=True, linewidths=0.5, square=True, fmt=".2f", cbar=True)
    plt.title(f"{xy[ii]} R2")


plt.savefig(f'{maindir}/SPMM/figs/X2pv_r2.png')

pd.read_csv(f'{maindir}/SPMM/data/test_20K.txt',header=None)


dataset = smi2ap_Dataset_pretrain('./data/test_20K.txt', data_length=[0, 300000])


tokenizer = BertTokenizer(vocab_file=f'./vocab_bpe_300.txt', do_lower_case=False, do_basic_tokenize=False)
tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

df = pd.read_csv('./data/test_20K.txt',header=None)
smis = list(df.iloc[:,0])

tkns = []
for smi in smis:
    text_input = tokenizer('[CLS]'+smi, padding='longest', return_tensors="pt")['input_ids'][0][2:-1].tolist()
    tkns.append(text_input)



python SPMM_smi2ap.py --data_path ./data/pretrain_300K.txt


python d_smiles2pv.py --input_file ./data/test_20K.txt --output ./smi2ap_output_18.pkl --itype smi2ap --checkpoint ./Pretrain_smi2ap/checkpoint_epoch_18.ckpt --npoint 55
python d_smiles2pv.py --input_file ./data/test_20K.txt --output ./smi2ap_output_49.pkl --itype smi2ap --checkpoint ./Pretrain_smi2ap/checkpoint_epoch_49.ckpt --npoint 55 --device 'cuda:1'


with open(f'{maindir}/SPMM/test_smi_tkns.pkl', 'wb') as f:
    pickle.dump(iis, f)

with open(f'{maindir}/SPMM/test_smi_tkns.pkl', 'rb') as f:
    smitkn = pickle.load(f)



# Create a DataFrame with zeros
df = pd.DataFrame(0, index=range(len(smitkn)), columns=list(range(5,300)))

# Fill 1s where numbers appear in data
for i, row in enumerate(smitkn):
    df.loc[i, row] = 1


with open(f'{maindir}/SPMM/normalize_ap.pkl', 'rb') as w:
    norm2 = pickle.load(w)


with open(f'{maindir}/SPMM/normalize.pkl', 'rb') as w:
    norm1 = pickle.load(w)




for ii,xx in enumerate([smi2pv_pred, smi2ap_pred_18, smi2ap_pred_49]):
    ref, cand = xx
    mean, std = [norm1, norm2, norm2][ii]
    mse = []
    n_rmse = []
    rs, cs = [], []
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        if ii==0:
            n_rmse.append(torch.sqrt((ref[i] - cand[i]) ** 2).tolist())
        else:
            n_rmse.append(torch.sqrt((ref[i]- cand[i]) ** 2).tolist()[0])
    df = pd.concat([df, pd.DataFrame(n_rmse)],axis=1)


pair_rsme = []
for i in range(295):
    ddf = df[df.iloc[:,i]==1]
    for j in range(295, 458):
        pair_rsme.append([i, j, np.mean(ddf.iloc[:,j])])
        


pnx= pn_rmse[pn_rmse.iloc[:,1]!=339].dropna()

tok2pred = {}
pnx = pnx.sort_values(2)
for i in range(100):
    tpp = pnx.iloc[i,1]
    if tpp<348: tppn = names[tpp- 295]
    elif tpp>403: tppn = rns[tpp- 403]
    if tok[pnx.iloc[i,0]] in tok2pred.keys():
        # elif tpp<403: tppn = tok[403- 295]
        # else: tppn = tok[458- 295]
        tok2pred[tok[pnx.iloc[i,0]]].append(tppn)
    else:
        tok2pred[tok[pnx.iloc[i,0]]] = [tppn]



testsmis = list(pd.read_csv(f'{maindir}/SPMM/data/test_20K.txt', header=None).iloc[:,0])
testapm = [smi2apm[smi] for smi in testsmis]
cvv2rns = dict(zip(cvv, [rns[a//10] for a in list(ap1a.columns)]))


# Create a DataFrame with zeros
apdf = pd.DataFrame(0, index=range(len(testapm)), columns=cvv)

# Fill 1s where numbers appear in data
for i, row in enumerate(testapm):
    for j in range(0, len(row), 2):
        apdf.iloc[i,cvv.index(row[j:j+2])] = 1



apdf2 = pd.DataFrame(0, index=range(len(testapm)), columns=list(range(55)))
# Fill 1s where numbers appear in data
for i, row in enumerate(testapm):
    for j in range(0, len(row), 2):
        apdf2.iloc[i,rns.index(cvv2rns[row[j:j+2]])] = 1




ref, cand = ap2pv_pred
mean, std = norm1
mse = []
n_mse = []
rs, cs = [], []
for i in range(len(ref)):
    r = (ref[i] * std) + mean
    c = (cand[i] * std) + mean
    rs.append(r)
    cs.append(c)


rs = torch.stack(rs)
cs = torch.stack(cs).squeeze()
r2s = [[r2_score(rs[list(apdf2[apdf2.iloc[:,i]==1].index),j],cs[list(apdf2[apdf2.iloc[:,i]==1].index),j]) for j in range(53)] for i in range(55)]


r2 = []
for i in range(rs.size(1)):
    r2.append(r2_score(rs[:, i], cs[:, i]))
r2 = np.array(r2)

# n_mse = torch.stack(n_mse)

# n_rmses = [torch.sqrt(torch.mean(n_mse[list(apdf2[apdf2.iloc[:,i]==1].index),:], dim=0)) for i in range(55)]


# xxx = [[cvv2rns[a[i:i+1]] for i in range(0,len(a),2)] for a in testapm]





295+53 = 348
295+53+55=403
(Cl)cc = Acceptor Halogen / ('Acceptor', 'Aromatic') / ('Aromatic', 'DA') / ('Donor', 'DA')

'##(=O)N' = ('NegIonizable', 'etc')
'##[N+](=O)[O-])' = ('Hydrophobe', 'DA') ('NegIonizable', 'Hydrophilic')
2 = ('etc', 'etc') ('NegIonizable', 'etc')
(C)C = ('Acceptor', 'DA')

cs = ('etc', 'etc')


python d_smiles2pv.py --input_file ./data/test_20K.txt --output ./smi2ap2_output_20.pkl --itype smi2ap --checkpoint ./Pretrain_smi2ap2/checkpoint_epoch_20.ckpt --npoint 110 --device 'cuda:2'


all_int = pd.read_csv(DIR_AP+'/final2/all_int.csv',index_col=0)
cdlen = all_int.value_counts('cid')
itids = iidf[iidf.tid.isin(adff.tid)].value_counts('tid')

iapdf =pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/iap.pkl')
iap = iapdf.loc[:,apmap.columns]

iiidf = iidf[iidf.cid.isin(iap.index)]
adfff = adff[adff.cid.isin(apmap3.index)]

aitids = list(itids[(itids>1000)&(itids<1600)].index)

random.seed(42)
aitids = [11130, 103659, 10206, 10977, 11638, 78, 20053, 165, 101216, 101097]
dfdf = pd.DataFrame()
for td in aitids:
    sadfc = list(adfff[adfff.tid==td].cid)
    apairs = [random.sample(sadfc, 2) for _ in range(20000)]
    sadf = pd.DataFrame(apairs)
    sadf['label'] = 1
    sidfc = list(iiidf[iiidf.tid==tid].cid)
    ipairs = [(random.choice(sadfc), random.choice(sidfc)) for _ in range(20000)]
    saidf = pd.DataFrame(ipairs)
    saidf['label'] = 0
    asdf = pd.concat([sadf, saidf])
    dfdf = pd.concat([dfdf, asdf])



dfdf = dfdf.drop_duplicates()

asmi = [cid2smi[cd] for cd in list(dfdf.iloc[:,0])]
ismi = [cid2smi2[cd] for cd in list(dfdf.iloc[:,1])]
dfdf2 = pd.DataFrame({'compound1':asmi, 'compound2':ismi, 'label':list(dfdf.label)})

dfdf2.sample(frac=1, random_state=42).to_csv(f'{maindir}/SPMM/for_CL_smiles.txt', index=False)

apairs = random.sample(list(combinations(sadfc, 2)), 100)
apairs = [random.sample(sadfc,2) for i in range(100)]

import random
from itertools import combinations

import glob
import pandas as pd


maindir = '/spstorage/USERS/gina/Project/Chemcriptome/SPMM'
ress = glob.glob(f'{maindir}/downstream_result/*txt')

a = ['bbbp','clearance', 'clintox']
b= ['esol','freesolv', 'lipo', 'bace']
c= ['lidi', 'sider']
d = ['target20','target10','bacec']

aa = a+b+c+d
t = ['c','r','m','r','r','r','r','c','m','m','m','c']
ci = [aa[i] for i,x in enumerate(t) if x=='c']
mi = [aa[i] for i,x in enumerate(t) if x=='m']
ri = [aa[i] for i,x in enumerate(t) if x=='r']


resdfs = [pd.read_csv(f'{maindir}/downstream_result/{f}.txt', sep=' ',header=None, nrows=5).drop_duplicates() for f in a] + [pd.read_csv(f'{maindir}/downstream_result/{f}.txt', sep=' ',header=None, skiprows=5) for f in b] +[pd.read_csv(f'{maindir}/downstream_result/{f}.txt', sep=' ',header=None, skiprows=4) for f in c]+[pd.read_csv(f'{maindir}/downstream_result/{f}.txt', sep=' ',header=None) for f in d]

bbbp 5 clearnace 5 clintox 중복 제거

esol skip 5 freesolv skip 5 lidi skip 4 lipo skip 5 sider skip 4

m2n = {'./Pretrain/checkpoint_epoch=28.ckpt':'300K_smi2property', './Pretrain_smi2ap/checkpoint_epoch_18.ckpt': '300K_smi2ap_18', './Pretrain_smi2ap/checkpoint_epoch_49.ckpt': '300K_smi2ap_49', './Pretrain_original/checkpoint_SPMM.ckpt': '50M_smi2property','./Pretrain_ap2property/checkpoint_epoch=28.ckpt': '300K_ap2property'}

cresdfs = [resdfs[i] for i,x in enumerate(t) if x=='c']
mresdfs = [resdfs[i] for i,x in enumerate(t) if x=='m']
rresdfs = [resdfs[i] for i,x in enumerate(t) if x=='r']

onms = ['300K_ap2property', '300K_smi2property', '50M_smi2property', '300K_smi2ap_18','300K_smi2ap_49']

cresres = []
for i,resdf in enumerate(cresdfs):
    mn = [m2n[dd] for dd in list(resdf.iloc[:,0])]
    res = list(resdf.iloc[:,1])
    mn2res = dict(zip(mn, res))
    cresres.append([mn2res[nm] if nm in mn2res else 0 for nm in onms])


import matplotlib.pyplot as plt
import numpy as np
iis = [ri[1:], ci,mi]


fig, ax = plt.subplots(figsize=(20, 6))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

data_sets = [rresres[1:], cresres, mresres]  # 세 개의 데이터셋
labels = ['300K_ap2property', '300K_smi2property', '50M_smi2property', '300K_smi2ap_18', '300K_smi2ap_49']
cnn = ['Regression','Classification','Multi-Classification']
iis = [tasks1, tasks2, tasks3]  # 세 개의 task 리스트 (정의 필요)

ylabv = ['RMSE','AUROC','AUROC']
for iidx, (ax, resres) in enumerate(zip(axes, data_sets)):
    A = pd.DataFrame(resres).T.values  # 데이터를 DataFrame에서 변환
    tasks = iis[iidx]  # 각 모델의 task 리스트
    x = np.arange(len(tasks))  # X축 위치
    width = 0.15  # 막대 너비
    for i, label in enumerate(labels):
        ax.bar(x + i * width, A[i], width, label=label)
    # 레이블 설정
    ax.set_xticks(x + width * (len(labels) / 2 - 0.5))
    ax.set_xticklabels(tasks)
    ax.set_xlabel('Tasks')
    ax.set_ylabel(ylabv[iidx])
    ax.set_title(f'{cnn[iidx]} Performance')
    ax.legend(loc='lower left')

plt.tight_layout()
plt.show()



    
rresres = []
for i,resdf in enumerate(rresdfs):
    mn = [m2n[dd] for dd in list(resdf.iloc[:,0])]
    if i in [1,2,3,4]:
        res = list(resdf.iloc[:,3])
    else:
        res = list(resdf.iloc[:,1])
    mn2res = dict(zip(mn, res))
    rresres.append([mn2res[nm] if nm in mn2res else 0 for nm in onms])

  
mresres = []
for i,resdf in enumerate(mresdfs):
    mn = [m2n[dd] for dd in list(resdf.iloc[:,0])]
    if i in [2,3]:
        res = list(resdf.iloc[:,4])
    else:
        res = list(resdf.iloc[:,1])
    mn2res = dict(zip(mn, res))
    mresres.append([mn2res[nm] if nm in mn2res else 0 for nm in onms])


cid2smi2= cid2smi.copy()
for cd in list(set(iixx[' cid'])):
    if cd not in cid2smi2:
        cid2smi2[cd] = list(iixx[iixx[' cid']==cd]['smiles'])[0]
        

# esol.txt skip 5

# bace.txt skip 7

# freesolv.txt

# clearance.txt

# lipo.txt



# bacec.txt

# bbbp.txt

# lidi.txt

# target20

# target10




# clintox.txt 2 

# sider.txt 27 skip 4



class CCTM_SS(pl.LightningModule):
    def __init__(self, tokenizer=None, tokenizer2 =None, config=None, loader_len=0, no_train=False):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.training_step_outputs = []

        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width = text_width

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width * 2, 2)

        # self.property_embed = nn.Linear(1, property_width)
        self.property_encoder = BertForMaskedLM(config=bert_config)
        # self.property_mtr_head = nn.Sequential(nn.Linear(property_width, property_width),
        #                                        nn.GELU(),
        #                                        nn.LayerNorm(property_width, bert_config.layer_norm_eps),
        #                                        nn.Linear(property_width, 1))
        # self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        # self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # unk token for PV

        # create momentum models
        self.property_encoder_m = BertForMaskedLM(config=bert_config2)
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        for p in self.property_encoder_m.parameters():  p.requires_grad = False
        for p in self.property_proj_m.parameters():     p.requires_grad = False
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False

        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.mlm_probability = config['mlm_probability']
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, prop_input_ids, prop_attention_mask, text_input_ids, text_attention_mask, cl_labels, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)

        text_embeds = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        prop_embeds = self.property_encoder.bert(prop_input_ids, attention_mask=prop_attention_mask, return_dict=True, mode='text').last_hidden_state
        prop_feat = F.normalize(self.property_proj(prop_embeds[:, 0, :]), dim=-1)
        # get momentum features

        with torch.no_grad():
            self._momentum_update()
            # prop_embeds_m = self.property_encoder_m(inputs_embeds=properties, return_dict=True).last_hidden_state
            prop_embeds_m = self.property_encoder_m.bert(prop_input_ids, attention_mask=prop_attention_mask, return_dict=True, mode='text').last_hidden_state
            prop_feat_m = F.normalize(self.property_proj_m(prop_embeds_m[:, 0, :]), dim=-1)
            # prop_feat_all = torch.cat([prop_feat_m.t(), self.prop_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            # text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = prop_feat_m @ text_feat.T / self.temp
            # sim_t2i_m = text_feat_m @ prop_feat_all / self.temp
            # sim_i2i_m = prop_feat_m @ prop_feat_all / self.temp
            # sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_targets = cl_labels

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            # sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = prop_feat @ text_feat.T / self.temp
        # sim_t2i = text_feat @ prop_feat_all / self.temp
        # sim_i2i = prop_feat @ prop_feat_all / self.temp
        # sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        # loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        # loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        # loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 2
        loss_ita = loss_i2t
        if torch.isnan(sim_i2t).any() or torch.isnan(sim_t2i).any() or torch.isnan(loss_ita):
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        # ================ ITM ================= #
        # forward the positve image-text pair
        pos_pos_prop = self.text_encoder.bert(encoder_embeds=prop_embeds,
                                              attention_mask=prop_attention_mask,
                                              encoder_hidden_states=text_embeds,
                                              encoder_attention_mask=text_attention_mask,
                                              return_dict=True,
                                              mode='fusion',
                                              ).last_hidden_state[:, 0, :]
        pos_pos_text_full = self.text_encoder.bert(encoder_embeds=text_embeds,
                                                   attention_mask=text_attention_mask,
                                                   encoder_hidden_states=prop_embeds,
                                                   encoder_attention_mask=prop_attention_mask,
                                                   return_dict=True,
                                                   mode='fusion',
                                                   ).last_hidden_state
        pos_pos_text = pos_pos_text_full[:, 0, :]
        pos_pos = torch.cat([pos_pos_prop, pos_pos_text], dim=-1)
        vl_output = self.itm_head(pos_pos)
        loss_itm = F.cross_entropy(vl_output, cl_labels)

        self._dequeue_and_enqueue(prop_feat_m, text_feat_m)

        # ================= MLM ================= #
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text_attention_mask,
                                           encoder_hidden_states=prop_embeds_m,
                                           encoder_attention_mask=prop_attention_mask,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, :-1, :]

        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=prop_embeds,
                                       encoder_attention_mask=prop_attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text

        # ================= MPM ================= #
        input_ids = prop_input_ids.clone()
        labels = input_ids.clone()[:, 1:]

        with torch.no_grad():
            logits_m = self.property_encoder_m(input_ids,
                                           attention_mask=prop_attention_mask,
                                           encoder_hidden_states=text_embeds_m,
                                           encoder_attention_mask=text_attention_mask,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, :-1, :]

        mpm_output = self.property_encoder(input_ids,
                                       attention_mask=prop_attention_mask,
                                       encoder_hidden_states=text_embeds,
                                       encoder_attention_mask=text_attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mpm = loss_fct(mpm_output.permute((0, 2, 1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mpm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mpm = (1 - alpha) * loss_mpm + alpha * loss_distill_text

        return loss_mlm, loss_mpm, loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, text_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask_pv(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        arg_sche = AttrDict(self.config['schedular'])
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print('qqq', metric)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        prop, text, labels = train_batch
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(labels.device)
        prop_input = self.tokenizer2(prop, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(labels.device)
        # print(text_input.input_ids[:4], prop[:4], text_input.input_ids.shape)
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop_input.input_ids[:, 1:], prop_input.attention_mask[:, 1:], text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], labels, alpha=alpha)
        loss = loss_mlm + loss_mpm + loss_ita + loss_itm
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss_mlm, prog_bar=True)
            self.log('loss_mpm', loss_mpm, prog_bar=True)
            self.log('loss_ita', loss_ita, prog_bar=True)
            self.log('loss_itm', loss_itm, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0:
            scheduler.step(self.current_epoch + self.warmup_steps)
        else:
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        self.training_step_outputs.append(torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm]))
        return torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm])

    def on_train_epoch_end(self):    # outputs: collection of returns from 'training_step'
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}, {tmp[1]:.4f}, {tmp[2]:.4f}, {tmp[3]:.4f}')
        self.training_step_outputs.clear()




class CCTM_AA(pl.LightningModule):
    def __init__(self, tokenizer=None, tokenizer2 =None, config=None, loader_len=0, no_train=False):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.training_step_outputs = []

        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width = text_width

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width * 2, 2)

        # self.property_embed = nn.Linear(1, property_width)
        self.property_encoder = BertForMaskedLM(config=bert_config)
        # self.property_mtr_head = nn.Sequential(nn.Linear(property_width, property_width),
        #                                        nn.GELU(),
        #                                        nn.LayerNorm(property_width, bert_config.layer_norm_eps),
        #                                        nn.Linear(property_width, 1))
        # self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        # self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # unk token for PV

        # create momentum models
        self.property_encoder_m = BertForMaskedLM(config=bert_config2)
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        for p in self.property_encoder_m.parameters():  p.requires_grad = False
        for p in self.property_proj_m.parameters():     p.requires_grad = False
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False

        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.mlm_probability = config['mlm_probability']
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, prop_input_ids, prop_attention_mask, text_input_ids, text_attention_mask, cl_labels, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)

        text_embeds = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        prop_embeds = self.property_encoder.bert(prop_input_ids, attention_mask=prop_attention_mask, return_dict=True, mode='text').last_hidden_state
        prop_feat = F.normalize(self.property_proj(prop_embeds[:, 0, :]), dim=-1)
        # get momentum features

        with torch.no_grad():
            self._momentum_update()
            # prop_embeds_m = self.property_encoder_m(inputs_embeds=properties, return_dict=True).last_hidden_state
            prop_embeds_m = self.property_encoder_m.bert(prop_input_ids, attention_mask=prop_attention_mask, return_dict=True, mode='text').last_hidden_state
            prop_feat_m = F.normalize(self.property_proj_m(prop_embeds_m[:, 0, :]), dim=-1)
            # prop_feat_all = torch.cat([prop_feat_m.t(), self.prop_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            # text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = prop_feat_m @ text_feat.T / self.temp
            # sim_t2i_m = text_feat_m @ prop_feat_all / self.temp
            # sim_i2i_m = prop_feat_m @ prop_feat_all / self.temp
            # sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_targets = cl_labels

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            # sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = prop_feat @ text_feat.T / self.temp
        # sim_t2i = text_feat @ prop_feat_all / self.temp
        # sim_i2i = prop_feat @ prop_feat_all / self.temp
        # sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        # loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        # loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        # loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 2
        loss_ita = loss_i2t
        if torch.isnan(sim_i2t).any() or torch.isnan(sim_t2i).any() or torch.isnan(loss_ita):
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        # ================ ITM ================= #
        # forward the positve image-text pair
        pos_pos_prop = self.text_encoder.bert(encoder_embeds=prop_embeds,
                                              attention_mask=prop_attention_mask,
                                              encoder_hidden_states=text_embeds,
                                              encoder_attention_mask=text_attention_mask,
                                              return_dict=True,
                                              mode='fusion',
                                              ).last_hidden_state[:, 0, :]
        pos_pos_text_full = self.text_encoder.bert(encoder_embeds=text_embeds,
                                                   attention_mask=text_attention_mask,
                                                   encoder_hidden_states=prop_embeds,
                                                   encoder_attention_mask=prop_attention_mask,
                                                   return_dict=True,
                                                   mode='fusion',
                                                   ).last_hidden_state
        pos_pos_text = pos_pos_text_full[:, 0, :]
        pos_pos = torch.cat([pos_pos_prop, pos_pos_text], dim=-1)
        vl_output = self.itm_head(pos_pos)
        loss_itm = F.cross_entropy(vl_output, cl_labels)

        self._dequeue_and_enqueue(prop_feat_m, text_feat_m)

        # ================= MLM ================= #
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text_attention_mask,
                                           encoder_hidden_states=prop_embeds_m,
                                           encoder_attention_mask=prop_attention_mask,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, :-1, :]

        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=prop_embeds,
                                       encoder_attention_mask=prop_attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text

        # ================= MPM ================= #
        input_ids = prop_input_ids.clone()
        labels = input_ids.clone()[:, 1:]

        with torch.no_grad():
            logits_m = self.property_encoder_m(input_ids,
                                           attention_mask=prop_attention_mask,
                                           encoder_hidden_states=text_embeds_m,
                                           encoder_attention_mask=text_attention_mask,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, :-1, :]

        mpm_output = self.property_encoder(input_ids,
                                       attention_mask=prop_attention_mask,
                                       encoder_hidden_states=text_embeds,
                                       encoder_attention_mask=text_attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mpm = loss_fct(mpm_output.permute((0, 2, 1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mpm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mpm = (1 - alpha) * loss_mpm + alpha * loss_distill_text

        return loss_mlm, loss_mpm, loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, text_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask_pv(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        arg_sche = AttrDict(self.config['schedular'])
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print('qqq', metric)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        prop, text, labels = train_batch
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(labels.device)
        prop_input = self.tokenizer2(prop, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(labels.device)
        # print(text_input.input_ids[:4], prop[:4], text_input.input_ids.shape)
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop_input.input_ids[:, 1:], prop_input.attention_mask[:, 1:], text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], labels, alpha=alpha)
        loss = loss_mlm + loss_mpm + loss_ita + loss_itm
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss_mlm, prog_bar=True)
            self.log('loss_mpm', loss_mpm, prog_bar=True)
            self.log('loss_ita', loss_ita, prog_bar=True)
            self.log('loss_itm', loss_itm, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0:
            scheduler.step(self.current_epoch + self.warmup_steps)
        else:
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        self.training_step_outputs.append(torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm]))
        return torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm])

    def on_train_epoch_end(self):    # outputs: collection of returns from 'training_step'
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}, {tmp[1]:.4f}, {tmp[2]:.4f}, {tmp[3]:.4f}')
        self.training_step_outputs.clear()



##

c25ap = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/c25_ap.pkl')


ap = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/AP_320K.pkl')


iapdf =pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/iap.pkl')

capdf =pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/cap.pkl')
# lltss2.to_pickle('/spstorage/USERS/gina/Project/Chemcriptome/cap.pkl')
aapp = pd.concat([ap,iapdf,capdf, c25ap])

aapp = aapp[~aapp.index.duplicated()]
left_cids = list(set(aapp.index)-set(smidf.iloc[:,0]))
# with open(f'{maindir}/left_cds.txt', 'w'):f.write()
# ismidf = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/icid2smi.csv')
with open(f'{maindir}/left_cds.txt', 'w') as f:
    f.write('\n'.join([str(int(cd)) for cd in left_cids]))
left_cids = list(set(left_cids)-set(ismidf.iloc[:,0]))

all_int = pd.read_csv(DIR_AP+'/final2/all_int.csv',index_col=0)
cid_list = list(all_int.value_counts('cid')[all_int.value_counts('cid')>10].index)
cid_list = list(set(cid_list)-set(pd.concat([ap,iapdf]).index))

# len(set(list(ap[ap.index.isin(cid_list)].index)+list(iapdf[iapdf.index.isin(cid_list)].index)))
sub_int = all_int[all_int.cid.isin(aapp.index)].loc[:,['cid','standard_value','chembl_id']]
sub_int = sub_int[sub_int.standard_value<100000.0]

cd2chem = dict()
for cd in list(aapp.index):
    scdf = sub_int[sub_int.cid==cd].sort_values('standard_value')
    # scdf = scdf[scdf.standard_value<1000]
    scdf = scdf.iloc[0:50,:]
    cd2chem[cd] = ''.join(list(scdf.chembl_id))


lcid2smi = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/left_cid2smi.txt',sep='\t', header=None)

c2u=pd.read_csv(f'{maindir}/SPMM/chem2uniprot.tsv',sep='\t')

c2uo = list(set(c2u.Organism))
c2uo = [c2uo[1],c2uo[7],c2uo[11]]


import sqlite3
import pandas as pd

conn = sqlite3.connect("chembl_35.db")

query = """
SELECT
    md.chembl_id AS molecule_chembl_id,
    td.chembl_id AS target_chembl_id,
    td.organism AS target_organism,
    a.standard_type,
    a.standard_value,
    a.standard_units,
    a.pchembl_value
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN molecule_dictionary md ON a.molregno = md.molregno
WHERE a.standard_type IS NOT NULL
"""

df = pd.read_sql(query, conn)
print(df.head())
tdf = pd.read_sql("SELECT * FROM target_dictionary;", conn)
ch2c = pd.read_csv('/spstorage/USERS/gina/ChEMBL/cmp_chem2cid.txt',header=None, sep='\t')
ch2c.columns = ['molecule_chembl_id','cid']
cus=['Homo sapiens','Rattus norvegicus','Mus musculus']

cdf =df[df.target_organism.isin(cus)]

ccdf = pd.merge(cdf,ch2c)

cccdf = ccdf[ccdf.standard_units=='nM']

t2u=pd.read_csv('/spstorage/USERS/gina/ChEMBL/tg_chem2uniprot.tsv',sep='\t')
cccdf= cccdf[cccdf.target_chembl_id.isin(t2u.From)]

u2e=pd.read_csv('/spstorage/USERS/gina/ChEMBL/u2e.csv',index_col=0).dropna()
t2e = pd.merge(t2u,u2e,left_on='Entry',right_on='UNIPROT')
t2e = t2e.loc[:,['From','ENSEMBL']].drop_duplicates()
t2e = t2e[~t2e.From.duplicated()]

adf = pd.merge(cccdf, t2e, left_on = 'target_chembl_id', right_on='From')

adf = adf.loc[:,['standard_value','cid', 'ENSEMBL']].drop_duplicates()

amm = adf.value_counts('cid')

sadfs= []
cds = list(set(amm[amm>=10].index))

for cd in cds:
    sadf = adf[adf.cid==cd]
    sadf = sadf.sort_values('standard_value')
    sadf = sadf[~sadf.ENSEMBL.duplicated()]
    sadfs.append(sadf)


from datasets import Dataset


acds = list(aapp[aapp.index.isin(cds)].index)
data = [{'input_id': list(sadf.iloc[0:50,:].ENSEMBL), 'value': list(sadf.iloc[0:50,:].standard_value), 'cid': list(sadf.cid)[0]} for i,sadf in enumerate(sadfs) if cds[i] in acds]

dataset = Dataset.from_dict(data)

# with open('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/smi2ap_ens.pkl','wb') as f:
#     pickle.dump(dict(zip(smis,aptcks)),f)


cvv = [f'{a}{b}' for a in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=()") for b in range(10)]
cvv = [cvv[v] for i,v in enumerate(ap1a.columns)]
cvv = dict(zip(cvv, range(len(cvv))))
with open('./tkn_idx.pkl', 'wb') as f:
    pickle.dump(cvv,f)




# NumPy 기반 정렬 및 조인
cid2smi = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/cid2smi.txt',sep='\t', header=None)
lcid2smi = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/left_cid2smi.txt',sep='\t', header=None)
cid2smi = pd.concat([cid2smi,lcid2smi]).dropna()
cid2smi.columns = ['cid','smi']
aapp = aapp[aapp.index.isin(cid2smi.cid)]
aapp = aapp.iloc[:,list(ap1a.columns)]
aapp.columns = list(cvv.keys())
cols = np.array(aapp.columns)
values = aapp.values

result = []
for row in values:
    nonzero_idx = np.where(row != 0)[0]
    sorted_idx = nonzero_idx[np.argsort(-row[nonzero_idx])]
    result.append(''.join(cols[sorted_idx]))

# df['sorted_names'] = results



lcid2smi = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/left_cid2smi.txt',sep='\t', header=None)
lcid2smi.columns = ['cid','smi']
cid2smi = pd.read_csv('/spstorage/USERS/gina/Project/Chemcriptome/cid2smi.txt',sep='\t', header=None)

import pandas as pd
import numpy as np
import pickle
import glob

maindir = '/spstorage/USERS/gina/Project/Chemcriptome'
apd = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/SPMM/downstream_APs.pkl')

c25ap = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/c25_ap.pkl')


ap = pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/AP_320K.pkl')


iapdf =pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/iap.pkl')

capdf =pd.read_pickle('/spstorage/USERS/gina/Project/Chemcriptome/cap.pkl')
# lltss2.to_pickle('/spstorage/USERS/gina/Project/Chemcriptome/cap.pkl')
aapp = pd.concat([apd, ap,iapdf,capdf, c25ap])

aapp = aapp[~aapp.index.duplicated()]


for f  in glob.glob(f'{maindir}/SPMM/data/4_MoleculeNet/*'):
    pd.read_csv(f)
    
    .to_csv(f'{maindir}/SPMM/data/4_MoleculeNet/{f.split("/")[-1].split(".")[0]}.tsv', sep='\t', index=False, header=None)

dap = aapp.loc[list(dsmi2cid.cid),:]
ap1a = pd.read_pickle(f'{maindir}/ap1a.pkl')

ap = pd.read_pickle(f'{maindir}/tests/smi2ap_df.pkl')

dap = dap.loc[:,list(ap1a.columns)]
xx =pd.DataFrame([0]*len(dap.index))
xx.index= dap.index
dap = pd.concat([xx,dap,xx], axis=1)
dap.columns= list(ap.columns)
dap.index = list(dsmi2cid.smiles)
ap = pd.concat([ap,dap], axis=0)
ap = ap[~ap.index.duplicated()]
ap.to_pickle(f'{maindir}/tests/smi2ap_df.pkl')


cid2ap_sort[cd] for 
dsmi2cid = pd.read_csv(f'{maindir}/SPMM/smi2cids.txt', sep='\t', header=None).dropna()
dsmi2cid.columns = ['smiles','cid']
dsmi2cid = dsmi2cid[dsmi2cid.cid.isin(aapp.index)]
dff = glob.glob(f'{maindir}/SPMM/data/*/*csv')
for f in dff[265:]:
    if ('AP_' not in f)&('55_sorting' not in f):
        ffs = f.split('/')
        ffs[-1] = 'AP_368_sorting_'+ffs[-1]
        dfv = pd.read_csv(f)
        wnt = dfv.columns
        dfv = pd.merge(dfv, dsmi2cid, left_on= list(dfv.columns)[0], right_on='smiles')
        vv = []
        for smi in list(dfv[list(dfv.columns)[0]]):
            if smi in list(ap.index):
                sap = ap.loc[smi,:].sort_values(ascending=False)
                vv.append(''.join(list(sap[sap!=0].index)))
            else:
                vv.append(np.nan)
        dfv['ap'] = vv
        dfv= dfv.loc[:,list(wnt)+['ap']]
        dfv.to_csv('/'.join(ffs), index=False)
