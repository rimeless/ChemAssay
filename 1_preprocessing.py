import sys
src_dir = '/spstorage/USERS/gina/source' 
sys.path.append(src_dir)
from basic import *
from CC import *
from operator import itemgetter, add
import glob
from collections import OrderedDict

adir = '/spstorage/USERS/gina/Project/FD/assay/'

### 1. all set

# 1768450 AIDs
bio = pd.read_csv('/spstorage/DB/PUBCHEM/assay/bioassays.tsv.gz', sep='\t', compression='gzip')

# 296642802 activities
biodf = pd.read_csv('/spstorage/DB/PUBCHEM/assay/bioactivities.tsv.gz', sep='\t', compression='gzip')

### 2. Merged set ( based target ) - 1800 AIDs / 


vdf = pd.read_pickle(f'{adir}/assay_1800.pkl')

# 
with open('/spstorage/USERS/gina/Project/FD/assay/hh2_processpool_results_except_test.pkl','rb') as f:
    actss, inactss, fp_actss, fp_inactss = pickle.load(f)


actdf= [a[0] for a in actss]
actdf2 = pd.DataFrame(actdf)

iactdf= [a[0] for a in inactss]
iactdf2 = pd.DataFrame(iactdf)


ap = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_ap.pkl')
tap = ap[ap.index.isin(ttbiodf.CID)]



ttv = [a.split('|') if isinstance(a, str) else a for a in list(bio['BioAssay Types'])]



X1=actdf2.to_numpy(dtype=np.float32)
X2=iactdf2.to_numpy(dtype=np.float32)
X3=tap.sample(frac=1).iloc[0:1000,:].to_numpy(dtype=np.float32)

m1=np.isfinite(X1).all(axis=1)
m2=np.isfinite(X2).all(axis=1)
m3=np.isfinite(X3).all(axis=1)

X1=X1[m1]
X2=X2[m2]
X3=X3[m3]

X=np.vstack([X1,X2,X3])
labels=np.array(["actdf2"]*X1.shape[0]+["iactdf2"]*X2.shape[0]+["tactdf2"]*X3.shape[0],dtype=object)

reducer=umap.UMAP(n_neighbors=30,min_dist=0.3,n_components=2,metric="cosine",random_state=0)
emb=reducer.fit_transform(X)

plt.figure(figsize=(7.5,6.5),dpi=200)
for name in ["actdf2","iactdf2","tactdf2"]:
    idx=labels==name
    plt.scatter(emb[idx,0],emb[idx,1],s=10,label=name)



plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP (actdf2 vs iactdf2 vs tactdf2)")
plt.legend(frameon=False,markerscale=2)
plt.tight_layout()
plt.show()


with open('/spstorage/USERS/gina/Project/FD/assay/hh2_processpool_results.pkl','rb') as f:
    actss, inactss, fp_actss, fp_inactss = pickle.load(f)



### 3. Merged set ( based compound similarity )
# act cutoff 70% / 80% | inact cutoff 20% | k 128 | b 64 | r 2
# active 0.7 / 0.75 / 0.8 / 0.85  
# # all aid 1768450 → # act>=3 aid 212568  → 병합: 74052 ~ 86662
# inactive cut-off 시 30% 정도 잔류

# groups | members_final | gact_final | ginact_final
# groups_idx: LSH+검증 단계에서 얻은 인덱스 그룹(list[list[int]])
# act_arrays, inact_arrays: 각 AID의 정렬된 uint64 배열 리스트

# ginact_final 에서 비어있지 않은 경우 14136  --> 18% 만이 비활성 정보 보유
# 나머지 82% AID 를 날리면 act도 전체 data 대비 27%가 날아감
# 처음 212568 에서 비어 있는 경우 193395  --> 9% 만이 비활성 정보 보유
# iact 269735329  act 8613099
# 1768450 --> 212568  12%만 있음    296642802 --> 280312699 (5% 정도 날아감)

with open(f'{adir}/assay_groups_act70_k{128}_b{64}_r{2}_merged.pkl','rb') as f:
    groups, members_final, gact_final, ginact_final = pickle.load(f)





# test


ttbiodf = pd.read_pickle('/spstorage/USERS/gina/Project/FD/ttbiodf.pkl')

tb1 = ttbiodf[ttbiodf.AID.isin(aaids)]
tb2 = ttbiodf[ttbiodf['UniProts IDs'].isin(aaids)]

with open('/spstorage/USERS/gina/Project/FD/assay/aaids.pkl','rb') as f:
    aaids = pickle.load(f)

dtis = pd.read_csv('/spstorage/USERS/sung/DATA/DTI_DBs/AllDTI_strictver.tsv',sep='\t')
dtis.columns = ['cid','uniprot','source']
dtis = dtis[dtis.uniprot.isin(aaids)]
dtis2 = pd.read_csv('/spstorage/DB/DTI/dataset_benchmark/Benchmark_DTI.csv', index_col=0)
dtis2 = dtis2[dtis2.Uniprot.isin(aaids)]

cds = list(ttbiodf.CID)
cds = list(set(cds))

aa = pd.read_pickle(f'/spstorage/USERS/gina/Project/FD/assay/simdata/score_fitting_method_0.pkl')
cds = cds + list(set(dtis[dtis.cid.isin(aa.columns)].cid)) 
unis = aa.index.tolist()





