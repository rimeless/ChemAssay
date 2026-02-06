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

bio2 = pd.read_csv('/spstorage/DB/PUBCHEM/assay/2026/bioassays.tsv.gz', sep='\t', compression='gzip')

# 296642802 activities
biodf = pd.read_csv('/spstorage/DB/PUBCHEM/assay/bioactivities.tsv.gz', sep='\t', compression='gzip')

biodf2 = pd.read_csv('/spstorage/DB/PUBCHEM/assay/2026/bioactivities.tsv.gz', sep='\t', compression='gzip')

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
resdf.to_pickle(f'{adir}/ap_cent_mean.pkl')
final_df.to_pickle(f'{adir}/alldf_pivot.pkl')




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





### APM, ECFP4, SMILES? 기준으로 각 assay 마다 profile 구하기. STD 도 active / inactive 각각 구하기


##
assay_profiles = []
all_aid = list(set(biodf.AID))
cbiodf = biodf[biodf.CID.isin(ap.index)]
for i, aids in enumerate(all_aid):
    if i%1000==0:
        print(i, '/', len(all_aid))
    bdf = cbiodf[cbiodf.AID==aids]
    bdf_act = bdf[bdf['Activity Outcome']=='Active']
    bdf_inact = bdf[bdf['Activity Outcome']=='Inactive']
    bdf_uns = bdf[bdf['Activity Outcome']=='Unspecified']
    bdf_inc = bdf[bdf['Activity Outcome']=='Inconclusive']
    if (len(bdf_uns)!=0):
        cbdf_uns = ap[ap.index.isin(bdf_uns.CID)]
        uns_mean = cbdf_uns.mean(axis=0).tolist()
        if len(bdf_uns)<5:
            uns_std = '-'
        else:
            uns_std = cbdf_uns.std(axis=0).mean().item()
    else:
        uns_mean = 0
        uns_std = 0
    if (len(bdf_inc)!=0):
        cbdf_inc = ap[ap.index.isin(bdf_inc.CID)]
        inc_mean = cbdf_inc.mean(axis=0).tolist()
        if len(bdf_inc)<5:
            inc_std = '-'
        else:
            inc_std = cbdf_inc.std(axis=0).mean().item()
    else:
        inc_mean = 0
        inc_std = 0
    if (len(bdf_act)!=0):
        cbdf_act = ap[ap.index.isin(bdf_act.CID)]
        act_mean = cbdf_act.mean(axis=0).tolist()
        if len(bdf_act)<5:
            act_std = '-'
        else:
            act_std = cbdf_act.std(axis=0).mean().item()
    else:
        act_mean = 0
        act_std = 0
    if len(bdf_inact)==0:
        inact_mean = 0
        inact_std = 0
    else:
        cbdf_inact = ap[ap.index.isin(bdf_inact.CID)]
        inact_mean = cbdf_inact.mean(axis=0).tolist()
        if len(bdf_inact)<5:
            inact_std = '-'
        else:
            inact_std = cbdf_inact.std(axis=0).mean().item()
    assay_profiles.append([act_mean, act_std, inact_mean, inact_std, uns_mean, uns_std, inc_mean, inc_std, len(bdf_act), len(bdf_inact), len(bdf_uns), len(bdf_inc)])


for i, aids in enumerate(all_aid):
    if i%1000==0:
        print(i, '/', len(all_aid))
    bdf = cbiodf[cbiodf.AID==aids]
    bdf_act = bdf[bdf['Activity Outcome']=='Active']
    bdf_inact = bdf[bdf['Activity Outcome']=='Inactive']
    bdf_uns = bdf[bdf['Activity Outcome']=='Unspecified']
    bdf_inc = bdf[bdf['Activity Outcome']=='Inconclusive']
    if (len(bdf_uns)!=0):
        cbdf_uns = ap[ap.index.isin(bdf_uns.CID)]
        uns_mean = cbdf_uns.mean(axis=0).tolist()
        if len(bdf_uns)<5:
            uns_std = '-'
        else:
            uns_std = cbdf_uns.std(axis=0).mean().item()
    else:
        uns_mean = 0
        uns_std = 0
    if (len(bdf_inc)!=0):
        cbdf_inc = ap[ap.index.isin(bdf_inc.CID)]
        inc_mean = cbdf_inc.mean(axis=0).tolist()
        if len(bdf_inc)<5:
            inc_std = '-'
        else:
            inc_std = cbdf_inc.std(axis=0).mean().item()
    else:
        inc_mean = 0
        inc_std = 0
    if (len(bdf_act)!=0):
        cbdf_act = ap[ap.index.isin(bdf_act.CID)]
        act_mean = cbdf_act.mean(axis=0).tolist()
        if len(bdf_act)<5:
            act_std = '-'
        else:
            act_std = cbdf_act.std(axis=0).mean().item()
    else:
        act_mean = 0
        act_std = 0
    if len(bdf_inact)==0:
        inact_mean = 0
        inact_std = 0
    else:
        cbdf_inact = ap[ap.index.isin(bdf_inact.CID)]
        inact_mean = cbdf_inact.mean(axis=0).tolist()
        if len(bdf_inact)<5:
            inact_std = '-'
        else:
            inact_std = cbdf_inact.std(axis=0).mean().item()
    assay_profiles.append([act_mean, act_std, inact_mean, inact_std, uns_mean, uns_std, inc_mean, inc_std, len(bdf_act), len(bdf_inact), len(bdf_uns), len(bdf_inc)])





import pandas as pd
import numpy as np

# 1. 필요한 컬럼만 추출 및 병합 (Memory 효율화)
# cbiodf와 ap(assay profile/CID features)를 CID 기준으로 병합합니다.
# ap의 인덱스가 CID라고 가정합니다.
merged_df = cbiodf[['AID', 'CID', 'Activity Outcome']].merge(
    ap, left_on='CID', right_index=True, how='inner'
)

# 2. 그룹화하여 평균(mean)과 표준편차(std)를 한꺼번에 계산
# AID와 Outcome별로 모든 feature의 평균을 구하고, 각 행의 count를 계산합니다.
grouped = merged_df.groupby(['AID', 'Activity Outcome'])

# Feature 컬럼 리스트 추출 (ap의 컬럼들)
feature_cols = ap.columns.tolist()

# 통계량 계산 (평균 / 표준편차 / 개수)
stats = grouped[feature_cols].agg(['mean', 'std', 'count'])

# 3. 'std'에 대한 사용자 정의 규칙 적용 (N < 5 이면 NaN 또는 0)
# 'std'의 평균(mean of stds)을 구하기 전에 각 feature별 std를 계산
# 질문자님의 기존 로직: cbdf.std(axis=0).mean()
final_stats = pd.DataFrame()
final_stats['mean_list'] = stats[feature_cols].xs('mean', axis=1, level=1).values.tolist()
final_stats['count'] = stats[feature_cols].xs('count', axis=1, level=1).iloc[:, 0]

# std 계산 및 N < 5 조건 적용
stds_all = stats[feature_cols].xs('std', axis=1, level=1)
avg_std = stds_all.mean(axis=1)
final_stats['std_val'] = np.where(final_stats['count'] < 5, np.nan, avg_std)

# 4. Outcome별로 컬럼 재구성 (Unstack/Pivot)
# AID별로 한 줄에 [act, inact, uns, inc] 정보가 다 들어가도록 변환합니다.
res = final_stats.unstack('Activity Outcome')

# 5. 부족한 Outcome 컬럼(Active, Inactive 등)이 없을 경우를 대비해 기본값 처리
outcomes = ['Active', 'Inactive', 'Unspecified', 'Inconclusive']
for out in outcomes:
    if out not in res.columns.get_level_values(1):
        # 해당 Outcome이 아예 없는 경우 0으로 채움
        pass 

# 최종 결과물 리스트 생성 (필요한 경우)
# 하지만 DataFrame 상태로 유지하는 것이 학습 시 훨씬 빠릅니다.


import pandas as pd
import numpy as np

# 1. 필요한 컬럼만 추출 및 병합 (Memory 효율화)
# cbiodf와 ap(assay profile/CID features)를 CID 기준으로 병합합니다.
# ap의 인덱스가 CID라고 가정합니다.
cb1= cbiodf[cbiodf.AID.isin(all_aid[0:1000])]

merged_df = cb1[['AID', 'CID', 'Activity Outcome']].merge(
    ap, left_on='CID', right_index=True, how='inner'
)

# 2. 그룹화하여 평균(mean)과 표준편차(std)를 한꺼번에 계산
# AID와 Outcome별로 모든 feature의 평균을 구하고, 각 행의 count를 계산합니다.
grouped = merged_df.groupby(['AID', 'Activity Outcome'])

# Feature 컬럼 리스트 추출 (ap의 컬럼들)
feature_cols = ap.columns.tolist()

# 통계량 계산 (평균 / 표준편차 / 개수)
stats = grouped[feature_cols].agg(['mean', 'std', 'count'])

# 3. 'std'에 대한 사용자 정의 규칙 적용 (N < 5 이면 NaN 또는 0)
# 'std'의 평균(mean of stds)을 구하기 전에 각 feature별 std를 계산
# 질문자님의 기존 로직: cbdf.std(axis=0).mean()
final_stats = pd.DataFrame()
final_stats['mean_list'] = stats[feature_cols].xs('mean', axis=1, level=1).values.tolist()
final_stats['count'] = stats[feature_cols].xs('count', axis=1, level=1).iloc[:, 0]

# std 계산 및 N < 5 조건 적용
stds_all = stats[feature_cols].xs('std', axis=1, level=1)
avg_std = stds_all.mean(axis=1)
final_stats['std_val'] = np.where(final_stats['count'] < 5, np.nan, avg_std)

# 4. Outcome별로 컬럼 재구성 (Unstack/Pivot)
# AID별로 한 줄에 [act, inact, uns, inc] 정보가 다 들어가도록 변환합니다.
res = final_stats.unstack('Activity Outcome')

# 5. 부족한 Outcome 컬럼(Active, Inactive 등)이 없을 경우를 대비해 기본값 처리
outcomes = ['Active', 'Inactive', 'Unspecified', 'Inconclusive']
for out in outcomes:
    if out not in res.columns.get_level_values(1):
        # 해당 Outcome이 아예 없는 경우 0으로 채움
        pass 

# 최종 결과물 리스트 생성 (필요한 경우)
# 하지만 DataFrame 상태로 유지하는 것이 학습 시 훨씬 빠릅니다.



import pandas as pd
import numpy as np
from tqdm import tqdm

def process_in_chunks(cbiodf, ap, chunk_size=10000):
    all_results = []
    
    # AID 리스트를 유니크하게 뽑아 청크로 나눔
    unique_aids = cbiodf['AID'].unique()
    
    # 176만 개의 AID를 chunk_size 단위로 반복
    for i in tqdm(range(0, len(unique_aids), chunk_size)):
        target_aids = unique_aids[i:i + chunk_size]
        
        # 1. 현재 청크에 해당하는 cbiodf 부분 추출
        temp_cbio = cbiodf[cbiodf['AID'].isin(target_aids)]
        
        # 2. 필요한 CID들만 ap에서 가져오기 (이게 핵심!)
        target_cids = temp_cbio['CID'].unique()
        temp_ap = ap.loc[ap.index.intersection(target_cids)]
        
        # 3. 소규모 병합 (이 정도 크기는 메모리에서 매우 빠름)
        merged = temp_cbio.merge(temp_ap, left_on='CID', right_index=True)
        
        # 4. 통계 계산 (Vectorized)
        feature_cols = ap.columns
        stats = merged.groupby(['AID', 'Activity Outcome'])[feature_cols].agg(['mean', 'std', 'count'])
        
        # 5. 결과 정리
        # 여기서 N < 5 조건 등을 적용하여 임시 저장
        all_results.append(stats)
        
    # 모든 청크 결과 병합
    final_df = pd.concat(all_results)
    return final_df

# 실행
# res = process_in_chunks(cbiodf, ap)

from joblib import Parallel, delayed
import numpy as np

# ap 데이터프레임을 디스크에 저장 (압축 없이 저장해야 mmap 속도가 빠름)
# ap의 인덱스(CID)는 나중에 매핑을 위해 별도로 관리하는 것이 좋습니다.
ap_values = ap.values.astype('float32') # 메모리 절약을 위해 float32 권장
ap_index = ap.index.values

import joblib
joblib.dump(ap_values, f'{adir}/ap_values.mmap')
joblib.dump(ap_index, f'{adir}/ap_index.mmap')

print("데이터 덤프 완료!")
def compute_chunk(target_aids, cbiodf, ap_mmap, ap_index_mmap):
    # mmap_mode='r' 덕분에 이 데이터는 모든 프로세스가 공유합니다.
    # 각 프로세스마다 메모리를 새로 점유하지 않습니다.
    
    # 1. 현재 담당한 AID 데이터 필터링
    subset = cbiodf[cbiodf['AID'].isin(target_aids)]
    
    # 2. CID를 인덱스로 변환하여 mmap 데이터에서 슬라이싱
    # (효율적인 인덱싱을 위해 CID -> Row Index 매핑 딕셔너리가 있으면 좋음)
    cid_to_idx = {cid: i for i, cid in enumerate(ap_index_mmap)}
    
    # 3. 필요한 로직 수행...
    # 예: 해당 CID들의 평균/표준편차 계산
    # stats = ...
    return stats

# 병렬 실행부
# mmap으로 로드 (메모리를 거의 먹지 않음)
ap_mmap = joblib.load(f'{adir}/ap_values.mmap', mmap_mode='r')
ap_index_mmap = joblib.load(f'{adir}/ap_index.mmap', mmap_mode='r')


cdbiodf = cbiodf.loc[:,['AID','CID','Activity Outcome']]
cdbiodf = cdbiodf.drop_duplicates()

# AID 쪼개기
num_cores = 4 # 메모리 여유에 따라 조절
aid_splits = np.array_split(cbiodf['AID'].unique(), num_cores)

results = Parallel(n_jobs=num_cores)(
    delayed(compute_chunk)(split, cbiodf, ap_mmap, ap_index_mmap) 
    for split in aid_splits
)


# 개념적인 AID별 Stratified Split 로직
def split_by_aid(df, test_ratio=0.1, min_samples=5):
    # 샘플이 너무 적은 AID는 그냥 Train에 넣거나 제외
    valid_aids = df['AID'].value_counts()[lambda x: x >= min_samples].index
    
    # AID별로 그룹화하여 인덱스를 무작위로 섞은 뒤 나눔
    train_indices, test_indices = [], []
    
    for aid, group in df[df['AID'].isin(valid_aids)].groupby('AID'):
        shuffled = group.sample(frac=1, random_state=42).index
        split_point = int(len(shuffled) * (1 - test_ratio))
        
        train_indices.extend(shuffled[:split_point])
        test_indices.extend(shuffled[split_point:])
        
    return train_indices, test_indices


cds = list(ap.index)


from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import RDLogger
import pandas as pd
from tqdm import tqdm

# RDKit의 자잘한 경고 로그 끄기 (속도와 정신건강에 좋습니다)
RDLogger.DisableLog('rdApp.*')

def sdf2scf_worker(info):
    cids, sdf, _ = info
    target_cids = set(cids) # list보다 set이 검색 속도가 훨씬 빠릅니다
    
    results = []
    indices = []
    smis = []
    suppl = Chem.SDMolSupplier(sdf, removeHs=True)
    
    for mol in suppl:
        if mol is None: continue
        
        try:
            # GetPropsAsDict() 대신 GetProp으로 필요한 것만 쏙
            cid = int(mol.GetProp('PUBCHEM_COMPOUND_CID'))
            
            if cid in target_cids:
                # 스캐폴드 추출
                scaffold_mol = GetScaffoldForMol(mol)
                scf_smiles = Chem.MolToSmiles(scaffold_mol)
                
                results.append(scf_smiles)
                indices.append(cid)
                smis.append(Chem.MolToSmiles(mol))
        except:
            continue
            
    return pd.DataFrame({'scaffold': results, 'smiles': smis}, index=indices)



# --- 병렬 실행 ---
# n_jobs=-1은 가용한 모든 CPU 코어를 사용합니다.
# 200만 개면 코어 수에 따라 몇 분에서 수십 분 내로 끝날 겁니다.
print(f"Processing ~2M compounds across {len(infos)} batches...")

parallel_results2 = Parallel(n_jobs=-1)(
    delayed(sdf2scf_worker)(info) for info in tqdm(infos)
)

# 결과 합치기
final_scf_df2 = pd.concat(parallel_results)
print(f"Total scaffolds extracted: {len(final_scf_df)}")




def sdf2scf(info):
	cids, sdf, tps = info
	mm = Chem.SDMolSupplier(sdf, removeHs=True) 
	mm = [m for m in mm if m is not None]
	mm = [m for m in mm if m.GetPropsAsDict()['PUBCHEM_COMPOUND_CID'] in cids]
	mn =  [m.GetPropsAsDict()['PUBCHEM_COMPOUND_CID'] for m in mm]
	scf = [Chem.MolToSmiles(GetScaffoldForMol(m)) for m in mm]
	scfs = pd.DataFrame(scf)
	scfs.index = mn
	return scfs



def sdf2smi(info):
	cids, sdf, tps = info
	mm = Chem.SDMolSupplier(sdf, removeHs=True) 
	mm = [m for m in mm if m is not None]
	mm = [m for m in mm if m.GetPropsAsDict()['PUBCHEM_COMPOUND_CID'] in cids]
	mn =  [m.GetPropsAsDict()['PUBCHEM_COMPOUND_CID'] for m in mm]
	scf = [Chem.MolToSmiles(m) for m in mm]
	scfs = pd.DataFrame(scf)
	scfs.index = mn
	return scfs


scfs = mclapply(infos, sdf2scf, ncpu=80)
smis = mclapply(infos, sdf2smi, ncpu=80)



# 1. AID와 Activity Outcome별로 CID를 리스트로 묶기
cid_lists = fbiodf2.groupby(['AID', 'Activity Outcome'])['CID'].apply(list)

# 결과 확인 (MultiIndex 형태)
# index: (AID, Activity Outcome), value: [CID1, CID2, ...]
# print(cid_lists.head())

# # 2. 보기 편하게 DataFrame으로 변환하고 싶다면
# ap_lists = cid_lists.reset_index()
# ap_lists.columns = ['AID', 'Activity Outcome', 'CID_list']


from collections import defaultdict

result_dict = defaultdict(dict)
for (aid, outcome), cids in cid_lists2.items():
    result_dict[aid][outcome] = cids



# # scaffold 정보 병합 (final_scf_df의 인덱스가 CID라고 가정)
# df_with_scf = cdbiodf.merge(final_scf_df, left_on='CID', right_index=True, how='left')

# # 스캐폴드가 없는 경우(오류 등)를 대비해 'Unknown' 처리
# df_with_scf['scaffold'] = df_with_scf['scaffold'].fillna('Unknown')



# CID를 키로, Scaffold SMILES를 값으로 하는 딕셔너리 생성
# (final_scf_df의 인덱스가 CID라고 가정)

final_scf_df = pd.read_pickle(f'{adir}/scaffold_df.pkl')

cid_to_scf = final_scf_df['scaffold'].to_dict()

# import random
# import random
# from collections import defaultdict

# # 1. 각 스캐폴드가 어느 AID에 몇 개씩 포함되어 있는지 맵 생성
# # scf_to_aid_counts[scaffold][aid] = count
# scf_to_aid_counts = defaultdict(lambda: defaultdict(int))
# aid_total_counts = defaultdict(int)

# for aid, outcomes in result_dict.items():
#     for outcome, cids in outcomes.items():
#         for cid in cids:
#             scf = cid_to_scf.get(cid, 'Unknown')
#             scf_to_aid_counts[scf][aid] += 1
#             aid_total_counts[aid] += 1

# # 2. 각 AID가 채워야 할 목표 테스트 개수 (10%)
# aid_test_targets = {aid: total * 0.1 for aid, total in aid_total_counts.items()}
# aid_current_test_counts = defaultdict(int)

# # 3. 전역 테스트 스캐폴드 집합
# global_test_scaffolds = set()

# # 4. 스캐폴드들을 무작위로 섞어서 순회
# all_scaffolds = list(scf_to_aid_counts.keys())
# random.seed(42)
# random.shuffle(all_scaffolds)

# for scf in all_scaffolds:
#     # 이 스캐폴드를 테스트셋에 넣었을 때, 
#     # 혜택을 받는(목표치를 아직 못 채운) AID가 있는지 확인
#     useful = False
#     for aid, count in scf_to_aid_counts[scf].items():
#         if aid_current_test_counts[aid] < aid_test_targets[aid]:
#             useful = True
#             break
    
#     if useful:
#         global_test_scaffolds.add(scf)
#         # 이 스캐폴드에 포함된 모든 AID의 테스트 카운트 업데이트
#         for aid, count in scf_to_aid_counts[scf].items():
#             aid_current_test_counts[aid] += count

# # 5. 결과 확인용: 각 AID별 실제 테스트 비율 계산
# ratios = [aid_current_test_counts[aid] / aid_total_counts[aid] for aid in aid_total_counts]
# print(f"평균 테스트 비율: {sum(ratios)/len(ratios):.4f}")



# global_random_test_cids = set()
# cid_to_aids = defaultdict(list)

# # 각 CID가 속한 AID 리스트 미리 확보
# for aid, outcomes in result_dict.items():
#     for outcome, cids in outcomes.items():
#         for cid in cids:
#             cid_to_aids[cid].append(aid)

# all_cids = list(cid_to_aids.keys())
# random.shuffle(all_cids)

# aid_current_random_counts = defaultdict(int)

# for cid in all_cids:
#     useful = False
#     for aid in cid_to_aids[cid]:
#         if aid_current_random_counts[aid] < aid_test_targets[aid]:
#             useful = True
#             break
    
#     if useful:
#         global_random_test_cids.add(cid)
#         for aid in cid_to_aids[cid]:
#             aid_current_random_counts[aid] += 1



# fabiodf = ffbiodf[ffbiodf.CID.isin(apap.index)]


# cid_list = list(set(biodf.CID)-set(cid_list))


# biodf2= pd.read_csv('/spstorage/DB/PUBCHEM/assay/2026/bioactivities.tsv.gz', sep='\t', compression='gzip')

# fbiodf2 = biodf2.loc[:,['AID','CID','Activity Outcome']].drop_duplicates()

# ap=pd.read_pickle(f'{adir}/pubchem_ap_opt.pkl')

# fbiodf2 = fbiodf2[fbiodf2.CID.isin(ap.index)]


# # 1. AID와 Activity Outcome별로 CID를 리스트로 묶기
# cid_lists2 = fbiodf2.groupby(['AID', 'Activity Outcome'])['CID'].apply(list)

# # 결과 확인 (MultiIndex 형태)
# # index: (AID, Activity Outcome), value: [CID1, CID2, ...]
# print(cid_lists.head())

# # 2. 보기 편하게 DataFrame으로 변환하고 싶다면
# ap_lists = cid_lists2.reset_index()
# ap_lists.columns = ['AID', 'Activity Outcome', 'CID_list']


import random
from collections import defaultdict

# 1. 각 스캐폴드가 어느 AID에 몇 개씩 포함되어 있는지 맵 생성
# scf_to_aid_counts[scaffold][aid] = count
scf_to_aid_counts = defaultdict(lambda: defaultdict(int))
aid_total_counts = defaultdict(int)

for aid, outcomes in result_dict.items():
    for outcome, cids in outcomes.items():
        for cid in cids:
            scf = cid_to_scf.get(cid)
            # 스캐폴드가 없거나 Unknown인 경우, CID 자체를 스캐폴드 ID로 사용
            if scf is None or scf == 'Unknown' or scf == '':
                scf = f"NOSCF_{cid}"
            scf_to_aid_counts[scf][aid] += 1
            aid_total_counts[aid] += 1


# scf_to_aid_counts2 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# aid_total_counts2 = defaultdict(int)

# for aid, outcomes in result_dict.items():
#     for outcome, cids in outcomes.items():
#         for cid in cids:
#             scf = cid_to_scf.get(cid)
#             # 스캐폴드가 없거나 Unknown인 경우, CID 자체를 스캐폴드 ID로 사용
#             if scf is None or scf == 'Unknown' or scf == '':
#                 scf = f"NOSCF_{cid}"
#             scf_to_aid_counts2[aid][outcome][scf] += 1
#             aid_total_counts2[aid] += 1



# # 2. 각 AID가 채워야 할 목표 테스트 개수 (10%)
# aid_test_targets = {aid: total * 0.1 for aid, total in aid_total_counts.items()}
# aid_current_test_counts = defaultdict(int)

# # 3. 전역 테스트 스캐폴드 집합
# global_test_scaffolds = set()
# # 4. 스캐폴드들을 무작위로 섞어서 순회
# all_scaffolds = list(scf_to_aid_counts.keys())
# random.seed(42)
# random.shuffle(all_scaffolds)

# total_compounds = sum(aid_total_counts.values())
# global_test_limit = total_compounds * 0.12 # 전역적으로 약 12%까지만 허용 (완충지대)
# current_global_test_count = 0

# for scf in all_scaffolds:
#     # 이 스캐폴드에 포함된 전체 화합물 수 계산
#     scf_size = sum(scf_to_aid_counts[scf].values())
    
#     # 1. 전역 리밋을 넘지 않는지 확인
#     if current_global_test_count + scf_size > global_test_limit:
#         continue
#     # 2. 이 스캐폴드가 '꼭 필요한' AID가 있는지 확인 (Stricter Condition)
#     # 현재 테스트 비율이 8% 미만인 AID가 하나라도 있을 때만 추가
#     needed = False
#     for aid, count in scf_to_aid_counts[scf].items():
#         if aid_current_test_counts[aid] < (aid_total_counts[aid] * 0.08): 
#             needed = True
#             break
    
#     if needed:
#         global_test_scaffolds.add(scf)
#         for aid, count in scf_to_aid_counts[scf].items():
#             aid_current_test_counts[aid] += count
#         current_global_test_count += scf_size

# # --- 결과 확인 ---
# final_ratios = [aid_current_test_counts[aid] / aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid] > 0]
# print(f"조정된 평균 테스트 비율: {sum(final_ratios)/len(final_ratios):.4f}")

# # 4. 스캐폴드들을 무작위로 섞어서 순회
# all_scaffolds = list(scf_to_aid_counts.keys())
# random.seed(42)
# random.shuffle(all_scaffolds)

# for scf in all_scaffolds:
#     # 이 스캐폴드를 테스트셋에 넣었을 때, 
#     # 혜택을 받는(목표치를 아직 못 채운) AID가 있는지 확인
#     useful = False
#     for aid, count in scf_to_aid_counts[scf].items():
#         if aid_current_test_counts[aid] < aid_test_targets[aid]:
#             useful = True
#             break
    
#     if useful:
#         global_test_scaffolds.add(scf)
#         # 이 스캐폴드에 포함된 모든 AID의 테스트 카운트 업데이트
#         for aid, count in scf_to_aid_counts[scf].items():
#             aid_current_test_counts[aid] += count

# # 5. 결과 확인용: 각 AID별 실제 테스트 비율 계산
# ratios = [aid_current_test_counts[aid] / aid_total_counts[aid] for aid in aid_total_counts]
# print(f"평균 테스트 비율: {sum(ratios)/len(ratios):.4f}")


# global_random_test_cids = set()
# cid_to_aids = defaultdict(list)

# # 각 CID가 속한 AID 리스트 미리 확보
# for aid, outcomes in result_dict.items():
#     for outcome, cids in outcomes.items():
#         for cid in cids:
#             cid_to_aids[cid].append(aid)

# all_cids = list(cid_to_aids.keys())
# random.shuffle(all_cids)

# aid_current_random_counts = defaultdict(int)

# for cid in all_cids:
#     useful = False
#     for aid in cid_to_aids[cid]:
#         if aid_current_random_counts[aid] < aid_test_targets[aid]:
#             useful = True
#             break
    
#     if useful:
#         global_random_test_cids.add(cid)
#         for aid in cid_to_aids[cid]:
#             aid_current_random_counts[aid] += 1


# final_ratios = [aid_current_random_counts[aid] / aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid] > 0]

# final_scf_df[final_scf_df.scaffold.isin(global_test_scaffolds)]



# ##
# import random
# from collections import defaultdict

# random.seed(42)

# tau=0.10
# aid_lower=0.08
# aid_upper=0.12
# out_lower=0.06
# out_upper=0.14
# min_out_total=50
# lambda_out=1.0
# w_cap=5.0
# global_buffer=1.20

# scf_to_aid_out_counts=defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# scf_to_aid_counts=defaultdict(lambda: defaultdict(int))
# scf_size_total=defaultdict(int)
# aid_total_counts=defaultdict(int)
# aid_out_total_counts=defaultdict(lambda: defaultdict(int))
# all_outcomes=set()

# for aid,outcomes in result_dict.items():
# 	for outcome,cids in outcomes.items():
# 		all_outcomes.add(outcome)
# 		for cid in cids:
# 			scf=cid_to_scf.get(cid)
# 			if scf is None or scf=='Unknown' or scf=='':
# 				scf=f"NOSCF_{cid}"
# 			scf_to_aid_out_counts[scf][aid][outcome]+=1
# 			scf_to_aid_counts[scf][aid]+=1
# 			scf_size_total[scf]+=1
# 			aid_total_counts[aid]+=1
# 			aid_out_total_counts[aid][outcome]+=1

# all_scaffolds=list(scf_to_aid_counts.keys())
# random.shuffle(all_scaffolds)

# aid_target={aid:aid_total_counts[aid]*tau for aid in aid_total_counts}
# aid_lower_target={aid:aid_total_counts[aid]*aid_lower for aid in aid_total_counts}
# aid_upper_target={aid:aid_total_counts[aid]*aid_upper for aid in aid_total_counts}

# aid_out_target=defaultdict(dict)
# aid_out_lower_target=defaultdict(dict)
# aid_out_upper_target=defaultdict(dict)
# for aid in aid_total_counts:
# 	for outcome in all_outcomes:
# 		n=aid_out_total_counts[aid].get(outcome,0)
# 		if n<=0:
# 			continue
# 		if n<min_out_total:
# 			lo=max(0,int(n*tau)-3)
# 			hi=min(n,int(n*tau)+3)
# 			t=n*tau
# 		else:
# 			lo=n*out_lower
# 			hi=n*out_upper
# 			t=n*tau
# 		aid_out_target[aid][outcome]=t
# 		aid_out_lower_target[aid][outcome]=lo
# 		aid_out_upper_target[aid][outcome]=hi

# aid_current_test_counts=defaultdict(int)
# aid_out_current_test_counts=defaultdict(lambda: defaultdict(int))
# global_test_scaffolds=set()

# total_compounds=sum(aid_total_counts.values())
# global_test_limit=total_compounds*tau*global_buffer
# current_global_test_count=0

# unselected=set(all_scaffolds)

# def _score_scf(scf):
# 	size=scf_size_total[scf]
# 	if current_global_test_count+size>global_test_limit:
# 		return None
# 	needed=False
# 	for aid,cnt in scf_to_aid_counts[scf].items():
# 		if aid_total_counts[aid]<=0:
# 			continue
# 		if aid_current_test_counts[aid]<aid_lower_target[aid] and cnt>0:
# 			needed=True
# 			break
# 	if not needed:
# 		for aid,od in scf_to_aid_out_counts[scf].items():
# 			for outcome,cnt in od.items():
# 				if cnt<=0:
# 					continue
# 				lo=aid_out_lower_target[aid].get(outcome,None)
# 				if lo is None:
# 					continue
# 				if aid_out_current_test_counts[aid][outcome]<lo:
# 					needed=True
# 					break
# 			if needed:
# 				break
# 	if not needed:
# 		return None
# 	overflow=False
# 	for aid,cnt in scf_to_aid_counts[scf].items():
# 		if aid_total_counts[aid]<=0:
# 			continue
# 		if aid_current_test_counts[aid]+cnt>aid_upper_target[aid]:
# 			overflow=True
# 			break
# 	if overflow:
# 		return None
# 	for aid,od in scf_to_aid_out_counts[scf].items():
# 		for outcome,cnt in od.items():
# 			hi=aid_out_upper_target[aid].get(outcome,None)
# 			if hi is None:
# 				continue
# 			if aid_out_current_test_counts[aid][outcome]+cnt>hi:
# 				overflow=True
# 				break
# 		if overflow:
# 			break
# 	if overflow:
# 		return None
# 	gain=0.0
# 	for aid,cnt in scf_to_aid_counts[scf].items():
# 		if cnt<=0:
# 			continue
# 		need=max(0.0,aid_target[aid]-aid_current_test_counts[aid])
# 		gain+=min(cnt,need)
# 	gain_out=0.0
# 	for aid,od in scf_to_aid_out_counts[scf].items():
# 		for outcome,cnt in od.items():
# 			if cnt<=0:
# 				continue
# 			t=aid_out_target[aid].get(outcome,None)
# 			if t is None:
# 				continue
# 			need=max(0.0,t-aid_out_current_test_counts[aid][outcome])
# 			gain_out+=min(cnt,need)
# 	score=(gain+lambda_out*gain_out)/(1.0+size**0.5)
# 	return score

# def _apply_scf(scf):
# 	global current_global_test_count
# 	global_test_scaffolds.add(scf)
# 	for aid,cnt in scf_to_aid_counts[scf].items():
# 		aid_current_test_counts[aid]+=cnt
# 	for aid,od in scf_to_aid_out_counts[scf].items():
# 		for outcome,cnt in od.items():
# 			aid_out_current_test_counts[aid][outcome]+=cnt
# 	current_global_test_count+=scf_size_total[scf]
# 	unselected.discard(scf)

# def _all_lower_satisfied():
# 	for aid in aid_total_counts:
# 		if aid_total_counts[aid]<=0:
# 			continue
# 		if aid_current_test_counts[aid]<aid_lower_target[aid]:
# 			return False
# 	for aid in aid_out_lower_target:
# 		for outcome,lo in aid_out_lower_target[aid].items():
# 			if aid_out_total_counts[aid].get(outcome,0)<=0:
# 				continue
# 			if aid_out_current_test_counts[aid][outcome]<lo:
# 				return False
# 	return True

# max_iters=len(all_scaffolds)
# for _ in range(max_iters):
# 	if _all_lower_satisfied():
# 		break
# 	best_scf=None
# 	best_score=None
# 	for scf in list(unselected):
# 		s=_score_scf(scf)
# 		if s is None:
# 			continue
# 		if best_score is None or s>best_score:
# 			best_score=s
# 			best_scf=scf
# 	if best_scf is None:
# 		break
# 	_apply_scf(best_scf)

# final_ratios=[aid_current_test_counts[aid]/aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid]>0]
# print(f"global_test_ratio: {current_global_test_count/total_compounds:.4f}")
# print(f"aid_test_ratio_mean: {sum(final_ratios)/len(final_ratios):.4f}")
# print(f"aid_test_ratio_min: {min(final_ratios):.4f}")
# print(f"aid_test_ratio_max: {max(final_ratios):.4f}")

# out_ratios=[]
# for aid in aid_out_total_counts:
# 	for outcome,n in aid_out_total_counts[aid].items():
# 		if n<=0:
# 			continue
# 		out_ratios.append(aid_out_current_test_counts[aid][outcome]/n)
# if len(out_ratios)>0:
# 	print(f"aid_out_test_ratio_mean: {sum(out_ratios)/len(out_ratios):.4f}")
# 	print(f"aid_out_test_ratio_min: {min(out_ratios):.4f}")
# 	print(f"aid_out_test_ratio_max: {max(out_ratios):.4f}")


## 최종 최종

import random
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

random.seed(42)

# 0. cid_to_aids 만들기 (CID -> 이 CID를 가진 모든 AID 집합)
cid_to_aids = defaultdict(set)
for aid, outcome_dict in result_dict.items():
    for outcome, cids in outcome_dict.items():
        for c in cids:
            cid_to_aids[c].add(aid)


# 0. AID별 전체 CID 목록/개수 미리 계산
aid_all_cids = {}
aid_total = {}

for aid, outcome_dict in result_dict.items():
    cids = []
    for outcome, cs in outcome_dict.items():
        cids.extend(cs)
    # 중복 제거 + 순서 보존
    cids = list(dict.fromkeys(cids))
    aid_all_cids[aid] = cids
    aid_total[aid] = len(cids)


# global_random_test_cids = set()
# cid_to_aids = defaultdict(list)

# # 각 CID가 속한 AID 리스트 미리 확보
# for aid, outcomes in result_dict.items():
#     for outcome, cids in outcomes.items():
#         for cid in cids:
#             cid_to_aids[cid].append(aid)

# all_cids = list(cid_to_aids.keys())
# random.shuffle(all_cids)

# aid_current_random_counts = defaultdict(int)

# for cid in all_cids:
#     useful = False
#     for aid in cid_to_aids[cid]:
#         if aid_current_random_counts[aid] < aid_test_targets[aid]:
#             useful = True
#             break
    
#     if useful:
#         global_random_test_cids.add(cid)
#         for aid in cid_to_aids[cid]:
#             aid_current_random_counts[aid] += 1




# 1. AID별 테스트 타깃 & 최대 테스트 허용 개수 설정
target_ratio = 0.10            # 원래 쓰던 비율
min_train_ratio = 0.80         # 최소 20%는 train으로 남기고 싶다 (원하는 값으로 조정)
min_train_abs = 0              # 각 AID는 최소 1개는 train에 남기기

aid_test_targets = {}
aid_min_train = {}
aid_max_test = {}

for aid, n in aid_total.items():
    # 최소 남겨야 할 train 개수
    min_train = max(min_train_abs, int(n * min_train_ratio))
    min_train = min(min_train, n)  # n보다 클 수는 없음
    aid_min_train[aid] = min_train
    # 이 AID에서 최대 몇 개까지 test로 보낼 수 있는지
    max_test = max(0, n - min_train)
    aid_max_test[aid] = max_test
    # 원래의 타깃 테스트 개수
    t = int(n * target_ratio)
    # 타깃이 최대 허용치보다 크면 줄임
    aid_test_targets[aid] = min(t, max_test)

# 2. 전역 상태
global_random_test_cids = set()
aid_current_random_counts = defaultdict(int)


random.seed(42)

# 3. AID 순서를 섞어서 공평하게
all_aids = list(result_dict.keys())
random.shuffle(all_aids)

for aid in tqdm(all_aids, desc="Balanced Splitting"):
    target = aid_test_targets[aid]
    if target <= 0:
        continue
    current = aid_current_random_counts[aid]
    if current >= target:
        continue
    needed = target - current
    if needed <= 0:
        continue
    # 이 AID가 가진 CID들 중에서 아직 test에 안 들어간 것들
    candidates = []
    for c in aid_all_cids[aid]:
        if c in global_random_test_cids:
            continue
        # 이 CID를 test로 뽑으면 연결된 모든 AID의 test 개수가
        # 해당 AID의 최대 허용치(aid_max_test)를 넘지 않는지 확인
        safe = True
        for related_aid in cid_to_aids[c]:
            # 이 related_aid가 허용 가능한 최대 test 개수에 이미 도달했다면,
            # 이 CID를 더 이상 test로 보낼 수 없음 (그 AID의 train이 사라질 수 있음)
            if aid_current_random_counts[related_aid] >= aid_max_test[related_aid]:
                safe = False
                break
        if safe:
            candidates.append(c)
    if not candidates:
        # 이 AID는 더 이상 안전하게 test를 뽑을 수 없음
        continue
    k = min(needed, len(candidates))
    selected = random.sample(candidates, k)
    for c in selected:
        global_random_test_cids.add(c)
        for related_aid in cid_to_aids[c]:
            aid_current_random_counts[related_aid] += 1


## 최종 random

global_random_test_cids = set()
aid_current_random_counts = defaultdict(int)

# 1. AID 리스트를 섞어서 순회 (편향 방지)
all_aids = list(result_dict.keys())
random.shuffle(all_aids)

for aid in tqdm(all_aids, desc="Balanced Splitting"):
    # 현재 AID가 이미 다른 CID들에 의해 10%가 채워졌는지 확인
    target = aid_test_targets[aid]
    current = aid_current_random_counts[aid]
    
    if current < target:
        # 2. 아직 부족하다면, 이 AID에 속한 전체 CID 중 '아직 선택 안 된' 것들 확보
        needed_count = int(target - current)
        
        # 이 AID의 모든 CID 리스트 추출
        aid_all_cids = []
        for outcome, cids in result_dict[aid].items():
            aid_all_cids.extend(cids)
        
        # 아직 테스트셋에 포함되지 않은 후보들 추출
        candidates = [c for c in aid_all_cids if c not in global_random_test_cids]
        
        if len(candidates) >= needed_count:
            selected = random.sample(candidates, needed_count)
        else:
            selected = candidates # 후보가 부족하면 남은 거라도 다 넣음
            
        # 3. 선택된 CID들을 전역 테스트셋에 추가하고, 연관된 모든 AID의 카운트 업데이트
        for c in selected:
            global_random_test_cids.add(c)
            # 이 CID가 속한 다른 모든 AID들의 카운트도 올려줘야 함 (중요!)
            for related_aid in cid_to_aids[c]:
                aid_current_random_counts[related_aid] += 1

### 

final_ratios=[aid_current_random_counts[aid]/aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid]>0]


## random 최종

global_random_test_cids = set()
aid_current_random_counts = defaultdict(int)

# 1. AID 리스트를 섞어서 순회 (편향 방지)
all_aids = list(result_dict.keys())
random.shuffle(all_aids)

for aid in tqdm(all_aids, desc="Balanced Splitting"):
    # 현재 AID가 이미 다른 CID들에 의해 10%가 채워졌는지 확인
    target = aid_test_targets[aid]
    current = aid_current_random_counts[aid]
    
    if current < target:
        # 2. 아직 부족하다면, 이 AID에 속한 전체 CID 중 '아직 선택 안 된' 것들 확보
        needed_count = int(target - current)
        
        # 이 AID의 모든 CID 리스트 추출
        aid_all_cids = []
        for outcome, cids in result_dict[aid].items():
            aid_all_cids.extend(cids)
        
        # 아직 테스트셋에 포함되지 않은 후보들 추출
        candidates = [c for c in aid_all_cids if c not in global_random_test_cids]
        
        if len(candidates) >= needed_count:
            selected = random.sample(candidates, needed_count)
        else:
            selected = candidates # 후보가 부족하면 남은 거라도 다 넣음
            
        # 3. 선택된 CID들을 전역 테스트셋에 추가하고, 연관된 모든 AID의 카운트 업데이트
        for c in selected:
            global_random_test_cids.add(c)
            # 이 CID가 속한 다른 모든 AID들의 카운트도 올려줘야 함 (중요!)
            for related_aid in cid_to_aids[c]:
                aid_current_random_counts[related_aid] += 1

## 최종?import random
import random
from collections import defaultdict
from tqdm import tqdm
random.seed(42)


target_ratio=0.10
upper_ratio=0.12
batch_cap=300
max_passes=25
deg_stages=[1,2,4,8,16,32,64,10**9]
global_test_cids=set()
all_aids=list(result_dict.keys())
aids_rev=list(reversed(all_aids))
aid_list_cids={}
aid_sets={}
aid_total={}
for aid in all_aids:
	cids=[]
	for outcome,cs in result_dict[aid].items():
		cids.extend(cs)
	cids=list(dict.fromkeys(cids))
	cids.sort(key=lambda c: len(set(cid_to_aids.get(c,[]))))
	aid_list_cids[aid]=cids
	aid_sets[aid]=set(cids)
	aid_total[aid]=len(cids)
     

aid_target={aid:int(aid_total[aid]*target_ratio) for aid in all_aids}
aid_upper={aid:max(aid_target[aid],int(aid_total[aid]*upper_ratio)) for aid in all_aids}
aid_ptr=defaultdict(int)
aid_test_hit=defaultdict(set)


def hit_count(aid):
	return len(aid_test_hit[aid])


def need(aid):
	return aid_target[aid]-hit_count(aid)


def can_add(aid):
	return hit_count(aid)<aid_upper[aid]


pbar=tqdm(total=sum(aid_target.values()),desc="Tail-first + low-degree-first (hitset-based)",mininterval=1.0)
for deg_cap in deg_stages:
	for _ in range(max_passes):
		progress=False
		for aid in aids_rev:
			if aid_total[aid]==0:
				continue
			if not can_add(aid):
				continue
			rem=need(aid)
			if rem<=0:
				continue
			want=min(batch_cap,rem,aid_upper[aid]-hit_count(aid))
			if want<=0:
				continue
			selected=[]
			ptr=aid_ptr[aid]
			cids=aid_list_cids[aid]
			n=len(cids)
			while want>0 and ptr<n:
				c=cids[ptr]
				if len(set(cid_to_aids.get(c,[])))>deg_cap:
					break
				ptr+=1
				if c in global_test_cids:
					continue
				selected.append(c)
				want-=1
			aid_ptr[aid]=ptr
			if len(selected)==0:
				continue
			for c in selected:
				global_test_cids.add(c)
				for ra in set(cid_to_aids.get(c,[])):
					if c in aid_sets.get(ra,set()):
						if len(aid_test_hit[ra])<aid_upper.get(ra,0):
							aid_test_hit[ra].add(c)
			pbar.update(len(selected))
			progress=True
		if not progress:
			break
          


pbar.close()
ratios=[]
zeros=[]
for aid in all_aids:
	if aid_total[aid]<=0:
		continue
	r=len(aid_test_hit[aid])/aid_total[aid]
	ratios.append(r)
	if len(aid_test_hit[aid])==0:
		zeros.append(aid)
          


print("global_test_size",len(global_test_cids))
print("aid_ratio_mean",sum(ratios)/len(ratios))
print("aid_ratio_min",min(ratios))
print("aid_ratio_max",max(ratios))
print("zero_aids",len(zeros))


### 최종 

import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

# --- 0. 환경 설정 및 하이퍼파라미터 ---
random.seed(42)
tau = 0.10             # 목표 테스트 비율 (10%)
aid_lower = 0.08       # AID별 최소 허용 비율
aid_upper = 0.12       # AID별 최대 허용 비율
global_buffer = 1.15   # 전체 테스트셋 크기 상한 (tau * 1.15)
lambda_out = 1.0       # Outcome 일관성 가중치

# --- 1. 데이터 구조 최적화 (Pre-processing) ---
print("Building data maps...")
scf_to_aid_out_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
scf_to_aid_counts = defaultdict(lambda: defaultdict(int))
scf_size_total = defaultdict(int)
aid_total_counts = defaultdict(int)
aid_out_total_counts = defaultdict(lambda: defaultdict(int))
aid_to_scfs = defaultdict(set) # 역색인: AID가 가진 스캐폴드들

for aid, outcomes in tqdm(result_dict.items(), desc="Mapping result_dict"):
    for outcome, cids in outcomes.items():
        for cid in cids:
            scf = cid_to_scf.get(cid)
            # 스캐폴드 없는 경우 처리 (Independent Scaffold 취급)
            if scf is None or scf == 'Unknown' or scf == '':
                scf = f"NOSCF_{cid}"
                
            scf_to_aid_out_counts[scf][aid][outcome] += 1
            scf_to_aid_counts[scf][aid] += 1
            scf_size_total[scf] += 1
            aid_total_counts[aid] += 1
            aid_out_total_counts[aid][outcome] += 1
            aid_to_scfs[aid].add(scf)

all_scaffolds = list(scf_to_aid_counts.keys())
unselected = set(all_scaffolds)
global_test_scaffolds = set()

# 타겟 설정
aid_target = {aid: n * tau for aid, n in aid_total_counts.items()}
aid_lower_target = {aid: n * aid_lower for aid, n in aid_total_counts.items()}
aid_upper_target = {aid: n * aid_upper for aid, n in aid_total_counts.items()}

# 변수 초기화
aid_current_test_counts = defaultdict(int)
aid_out_current_test_counts = defaultdict(lambda: defaultdict(int))
total_compounds = sum(aid_total_counts.values())
global_test_limit = total_compounds * tau * global_buffer
current_global_test_count = 0

# --- 헬퍼 함수 정의 ---
def _apply_scf(scf):
    global current_global_test_count
    global_test_scaffolds.add(scf)
    for aid, cnt in scf_to_aid_counts[scf].items():
        aid_current_test_counts[aid] += cnt
        for outcome, o_cnt in scf_to_aid_out_counts[scf][aid].items():
            aid_out_current_test_counts[aid][outcome] += o_cnt
    current_global_test_count += scf_size_total[scf]
    unselected.discard(scf)

def _score_scf(scf):
    size = scf_size_total[scf]
    if current_global_test_count + size > global_test_limit: return None
    
    gain = 0.0
    overflow = False
    for aid, cnt in scf_to_aid_counts[scf].items():
        # 상한선 체크
        if aid_current_test_counts[aid] + cnt > aid_upper_target[aid]:
            overflow = True; break
        # 가중치 계산 (부족한 곳을 채울수록 높은 점수)
        need = max(0.0, aid_target[aid] - aid_current_test_counts[aid])
        gain += min(cnt, need)
    
    if overflow or gain <= 0: return None
    return gain / (1.0 + size**0.5) # 사이즈가 작을수록 우선순위 (효율적 채우기)

# --- Stage 1: Fast Bulk Pass (O(S)) ---
print("\nStage 1: Running Bulk Pass...")
random.shuffle(all_scaffolds)
for scf in tqdm(all_scaffolds, desc="Bulk filling"):
    size = scf_size_total[scf]
    if current_global_test_count + size > global_test_limit * 0.9: continue # 90%까지만 채움
    
    can_add = True
    for aid, cnt in scf_to_aid_counts[scf].items():
        # 매우 보수적인 기준: 타겟(10%)을 조금이라도 넘기면 패스
        if aid_current_test_counts[aid] + cnt > aid_target[aid]:
            can_add = False; break
    if can_add: _apply_scf(scf)

# --- Stage 2: Deficit-Driven Precision (AID 중심) ---
print("\nStage 2: Precision Tuning for needy AIDs...")
def get_needy_aids():
    return [aid for aid, cnt in aid_current_test_counts.items() 
            if cnt < aid_lower_target[aid] and aid_total_counts[aid] > 0]

for _ in range(5): # 최대 5번 반복하며 미세 조정
    needy_aids = get_needy_aids()
    if not needy_aids: break
    random.shuffle(needy_aids)
    
    for aid in tqdm(needy_aids, desc=f"Iteration {_ + 1}"):
        candidates = aid_to_scfs[aid] & unselected
        if not candidates: continue
        
        best_scf, best_score = None, -1
        for scf in candidates:
            s = _score_scf(scf)
            if s and s > best_score:
                best_score, best_scf = s, scf
        if best_scf: _apply_scf(best_scf)

# --- 2. 결과 리포트 ---
print("\n" + "="*30)
final_ratios = [aid_current_test_counts[aid]/aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid] > 0]
print(f"Global Test Ratio: {current_global_test_count/total_compounds:.4f}")
print(f"AID Ratio Mean: {np.mean(final_ratios):.4f} (Min: {np.min(final_ratios):.4f}, Max: {np.max(final_ratios):.4f})")
print(f"Total Test Scaffolds: {len(global_test_scaffolds)}")
print("="*30)



rrr =fbiodf2[fbiodf2.CID.isin(global_random_test_cids)]
# rrr.to_pickle(f'{adir}/test_random_biodf.pkl')

# fbiodf2.to_pickle(f'{adir}/fbiodf2.pkl')
sub_f = final_scf_df[final_scf_df.scaffold.isin(global_test_scaffolds)]
fff =fbiodf2[fbiodf2.CID.isin(sub_f.index)]
# fff.to_pickle(f'{adir}/test_biodf.pkl')

fbiodf2= pd.read_pickle(f'{adir}/fbiodf2.pkl')

ap = pd.read_pickle(f'{adir}/pubchem_ap_opt.pkl')

ap = pd.read_pickle(f'{adir}/pubchem_fp_opt.pkl')

# fff = pd.read_pickle(f'{adir}/test_biodf.pkl')

# traindf = fbiodf2[~fbiodf2.index.isin(fff.index)]

testdf = pd.read_pickle(f'{adir}/test_random_biodf.pkl')

testdf = pd.read_pickle(f'{adir}/test_biodf.pkl')

# ap = ap[~ap.index.duplicated()]


traindf = fbiodf2[~fbiodf2.CID.isin(testdf.CID)]



# 1. AID와 Activity Outcome별로 CID를 리스트로 묶기
cid_lists = traindf.groupby(['AID', 'Activity Outcome'])['CID'].apply(list)

all_cid_lists = fbiodf2.loc[:,['AID','CID']].drop_duplicates().groupby(['AID'])['CID'].apply(list)






unique_aids = traindf['AID'].unique()
total_aids = len(unique_aids)


from multiprocessing import Pool, cpu_count
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

# ---------- helpers ----------
def mahalanobis_to_group(X,G,reg_lambda=1e-3):
    n_g,d=G.shape
    if n_g<2:return np.full(X.shape[0],np.nan,dtype="float32")
    lw=LedoitWolf().fit(G)
    cov=lw.covariance_.astype("float32")+reg_lambda*np.eye(d,dtype="float32")
    cov_inv=np.linalg.inv(cov)
    mu=G.mean(axis=0,keepdims=True)
    diff=X-mu
    left=diff@cov_inv
    d2=(left*diff).sum(axis=1)
    d2[d2<0]=0.0
    return np.sqrt(d2).astype("float32")

def knn_score_to_group(X,G,k=5,metric="cosine",alpha=5.0):
    n_g=G.shape[0]
    if n_g==0:return np.full(X.shape[0],np.nan,dtype="float32")
    k_eff=min(k,n_g)
    nn=NearestNeighbors(n_neighbors=k_eff,metric=metric,n_jobs=1)
    nn.fit(G)
    dists,_=nn.kneighbors(X,return_distance=True)
    return np.exp(-alpha*dists.mean(axis=1)).astype("float32")

# ---------- globals for workers ----------
UNIQUE_AIDS=TRAIN_DF=AP=CID_LISTS=ALL_CID_LISTS=None
def init_worker(unique_aids,traindf,ap,cid_lists,all_cid_lists):
    global UNIQUE_AIDS,TRAIN_DF,AP,CID_LISTS,ALL_CID_LISTS
    UNIQUE_AIDS=unique_aids;TRAIN_DF=traindf;AP=ap;CID_LISTS=cid_lists;ALL_CID_LISTS=all_cid_lists

unique_aids=[1794731, 1794732, 1794733, 1794735, 1794736, 1794738, 1794739, 1794740, 1794742, 1794745, 1794746, 1794748, 1794750, 1794751, 1794752, 1794753, 1794754, 1794755, 1794756, 1794757, 1794758, 1794759, 1794760, 1794761, 1794763, 1794764, 1794765, 1794766, 1794767, 1794768, 1794769, 1794771, 1794772, 1794774, 1794775, 1794776, 1794777, 1794778, 1794779, 1794780, 1794782, 1794783, 1794784, 1794785, 1794786, 1794787, 1794788, 1794789, 1794790, 1794792, 1794793, 1794794, 1794795, 1794796, 1794798, 1794799, 1794800, 2202374]
aass = []
for ai in range(len(unique_aids)):
    aass.append(process_stage1_aid(ai))
    print(aid)


def process_stage1_aid(aid_index):
    storage_chunk=[]
    aid=unique_aids[aid_index]
    subs=cid_lists[aid]
    for outcome in subs.index:
        cids=subs[outcome];valid=ap.index.intersection(cids)
        if len(valid)==0:continue
        sap=ap.loc[valid]
        row={'AID':aid,'Activity Outcome':outcome,'Count':len(valid),'ap_std_mean':sap.std().mean()}
        row.update(sap.mean().to_dict());storage_chunk.append(row)
    return storage_chunk

##############
## Stage 1 & 2 Workers
##############
# ---------- stage1 worker ----------
def process_stage1_aid(aid_index):
    storage_chunk=[];sample_chunk=[]
    aid=UNIQUE_AIDS[aid_index]
    if aid not in CID_LISTS:return storage_chunk,sample_chunk
    subs=CID_LISTS[aid];tap=ALL_CID_LISTS[aid]
    for outcome in subs.index:
        cids=subs[outcome];valid=AP.index.intersection(cids)
        if len(valid)==0:continue
        sap=AP.loc[valid]
        row={'AID':aid,'Activity Outcome':outcome,'Count':len(valid),'ap_std_mean':sap.std().mean()}
        row.update(sap.mean().to_dict());storage_chunk.append(row)
    if isinstance(tap,pd.DataFrame):
        X=tap.to_numpy(dtype="float32");tap_cids=tap.index.values
    else:
        X=np.asarray(tap,dtype="float32");tap_cids=np.arange(X.shape[0])
    aid_rows=TRAIN_DF[TRAIN_DF['AID']==aid]
    act=aid_rows[aid_rows['Activity Outcome']=='active']['CID'].unique()
    ina=aid_rows[aid_rows['Activity Outcome']=='inactive']['CID'].unique()
    G_act=AP.loc[AP.index.intersection(act)].to_numpy(dtype="float32")
    G_in=AP.loc[AP.index.intersection(ina)].to_numpy(dtype="float32")
    ma=mahalanobis_to_group(X,G_act);mi=mahalanobis_to_group(X,G_in)
    ka=knn_score_to_group(X,G_act);ki=knn_score_to_group(X,G_in)
    for cid,a,b,c,d in zip(tap_cids,ma,mi,ka,ki):
        sample_chunk.append({'AID':aid,'CID':cid,'mahal_active':a,'mahal_inactive':b,'knn_active':c,'knn_inactive':d})
    return storage_chunk,sample_chunk

# ---------- stage2 worker ----------
def process_aid_range(args):
    start,end=args;storage=[];samples=[]
    aids=UNIQUE_AIDS[start:end]
    chunk=TRAIN_DF[TRAIN_DF['AID'].isin(aids)]
    if chunk.empty:return storage,samples
    counts=chunk.groupby(['AID','Activity Outcome']).size()
    grp=chunk.join(AP,on='CID').groupby(['AID','Activity Outcome']).agg(['mean','std'])
    if not grp.empty:
        num=grp.select_dtypes(include=[np.number])
        means=num.xs('mean',level=1,axis=1).iloc[:,1:(len(AP.columns)+1)]
        stdm=num.xs('std',level=1,axis=1).iloc[:,1:(len(AP.columns)+1)].mean(axis=1).fillna('-')
        for (aid,o),vals in means.iterrows():
            row={'AID':aid,'Activity Outcome':o,'Count':counts.get((aid,o),0),'ap_std_mean':stdm.loc[(aid,o)]}
            row.update(vals.to_dict());storage.append(row)
    for aid,df in chunk.groupby('AID'):
        cids=df['CID'].unique();valid=AP.index.intersection(cids)
        if len(valid)==0:continue
        X=AP.loc[valid].to_numpy(dtype="float32")
        act=df[df['Activity Outcome']=='active']['CID'].unique()
        ina=df[df['Activity Outcome']=='inactive']['CID'].unique()
        G_act=AP.loc[AP.index.intersection(act)].to_numpy(dtype="float32")
        G_in=AP.loc[AP.index.intersection(ina)].to_numpy(dtype="float32")
        ma=mahalanobis_to_group(X,G_act);mi=mahalanobis_to_group(X,G_in)
        ka=knn_score_to_group(X,G_act);ki=knn_score_to_group(X,G_in)
        for cid,a,b,c,d in zip(valid,ma,mi,ka,ki):
            samples.append({'AID':aid,'CID':cid,'mahal_active':a,'mahal_inactive':b,'knn_active':c,'knn_inactive':d})
    return storage,samples


### mab만 다시 


# ---------- stage1 worker ----------
def process_stage1_aid(aid_index):
    storage_chunk=[]
    aid=unique_aids[aid_index]
    subs=cid_lists[aid];tcds=all_cid_lists[aid]
    X = ap.loc[tcds].to_numpy(dtype="float32")
    for outcome in subs.index:
        cids=subs[outcome];valid=ap.index.intersection(cids)
        if len(valid)==0:continue
        sap=ap.loc[valid].to_numpy(dtype="float32")
        ma=mahalanobis_to_group(X,sap)
        ka=knn_score_to_group(X,sap)
        row = {'AID': [aid]*len(ma), 'CID': tcds}
        row[f'{outcome}_mahal_mean']=ma
        row[f'{outcome}_knn_mean']=ka
    storage_chunk.append(row)
    return storage_chunk


import os
import tempfile
import numpy as np
import joblib
from tqdm import tqdm
from multiprocessing import get_context, cpu_count

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

_AP_MAT=None
_CID2ROW=None
_AP_MAT_PATH=None
_CID2ROW_PATH=None
_UNIQUE_AIDS=None
_CID_LISTS=None
_ALL_CID_LISTS=None

def init_worker(ap_mat_path, cid2row_path, unique_aids, cid_lists, all_cid_lists):
	global _AP_MAT,_CID2ROW,_AP_MAT_PATH,_CID2ROW_PATH,_UNIQUE_AIDS,_CID_LISTS,_ALL_CID_LISTS
	_AP_MAT_PATH=ap_mat_path
	_CID2ROW_PATH=cid2row_path
	_UNIQUE_AIDS=unique_aids
	_CID_LISTS=cid_lists
	_ALL_CID_LISTS=all_cid_lists
	_AP_MAT=joblib.load(_AP_MAT_PATH,mmap_mode="r")
	_CID2ROW=joblib.load(_CID2ROW_PATH)

def _rows_from_cids(cids):
	idx=[]
	get=_CID2ROW.get
	for c in cids:
		j=get(c)
		if j is not None:
			idx.append(j)
	if len(idx)==0:
		return None
	return np.asarray(idx,dtype=np.int64)

def process_stage1_aid_memmap(aid_index):
	aid=_UNIQUE_AIDS[aid_index]
	subs=_CID_LISTS[aid]
	tcds=_ALL_CID_LISTS[aid]
	t_idx=_rows_from_cids(tcds)
	if t_idx is None:
		return None
	X=_AP_MAT[t_idx]
	out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
	for outcome in subs.index:
		cids=subs[outcome]
		s_idx=_rows_from_cids(cids)
		if s_idx is None:
			continue
		sap=_AP_MAT[s_idx]
		ma=mahalanobis_to_group(X,sap)
		ka=knn_score_to_group(X,sap)
		out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
		out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)
	return out


def process_stage1_aid_memmap(aid_index):
	aid=_UNIQUE_AIDS[aid_index]
	subs=_CID_LISTS[aid]
	tcds=_ALL_CID_LISTS[aid]
	t_idx=_rows_from_cids(tcds)
	if t_idx is None:
		return None
	X=_AP_MAT[t_idx].astype(np.float32)
	out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
	for outcome in subs.index:
		cids=subs[outcome]
		s_idx=_rows_from_cids(cids)
		if s_idx is None:
			continue
		sap=_AP_MAT[s_idx].astype(np.float32)
		ma=mahalanobis_to_group(X,sap)
		ka=knn_score_to_group(X,sap)
		out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
		out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)
	return out



def process_aid_range(args):
	start,end=args
	out=[]
	for i in range(start,end):
		r=process_stage1_aid_memmap(i)
		if r is not None:
			out.append(r)
	return out

def run_stage1_pool_memmap(ap, unique_aids, cid_lists, all_cid_lists, n_proc=None, chunk_size=1000, maxtasksperchild=200, tmp_dir=None, ctx_method="fork"):
	if tmp_dir is None:
		tmp_dir=tempfile.mkdtemp(prefix="stage1_memmap_")
	ap_mat_path=os.path.join(tmp_dir,"ap_mat.joblib")
	cid2row_path=os.path.join(tmp_dir,"cid2row.joblib")
	ap_mat=ap.to_numpy(dtype=np.float32,copy=True)
	joblib.dump(ap_mat,ap_mat_path,compress=0)
	cid2row={cid:i for i,cid in enumerate(ap.index.to_list())}
	joblib.dump(cid2row,cid2row_path,compress=3)
	if n_proc is None:
		n_proc=cpu_count()
	ctx=get_context(ctx_method)
	tasks=[(i,min(i+chunk_size,len(unique_aids))) for i in range(0,len(unique_aids),chunk_size)]
	results=[]
	with ctx.Pool(processes=n_proc,initializer=init_worker,initargs=(ap_mat_path,cid2row_path,unique_aids,cid_lists,all_cid_lists),maxtasksperchild=maxtasksperchild) as pool:
		it=pool.imap_unordered(process_aid_range,tasks,chunksize=1)
		for batch in tqdm(it,total=len(tasks),desc="stage1",mininterval=1.0):
			if batch:
				results.extend(batch)
	return results


with open(f'{adir}/unique_aids.pkl','wb') as f:
     pickle.dump(unique_aids, f)


with open(f'{adir}/unique_aids.pkl','rb') as f:
    unique_aids = pickle.load(f)
        
cid_lists.to_pickle(f'{adir}/cid_lists.pkl')
all_cid_lists.to_pickle(f'{adir}/all_cid_lists.pkl')


cid_lists = pd.read_pickle(f'{adir}/cid_lists.pkl')
all_cid_lists = pd.read_pickle(f'{adir}/all_cid_lists.pkl')

# tmp_dir=tempfile.mkdtemp(prefix="stage1_memmap_")
ap_mat_path=os.path.join(adir,"ap_mat.joblib")
cid2row_path=os.path.join(adir,"cid2row.joblib")
ap_mat=ap.to_numpy(dtype=np.float32,copy=True)



ap_mat_path=os.path.join(adir,"fp_mat.joblib")
cid2row_path=os.path.join(adir,"fp_cid2row.joblib")
ap_mat=ap.to_numpy(dtype=np.unit8,copy=True)


joblib.dump(ap_mat,ap_mat_path,compress=0)
cid2row={cid:i for i,cid in enumerate(ap.index.to_list())}
joblib.dump(cid2row,cid2row_path,compress=3)


# resdf = pd.concat([pd.DataFrame(res) for res in results])

# resdf.to_pickle(f'{adir}/fp_cent_mean2.pkl')
resdf.to_pickle(f'{adir}/fp_cent_mean3.pkl')

unique_aids= np.array(list(a.iloc[113240:-7].sample(frac=1).index))


n_proc=len(unique_aids)
chunk_size=1
maxtasksperchild=200
ctx_method="fork"
ctx=get_context(ctx_method)

# st = time.time()
# joblib.dump(ap_mat,ap_mat_path,compress=0)


# st = time.time()
# joblib.load(ap_mat_path,mmap_mode="r")
# print(time.time()-st)
les2=[len(all_cid_lists[aid]) for aid in unique_aids]
aid = unique_aids[les2.index(427972)]
[398859, 398891, 427972, 459289, 544814, 613288, 614711, 615362, 615686]


apap=joblib.load(ap_mat_path,mmap_mode="r")
cid2row = joblib.load(cid2row_path)



ff = pd.concat([pd.DataFrame(res) for res in results])
fresdf = pd.concat([fresdf, ff])
fresdf.to_pickle(f'{adir}/fp_cent_mean.pkl')

def _rows_from_cids(cids):
	idx=[]
	get=cid2row.get
	for c in cids:
		j=get(c)
		if j is not None:
			idx.append(j)
	if len(idx)==0:
		return None
	return np.asarray(idx,dtype=np.int64)


aid= 1508602
aid= 1259374
aid = 1259310
aid = 1259422
aid = 1671190
aid = 1347041
aid = 1272365

subs=cid_lists[aid]
tcds=all_cid_lists[aid]
t_idx=_rows_from_cids(tcds)
X=apap[t_idx].astype(np.float32)
out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
st = time.time()
for outcome in subs.index:
    print(outcome)
    cids=subs[outcome]
    s_idx=_rows_from_cids(cids)
    sap=apap[s_idx].astype(np.float32)
    ma=mahalanobis_to_group(X,sap)
    print(f'{time.time()-st}')
    ka=knn_score_to_group(X,sap)
    print(f'{time.time()-st}')
    out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
    out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)


print(f'{time.time()-st}')
unique_aids= np.array(list(a.iloc[113240:-5].sort_values(0, ascending=False).index))


tasks=[(i,min(i+chunk_size,len(unique_aids))) for i in range(0,len(unique_aids),chunk_size)]
results=[]
with ctx.Pool(processes=n_proc,initializer=init_worker,initargs=(ap_mat_path,cid2row_path,unique_aids,cid_lists,all_cid_lists),maxtasksperchild=maxtasksperchild) as pool:
    it=pool.imap_unordered(process_aid_range,tasks,chunksize=1)
    for batch in tqdm(it,total=len(tasks),desc="stage1",mininterval=1.0):
        if batch:
            results.extend(batch)




resdf2.to_pickle(f'{adir}/fp_cent_mean.pkl')
unique_aids = list(set(unique_aids)-set(resdf2.AID))
unique_aids = np.array(unique_aids)

# 실행 예시
stage1_results = run_stage1_pool_memmap(
    ap=ap,
    unique_aids=unique_aids,
    cid_lists=cid_lists,
    all_cid_lists=all_cid_lists,
    n_proc=min(cpu_count(), 64),
    chunk_size=1000,
    maxtasksperchild=200,
    tmp_dir=None,
    ctx_method="fork"
)



def _rows_from_cids(cids):
	idx=[]
	get=cid2row.get
	for c in cids:
		j=get(c)
		if j is not None:
			idx.append(j)
	if len(idx)==0:
		return None
	return np.asarray(idx,dtype=np.int64)


def process_stage1_aid_memmap(aid_index):
	aid=unique_aids[aid_index]
	subs=cid_lists[aid]
	tcds=all_cid_lists[aid]
	t_idx=_rows_from_cids(tcds)
	X=ap[t_idx]
	out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
	for outcome in subs.index:
		cids=subs[outcome]
		s_idx=_rows_from_cids(cids)
		if s_idx is None:
			continue
		sap=ap[s_idx]
		ma=mahalanobis_to_group(X,sap)
		ka=knn_score_to_group(X,sap)
		out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
		out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)
	return out


st = time.time()
aa= process_stage1_aid_memmap(0)
print(time.time()-st)

## 여기까지 mab

# ---------- stage1 worker ----------
def process_stage1_aid(aid_index):
    storage_chunk=[]
    aid=UNIQUE_AIDS[aid_index]
    X = AP.index.intersection(tap).to_numpy(dtype="float32")
    subs=CID_LISTS[aid];tap=ALL_CID_LISTS[aid]
    row = {'AID': aid}
    for outcome in subs.index:
        cids=subs[outcome];valid=AP.index.intersection(cids)
        if len(valid)==0:continue
        sap=AP.loc[valid].to_numpy(dtype="float32")
        ma=mahalanobis_to_group(X,sap)
        ka=knn_score_to_group(X,sap)
        row[f'{outcome}_mahal_mean']=np.nanmean(ma)
        row[f'{outcome}_knn_mean']=np.nanmean(ka)
    storage_chunk.append(row)
    return storage_chunk

# ---------- stage2 worker ----------
def process_aid_range(args):
    start,end=args;storage=[];samples=[]
    aids=UNIQUE_AIDS[start:end]
    chunk=TRAIN_DF[TRAIN_DF['AID'].isin(aids)]
    if chunk.empty:return storage,samples
    counts=chunk.groupby(['AID','Activity Outcome']).size()
    grp=chunk.join(AP,on='CID').groupby(['AID','Activity Outcome']).agg(['mean','std'])
    if not grp.empty:
        num=grp.select_dtypes(include=[np.number])
        means=num.xs('mean',level=1,axis=1).iloc[:,1:(len(AP.columns)+1)]
        stdm=num.xs('std',level=1,axis=1).iloc[:,1:(len(AP.columns)+1)].mean(axis=1).fillna('-')
        for (aid,o),vals in means.iterrows():
            row={'AID':aid,'Activity Outcome':o,'Count':counts.get((aid,o),0),'ap_std_mean':stdm.loc[(aid,o)]}
            row.update(vals.to_dict());storage.append(row)
    for aid,df in chunk.groupby('AID'):
        cids=df['CID'].unique();valid=AP.index.intersection(cids)
        if len(valid)==0:continue
        X=AP.loc[valid].to_numpy(dtype="float32")
        act=df[df['Activity Outcome']=='active']['CID'].unique()
        ina=df[df['Activity Outcome']=='inactive']['CID'].unique()
        G_act=AP.loc[AP.index.intersection(act)].to_numpy(dtype="float32")
        G_in=AP.loc[AP.index.intersection(ina)].to_numpy(dtype="float32")
        ma=mahalanobis_to_group(X,G_act);mi=mahalanobis_to_group(X,G_in)
        ka=knn_score_to_group(X,G_act);ki=knn_score_to_group(X,G_in)
        for cid,a,b,c,d in zip(valid,ma,mi,ka,ki):
            samples.append({'AID':aid,'CID':cid,'mahal_active':a,'mahal_inactive':b,'knn_active':c,'knn_inactive':d})
    return storage,samples

# ---------- run ----------
total=len(unique_aids);n_stage1=min(3000,total)
storage_list=[];sample_scores=[]
with Pool(processes=cpu_count(),initializer=init_worker,initargs=(unique_aids,traindf,ap,cid_lists,all_cid_lists)) as pool:
    if n_stage1>0:
        for st,sm in tqdm(pool.imap_unordered(process_stage1_aid,range(n_stage1)),total=n_stage1,desc="Stage1"):
            storage_list.extend(st);sample_scores.extend(sm)
    if n_stage1<total:
        tasks=[(i,min(i+1000,total)) for i in range(n_stage1,total,1000)]
        for st,sm in tqdm(pool.imap_unordered(process_aid_range,tasks),total=len(tasks),desc="Stage2"):
            storage_list.extend(st);sample_scores.extend(sm)


df_group_stats=pd.DataFrame(storage_list)
df_sample_scores=pd.DataFrame(sample_scores)


from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

# --- 1. 워커 전역 변수 초기화 함수 ---
# 워커가 생성될 때 딱 한 번 실행되어 데이터를 메모리에 고정합니다.
def init_worker(shared_ap, shared_cid_lists):
    global AP, CID_LISTS
    AP = shared_ap
    CID_LISTS = shared_cid_lists

# --- 2. 워커 함수 (lll의 튜플 하나를 받음) ---
def process_worker(task):
    aid, outcome = task
    try:
        # 전역 변수 AP, CID_LISTS 사용
        subs = CID_LISTS[aid]
        cids = subs[outcome]
        
        valid = AP.index.intersection(cids)
        if len(valid) == 0:
            return None
        
        sap = AP.loc[valid]
        
        # 통계 계산
        res = {
            'AID': aid,
            'Activity Outcome': outcome,
            'Count': len(valid),
            'ap_std_mean': sap.std(ddof=1).mean() # ddof=1: n-1로 나눔
        }
        res.update(sap.mean().to_dict())
        return res
    except Exception:
        return None

# --- 3. 실행부 ---
if __name__ == '__main__':
    # lll은 이미 정의되어 있다고 가정 [(aid, outcome), ...]
    # AP, CID_LISTS도 전역에 존재한다고 가정
    
    print(f"Starting Pool with {cpu_count()} cores...")
    
    # chunksize: 한 워커에게 한 번에 던질 작업량 (너무 작으면 통신 오버헤드 발생)
    # 1.8M개 작업이므로 500~1000이 적당합니다.
c_size = 500 

adddf = []

with Pool(processes=cpu_count(), 
            initializer=init_worker, 
            initargs=(ap, cid_lists)) as pool:
    
    # imap_unordered는 결과를 나오는 대로 바로바로 반환해서 빠름
    for result in tqdm(pool.imap_unordered(process_worker, lll), 
                        total=len(lll), 
                        desc="Stage1 Global Parallel"):
        if result:
            adddf.append(result)

# 최종 데이터프레임 변환
collect_df = pd.DataFrame(adddf)


# i = 0
# storage_list = []
# current_chunk_size = 1000
# pbar = tqdm(total=total_aids, desc="Dynamic Chunk Processing")

# while i < total_aids:
#     # --- [Stage 1] 개별 처리 구간 (디버깅용) ---
#     if i < 3000:
#         aid = unique_aids[i]
#         if aid in cid_lists:
#             subs = cid_lists[aid]
#             for outcome in subs.index:
#                 cids = subs[outcome]
#                 valid_cids = ap.index.intersection(cids)
#                 if len(valid_cids) == 0: continue
                
#                 sap = ap.loc[valid_cids]
#                 ap_means = sap.mean()
#                 ap_std_mean = sap.std().mean()
                
#                 row = {
#                     'AID': aid,
#                     'Activity Outcome': outcome,
#                     'Count': len(valid_cids),
#                     'ap_std_mean': ap_std_mean
#                 }
#                 row.update(ap_means.to_dict())
#                 storage_list.append(row)
        
#         # [수정] 인덱스 증가 필수!
#         i += 1
#         pbar.update(1)
#     # --- [Stage 2] 대량 청크 처리 구간 ---
#     else:
#         end_idx = min(i + current_chunk_size, total_aids)
#         target_aids = unique_aids[i:end_idx]
        
#         chunk_bio = traindf[traindf['AID'].isin(target_aids)]
        
#         # [수정] Count를 미리 계산하여 속도 향상
#         counts = chunk_bio.groupby(['AID', 'Activity Outcome']).size()
        
#         # Join & Agg
#         grouped = chunk_bio.join(ap, on='CID').groupby(['AID', 'Activity Outcome']).agg(['mean', 'std'])
        
#         if not grouped.empty:
#             # [수정] iloc 대신 숫자형 컬럼만 자동 선택하도록 개선
#             numeric_grouped = grouped.select_dtypes(include=[np.number])
            
#             means_df = numeric_grouped.xs('mean', level=1, axis=1).iloc[:,1:551]
#             # [수정] NaN은 0으로 채우는 것이 안전함
#             ap_std_mean_series = numeric_grouped.xs('std', level=1, axis=1).iloc[:,1:551].mean(axis=1).fillna('-')
            
#             for (aid, outcome), mean_values in means_df.iterrows():
#                 row_dict = {
#                     'AID': aid,
#                     'Activity Outcome': outcome,
#                     'Count': counts.get((aid, outcome), 0), # 미리 구한 카운트 사용
#                     'ap_std_mean': ap_std_mean_series.loc[(aid, outcome)]
#                 }
#                 row_dict.update(mean_values.to_dict())
#                 storage_list.append(row_dict)
        
#         actual_processed = end_idx - i
#         pbar.update(actual_processed)
#         i = end_idx
#         del grouped

# pbar.close()




# with open(f'{adir}/aid_ap_stats.pkl', 'wb') as f:
#     pickle.dump(storage_list, f)



unique_aids = traindf['AID'].unique()
total_aids = len(unique_aids)




i = 0
storage_list = []
current_chunk_size = 1000
pbar = tqdm(total=total_aids, desc="Dynamic Chunk Processing")

while i < total_aids:
    # --- [Stage 1] 개별 처리 구간 (디버깅용) ---
    if i < 3000:
        aid = unique_aids[i]
        if aid in cid_lists:
            subs = cid_lists[aid]
            for outcome in subs.index:
                cids = subs[outcome]
                valid_cids = ap.index.intersection(cids)
                if len(valid_cids) == 0: continue
                
                sap = ap.loc[valid_cids]
                ap_means = sap.mean()
                ap_std_mean = sap.std().mean()
                
                row = {
                    'AID': aid,
                    'Activity Outcome': outcome,
                    'Count': len(valid_cids),
                    'ap_std_mean': ap_std_mean
                }
                row.update(ap_means.to_dict())
                storage_list.append(row)
        
        # [수정] 인덱스 증가 필수!
        i += 1
        pbar.update(1)
    # --- [Stage 2] 대량 청크 처리 구간 ---
    else:
        end_idx = min(i + current_chunk_size, total_aids)
        target_aids = unique_aids[i:end_idx]
        
        chunk_bio = traindf[traindf['AID'].isin(target_aids)]
        
        # [수정] Count를 미리 계산하여 속도 향상
        counts = chunk_bio.groupby(['AID', 'Activity Outcome']).size()
        
        # Join & Agg
        grouped = chunk_bio.join(ap, on='CID').groupby(['AID', 'Activity Outcome']).agg(['mean', 'std'])
        
        if not grouped.empty:
            # [수정] iloc 대신 숫자형 컬럼만 자동 선택하도록 개선
            numeric_grouped = grouped.select_dtypes(include=[np.number])
            
            means_df = numeric_grouped.xs('mean', level=1, axis=1).iloc[:,1:(len(ap.columns)+1)]
            # [수정] NaN은 0으로 채우는 것이 안전함
            ap_std_mean_series = numeric_grouped.xs('std', level=1, axis=1).iloc[:,1:(len(ap.columns)+1)].mean(axis=1).fillna('-')
            
            for (aid, outcome), mean_values in means_df.iterrows():
                row_dict = {
                    'AID': aid,
                    'Activity Outcome': outcome,
                    'Count': counts.get((aid, outcome), 0), # 미리 구한 카운트 사용
                    'ap_std_mean': ap_std_mean_series.loc[(aid, outcome)]
                }
                row_dict.update(mean_values.to_dict())
                storage_list.append(row_dict)
        
        actual_processed = end_idx - i
        pbar.update(actual_processed)
        i = end_idx
        del grouped

pbar.close()


with open(f'{adir}/aid_ap_stats.pkl', 'wb') as f:
    pickle.dump(storage_list, f)



import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

def mahalanobis_to_group(X: np.ndarray, G: np.ndarray, reg_lambda: float = 1e-3) -> np.ndarray:
    """
    X : (n_x, d)  - tap (CID profiles for this AID)
    G : (n_g, d)  - this AID의 active 또는 inactive group profile들
    """
    n_g, d = G.shape
    if n_g < 2:
        # 샘플이 너무 적으면 Mahalanobis 의미가 약해서 NaN 리턴
        return np.full(X.shape[0], np.nan, dtype="float32")
    # shrinkage covariance (LedoitWolf)
    lw = LedoitWolf().fit(G)
    cov = lw.covariance_.astype("float32")
    # (추가 regularization 선택 사항)
    cov = cov + reg_lambda * np.eye(d, dtype="float32")
    cov_inv = np.linalg.inv(cov)
    mu = G.mean(axis=0, keepdims=True)   # (1, d)
    diff = X - mu                         # (n_x, d)
    left = diff @ cov_inv                 # (n_x, d)
    d2 = np.sum(left * diff, axis=1)      # (n_x,)
    # 수치오차로 약간 음수 나오는 것 방어
    d2[d2 < 0] = 0.0
    return np.sqrt(d2).astype("float32")


def knn_score_to_group(X: np.ndarray, G: np.ndarray, k: int = 5,
                       metric: str = "cosine", alpha: float = 5.0) -> np.ndarray:
    """
    X : (n_x, d) - tap (CID profiles)
    G : (n_g, d) - this AID의 active 또는 inactive profiles
    k : kNN 이웃 수
    metric : "cosine" 또는 "euclidean"
    alpha : 거리 → score 변환 (score = exp(-alpha * mean_dist))
    """
    n_g = G.shape[0]
    if n_g == 0:
        return np.full(X.shape[0], np.nan, dtype="float32")
    k_eff = min(k, n_g)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, n_jobs=1)
    nn.fit(G)
    dists, _ = nn.kneighbors(X, return_distance=True)  # (n_x, k_eff)
    mean_d = dists.mean(axis=1)                        # (n_x,)
    score = np.exp(-alpha * mean_d)                    # 0~1 사이 근사 score
    return score.astype("float32")



from tqdm import tqdm
import numpy as np
import pandas as pd

i = 0
storage_list = []     # AID × Outcome level 통계
sample_scores = []    # AID × CID level Mahalanobis / kNN 점수
current_chunk_size = 1000
pbar = tqdm(total=total_aids, desc="Dynamic Chunk Processing")

# 지금 mab 추가해놨는데 빼기
while i < total_aids:
    # --- [Stage 1] 개별 처리 구간 (디버깅용) ---
    if i < 3000:
        aid = unique_aids[i]
        if aid in cid_lists:
            subs = cid_lists[aid]
            tap = ap.loc[all_cid_lists[aid],:]   # tap: sample x dimension (cid profile)
            # 1) 기존 group 통계
            for outcome in subs.index:
                cids = subs[outcome]
                valid_cids = ap.index.intersection(cids)
                if len(valid_cids) == 0:
                    continue
                sap = ap.loc[valid_cids]
                ap_means = sap.mean()
                ap_std_mean = sap.std().mean()
                row = {
                    'AID': aid,
                    'Activity Outcome': outcome,
                    'Count': len(valid_cids),
                    'ap_std_mean': ap_std_mean
                }
                row.update(ap_means.to_dict())
                storage_list.append(row)
            # 2) tap에 대해 AID active/inactive 기준 Mahalanobis + kNN
            if isinstance(tap, pd.DataFrame):
                tap_X = tap.to_numpy(dtype="float32")   # (n_tap, d)
                tap_cids = tap.index.values
            else:
                tap_X = np.asarray(tap, dtype="float32")
                # CID 리스트 따로 있으면 여기에 사용
                tap_cids = np.arange(tap_X.shape[0])
            aid_rows = traindf[traindf['AID'] == aid]
            act_cids = aid_rows[aid_rows['Activity Outcome'] == 'Active']['CID'].unique()
            inact_cids = aid_rows[aid_rows['Activity Outcome'] == 'Inactive']['CID'].unique()
            G_act = ap.loc[ap.index.intersection(act_cids)].to_numpy(dtype="float32")
            G_inact = ap.loc[ap.index.intersection(inact_cids)].to_numpy(dtype="float32")
            mahal_act   = mahalanobis_to_group(tap_X, G_act)
            mahal_inact = mahalanobis_to_group(tap_X, G_inact)
            knn_act     = knn_score_to_group(tap_X, G_act, k=5, metric="cosine", alpha=5.0)
            knn_inact   = knn_score_to_group(tap_X, G_inact, k=5, metric="cosine", alpha=5.0)
            for cid, m_a, m_i, s_a, s_i in zip(tap_cids, mahal_act, mahal_inact, knn_act, knn_inact):
                sample_scores.append({
                    'AID': aid,
                    'CID': cid,
                    'mahal_active':   m_a,
                    'mahal_inactive': m_i,
                    'knn_active':     s_a,
                    'knn_inactive':   s_i,
                })
        i += 1
        pbar.update(1)
    # --- [Stage 2] 대량 청크 처리 구간 ---
    else:
        start_time= time.time()
        end_idx = min(i + current_chunk_size, total_aids)
        target_aids = unique_aids[i:end_idx]
        chunk_bio = traindf[traindf['AID'].isin(target_aids)]
        # 기존 AID × Outcome level 통계
        counts = chunk_bio.groupby(['AID', 'Activity Outcome']).size()
        grouped = (
            chunk_bio
            .join(ap, on='CID')
            .groupby(['AID', 'Activity Outcome'])
            .agg(['mean', 'std'])
        )
        if not grouped.empty:
            numeric_grouped = grouped.select_dtypes(include=[np.number])
            means_df = numeric_grouped.xs('mean', level=1, axis=1).iloc[:, 1:(len(ap.columns)+1)]
            ap_std_mean_series = (
                numeric_grouped
                .xs('std', level=1, axis=1)
                .iloc[:, 1:(len(ap.columns)+1)]
                .mean(axis=1)
                .fillna('-')
            )
            for (aid_g, outcome), mean_values in means_df.iterrows():
                row_dict = {
                    'AID': aid_g,
                    'Activity Outcome': outcome,
                    'Count': counts.get((aid_g, outcome), 0),
                    'ap_std_mean': ap_std_mean_series.loc[(aid_g, outcome)]
                }
                row_dict.update(mean_values.to_dict())
                storage_list.append(row_dict)
        # 🚨 여기부터 추가: Stage 2에서도 AID별 sample_scores 계산
        # chunk_bio에는 target_aids에 해당하는 모든 row가 들어 있음
        for aid_g, aid_df in chunk_bio.groupby('AID'):
            print(f'{aid_g} processing for sample scores...{time.time()-start_time:.2f}s elapsed')
            # 이 AID에 대해 CID list (sample들)
            tap_cids = aid_df['CID'].unique()
            valid_cids = ap.index.intersection(tap_cids)
            if len(valid_cids) == 0:
                continue
            X = ap.loc[valid_cids].to_numpy(dtype="float32")  # (n_x, d)
            # active / inactive group profiles (이 AID 기준)
            act_cids = aid_df[aid_df['Activity Outcome'] == 'Active']['CID'].unique()
            inact_cids = aid_df[aid_df['Activity Outcome'] == 'Inactive']['CID'].unique()
            G_act = ap.loc[ap.index.intersection(act_cids)].to_numpy(dtype="float32")
            G_inact = ap.loc[ap.index.intersection(inact_cids)].to_numpy(dtype="float32")
            # Mahalanobis + kNN
            mahal_act   = mahalanobis_to_group(X, G_act)
            mahal_inact = mahalanobis_to_group(X, G_inact)
            knn_act     = knn_score_to_group(X, G_act, k=5, metric="cosine", alpha=5.0)
            knn_inact   = knn_score_to_group(X, G_inact, k=5, metric="cosine", alpha=5.0)
            for cid, m_a, m_i, s_a, s_i in zip(valid_cids, mahal_act, mahal_inact, knn_act, knn_inact):
                sample_scores.append({
                    'AID': aid_g,
                    'CID': cid,
                    'mahal_active':   m_a,
                    'mahal_inactive': m_i,
                    'knn_active':     s_a,
                    'knn_inactive':   s_i,
                })
        # 🚨 추가 끝
        actual_processed = end_idx - i
        pbar.update(actual_processed)
        i = end_idx
        del grouped

pbar.close()

# 최종 결과
df_group_stats   = pd.DataFrame(storage_list)   # AID × Outcome level
df_sample_scores = pd.DataFrame(sample_scores) # AID × CID level (Mahalanobis / kNN)



with open(f'{adir}/aid_ap_stats.pkl', 'wb') as f:
    pickle.dump(storage_list, f)



# # 3. 그룹별 Mean과 Std 계산
#     # 각 피처별로 mean과 std를 먼저 구합니다.
#     grouped = merged.groupby(['AID', 'Activity Outcome'])[ap_cols].agg(['mean', 'std'])
    
#     # 4. 결과 가공
#     # (1) AP 피처들의 Mean 값들
    
    
#     # (2) AP 피처들의 Std 값들의 평균 (Mean of Std)
#     # 각 피처의 std를 구한 결과(level=1의 'std')에 대해 가로 방향(axis=1) 평균을 냅니다.
#     stds_part = grouped.xs('std', level=1, axis=1).mean(axis=1).to_frame('ap_std_mean')
    
#     # 두 결과를 합쳐서 저장
#     chunk_result = pd.concat([means_part, stds_part], axis=1)
#     all_results.append(chunk_result)
    
#     i = end_idx
#     pbar.update(len(target_aids))


# # 모든 청크 합치기


# unique_aids = fbiodf2['AID'].unique()
# chunk_size = 10000  # 한 번에 처리할 AID 개수
# all_chunks = []
# all_stds = []
# all_Ns = []

# for i in range(0, len(unique_aids), chunk_size):
#     print(f"Processing AIDs {i} to {i+chunk_size}...")
#     target_aids = unique_aids[i:i+chunk_size]
    
#     # 해당 AID에 속하는 fbiodf2 부분 추출
#     temp_cbio = fbiodf2[fbiodf2['AID'].isin(target_aids)]
    
#     # ap_df와 조인 후 평균 계산
#     temp_df = temp_cbio.join(ap, on='CID').groupby(['AID', 'Activity Outcome'])
#     temp_mean = temp_df.mean()
#     all_chunks.append(temp_mean)
    


# 최종 합치기
# final_means = pd.concat(all_chunks)


ap = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_ap_opt.pkl')

apdf = pd.read_pickle(f'{adir}/aid_ap_stats.pkl')


fpdf = pd.read_pickle(f'{adir}/aid_fp_stats.pkl')

testdf = pd.read_pickle(f'{adir}/test_biodf.pkl')

trap = ap[~ap.index.isin(testdf.CID)]

tapdf = apdf[apdf.Count!=1]

# lls_np = np.load('/spstorage/USERS/gina/Project/FD/assay/lls_mem.npy', mmap_mode='r')
# cdss_np = np.load('/spstorage/USERS/gina/Project/FD/assay/cdss_mem.npy')

cdss_np = cdss_np[10000:20000, :]
cdss_np = cdss_np[20000:40000, :]

cdss_np = cdss_np[40000:60000, :]

cdss_np = cdss_np[66000:90000, :]


df_result.to_pickle(f'{adir}/df_result_fp.pkl')






lls_np = np.array(apdf.iloc[:,4:554])
cdss_np = np.array(ap)


lls_np = np.asarray(lls_np, dtype=np.float32)
cdss_np = np.asarray(cdss_np, dtype=np.float32)

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def ruzicka_tanimoto_allqueries(X, Y):
	nq = X.shape[0]
	ny = Y.shape[0]
	d = X.shape[1]
	out = np.empty((nq, ny), dtype=np.float32)
	for i in prange(nq):
		for j in range(ny):
			num = 0.0
			den = 0.0
			for k in range(d):
				x = X[i, k]
				y = Y[j, k]
				if x < y:
					num += x
					den += y
				else:
					num += y
					den += x
			out[i, j] = num / den if den > 0 else 0.0
	return out



from joblib import Parallel, delayed

def wttani(x, lls_np):
    mins = np.minimum(x, lls_np)
    maxs = np.maximum(x, lls_np)
    tanis = np.sum(mins, axis=1) / np.sum(maxs, axis=1)
    return tanis

def parallel_wttani(cdss_np, lls_np, indices, n_jobs=8):
    results = Parallel(n_jobs=n_jobs, backend='threading', batch_size=8)(
        delayed(wttani)(cdss_np[idx], lls_np) for idx in indices
    )
    return results



indices = list(range(len(cdss_np)))  # if you want 300k samples

start_time = time.time()
results = parallel_wttani(cdss_np, lls_np, indices, n_jobs=75)
print("---{}s seconds---".format(time.time()-start_time))



## centroid ㅁ
df_pairs= fbiodf2.iloc[:,0:2].drop_duplicates()

# ---------------------------
# 0. 준비: feature 컬럼 이름 정의
# ---------------------------
# apdf: [AID, Activity, ..., feat_1, feat_2, ..., feat_550]
# ap: [CID, feat_1, feat_2, ..., feat_550]
# 라고 가정하고, 실제 데이터에 맞게 index를 바꿔줘.
import numpy as np
import pandas as pd

# apdf: columns = [AID, Activity, ..., feat_1, ..., feat_550]
# ap:     index = CID, columns = feat_1, ..., feat_550
# df_pairs: columns = [AID, CID, ...]

# 1) feature 컬럼 이름 잡기 (4번째~554번째가 profile이라고 했음)
aid_feature_cols = apdf.columns[4:len(apdf.columns)]  # 0-based index

# AID별 active / inactive profile을 별도 DF로 (인덱스: AID, 값: 550D 벡터)
aid_active = (
    apdf[apdf["Activity Outcome"] == "Active"]
    .set_index("AID")[aid_feature_cols]
    .astype("float32")
)

aid_inactive = (
    apdf[apdf["Activity Outcome"] == "Inactive"]
    .set_index("AID")[aid_feature_cols]
    .astype("float32")
)

# CID profile DF (이미 ap가 그런 구조라고 했으니, dtype만 맞춰줌)
cid_profiles = ap.astype("float32")

feature_cols = list(aid_feature_cols)

def cosine_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x, y: shape (n_samples, n_features)
    각 row에 대해 cosine similarity
    """
    dot = np.sum(x * y, axis=1)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    denom = nx * ny
    # 0벡터 방지
    denom[denom == 0] = np.nan
    return dot / denom

def wttani_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    continuous Tanimoto (weighted Tanimoto)
    x, y: shape (n_samples, n_features)
    """
    mins = np.minimum(x, y)
    maxs = np.maximum(x, y)
    num = np.sum(mins, axis=1)
    den = np.sum(maxs, axis=1)
    den[den == 0] = np.nan
    return num / den



def tanimoto_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    standard Tanimoto (Jaccard for vectors)
    x, y: shape (n_samples, n_features)
    """
    xy = np.sum(x * y, axis=1)
    xx = np.sum(x * x, axis=1)
    yy = np.sum(y * y, axis=1)
    den = xx + yy - xy
    den[den == 0] = np.nan
    return xy / den
# from multiprocessing import Pool, cpu_count
# import os

# # 전역으로 사용할 변수 (worker에서 접근)
# AID_ACTIVE = None
# AID_INACTIVE = None
# CID_PROF = None
# DF_PAIRS = None

# def init_worker(aid_act, aid_inact, cid_prof, df_pairs):
#     """
#     각 worker 프로세스 시작할 때 한 번만 호출됨.
#     큰 DF들을 전역 변수에 넣어두고, copy-on-write로 공유.
#     """
#     global AID_ACTIVE, AID_INACTIVE, CID_PROF, DF_PAIRS
#     AID_ACTIVE = aid_act
#     AID_INACTIVE = aid_inact
#     CID_PROF = cid_prof
#     DF_PAIRS = df_pairs


# def compute_similarity_for_pairs(
#     df_pairs: pd.DataFrame,
#     aid_active: pd.DataFrame,
#     aid_inactive: pd.DataFrame,
#     cid_profiles: pd.DataFrame,
#     chunk_size: int = 100_000,
# ):
#     """
#     메모리 절약을 위해 df_pairs를 chunk 단위로 돌면서
#     cosine / wttani (active/inactive) 점수를 계산.
#     generator 형태로 chunk 결과를 yield.
#     """
#     n = len(df_pairs)
#     for start in range(0, n, chunk_size):
#         end = min(start + chunk_size, n)
#         chunk = df_pairs.iloc[start:end].copy()
#         # 1) AID / CID 키
#         aids = chunk["AID"].values
#         cids = chunk["CID"].values
#         # 2) 프로파일 인덱싱 (AID, CID → 벡터)
#         X_cid = cid_profiles.reindex(cids).to_numpy(dtype="float32")
#         X_act = aid_active.reindex(aids).to_numpy(dtype="float32")
#         X_inact = aid_inactive.reindex(aids).to_numpy(dtype="float32")
#         # 3) 유사도 계산
#         chunk["cosine_active"]    = cosine_pairwise(X_cid, X_act)
#         chunk["cosine_inactive"]  = cosine_pairwise(X_cid, X_inact)
#         chunk["wttani_active"]    = wttani_pairwise(X_cid, X_act)
#         chunk["wttani_inactive"]  = wttani_pairwise(X_cid, X_inact)
#         # 필요하면 여기서 바로 디스크에 append_write 해도 되고,
#         # 아니면 generator처럼 밖에서 모아서 쓸 수도 있음.
#         yield chunk


# def process_chunk(args):
#     start, end = args
#     # DF_PAIRS는 전역 변수
#     chunk = DF_PAIRS.iloc[start:end].copy()
#     aids = chunk["AID"].values
#     cids = chunk["CID"].values
#     # reindex로 AID/CID → 프로파일 벡터
#     X_cid = CID_PROF.reindex(cids).to_numpy(dtype="float32")
#     X_act = AID_ACTIVE.reindex(aids).to_numpy(dtype="float32")
#     X_inact = AID_INACTIVE.reindex(aids).to_numpy(dtype="float32")
#     # 유사도 계산
#     chunk["cosine_active"]    = cosine_pairwise(X_cid, X_act)
#     chunk["cosine_inactive"]  = cosine_pairwise(X_cid, X_inact)
#     chunk["wttani_active"]    = wttani_pairwise(X_cid, X_act)
#     chunk["wttani_inactive"]  = wttani_pairwise(X_cid, X_inact)
    
#     return chunk


from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 전역으로 공유할 DF들
AID_ACTIVE = None
AID_INACTIVE = None
CID_PROF = None
DF_PAIRS = None

def init_worker(aid_act, aid_inact, cid_prof, df_pairs):
    """
    각 worker 프로세스 시작 시 한 번 실행.
    큰 DF들을 전역 변수에 바인드해서 copy-on-write 공유.
    """
    global AID_ACTIVE, AID_INACTIVE, CID_PROF, DF_PAIRS
    AID_ACTIVE = aid_act
    AID_INACTIVE = aid_inact
    CID_PROF = cid_prof
    DF_PAIRS = df_pairs


def process_chunk(args):
    """
    하나의 chunk(start:end)를 처리해서
    AID, CID, similarity 4개 컬럼이 붙은 DataFrame을 리턴.
    """
    start, end = args
    chunk = DF_PAIRS.iloc[start:end].copy()
    aids = chunk["AID"].values
    cids = chunk["CID"].values
    X_cid = CID_PROF.reindex(cids).to_numpy(dtype="float32")
    X_act = AID_ACTIVE.reindex(aids).to_numpy(dtype="float32")
    X_inact = AID_INACTIVE.reindex(aids).to_numpy(dtype="float32")
    chunk["cosine_active"]    = cosine_pairwise(X_cid, X_act)
    chunk["cosine_inactive"]  = cosine_pairwise(X_cid, X_inact)
    chunk["wttani_active"]    = wttani_pairwise(X_cid, X_act)
    chunk["wttani_inactive"]  = wttani_pairwise(X_cid, X_inact)
    return chunk


def process_chunk(args):
    """
    하나의 chunk(start:end)를 처리해서
    AID, CID, similarity 4개 컬럼이 붙은 DataFrame을 리턴.
    """
    start, end = args
    chunk = DF_PAIRS.iloc[start:end].copy()
    aids = chunk["AID"].values
    cids = chunk["CID"].values
    X_cid = CID_PROF.reindex(cids).to_numpy(dtype="float32")
    X_act = AID_ACTIVE.reindex(aids).to_numpy(dtype="float32")
    X_inact = AID_INACTIVE.reindex(aids).to_numpy(dtype="float32")
    chunk["cosine_active"]    = cosine_pairwise(X_cid, X_act)
    chunk["cosine_inactive"]  = cosine_pairwise(X_cid, X_inact)
    chunk["tani_active"]    = tanimoto_pairwise(X_cid, X_act)
    chunk["tani_inactive"]  = tanimoto_pairwise(X_cid, X_inact)
    return chunk

def run_parallel_collect(df_pairs, aid_active, aid_inactive, cid_profiles,
                         chunk_size=100_000):
    """
    df_pairs 전체에 대해 병렬로 similarity 계산하고,
    최종적으로 하나의 DataFrame(df_result)에 모아 리턴.
    ⚠️ 3억 row 전부 모으면 메모리 사용량이 매우 큼.
    """
    n = len(df_pairs)
    tasks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        tasks.append((start, end))
    n_tasks = len(tasks)
    n_cpu = cpu_count()
    results = []
    with Pool(
        processes=n_cpu,
        initializer=init_worker,
        initargs=(aid_active, aid_inactive, cid_profiles, df_pairs),
    ) as pool:
        for chunk_res in tqdm(
            pool.imap_unordered(process_chunk, tasks),
            total=n_tasks,
            desc="Processing chunks",
        ):
            # 메모리에 모으기
            results.append(chunk_res)
    # 순서는 imap_unordered라 섞여 있을 수 있음 → 필요하면 정렬
    df_result = pd.concat(results, ignore_index=True)
    # 입력 df_pairs 순서랑 맞추고 싶으면, 원래 인덱스를 보존해서 정렬하는 식으로 바꿀 수도 있음.
    return df_result


df_pairs2 = df_pairs[df_pairs.AID.isin(yy.AID.unique())]
df_result2 = run_parallel_collect(
    df_pairs=df_pairs2,
    aid_active=aid_active,
    aid_inactive=aid_inactive,
    cid_profiles=cid_profiles,
    chunk_size=100_000,  # 메모리 상황 보고 조절
)


result_chunks = []

import time
start_time =time.time()
for chunk_res in compute_similarity_for_pairs(
    df_pairs,
    aid_active,
    aid_inactive,
    cid_profiles,
    chunk_size=1000,
):
    result_chunks.append(chunk_res)


print(f" 계산 완료 (소요: {time.time()-start_time:.2f}초)")

df_result = pd.concat(result_chunks, ignore_index=True)



import os
import tempfile
import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

_AP_MAT=None
_CID2ROW=None
_AP_MAT_PATH=None
_CID2ROW_PATH=None

def _init_globals(ap_mat_path, cid2row_path):
	global _AP_MAT,_CID2ROW,_AP_MAT_PATH,_CID2ROW_PATH
	_AP_MAT_PATH=ap_mat_path
	_CID2ROW_PATH=cid2row_path

def _ensure_loaded():
	global _AP_MAT,_CID2ROW
	if _AP_MAT is None:
		_AP_MAT=joblib.load(_AP_MAT_PATH,mmap_mode="r")
	if _CID2ROW is None:
		_CID2ROW=joblib.load(_CID2ROW_PATH)

def _rows_from_cids(cids):
	_ensure_loaded()
	idx=[]
	get=_CID2ROW.get
	for c in cids:
		j=get(c)
		if j is not None:
			idx.append(j)
	if len(idx)==0:
		return None
	return np.asarray(idx,dtype=np.int64)

def process_stage1_aid_memmap(aid_index, unique_aids, cid_lists, all_cid_lists, mahalanobis_to_group, knn_score_to_group):
	_ensure_loaded()
	aid=unique_aids[aid_index]
	subs=cid_lists[aid]
	tcds=all_cid_lists[aid]
	t_idx=_rows_from_cids(tcds)
	if t_idx is None:
		return None
	X=_AP_MAT[t_idx]
	out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
	for outcome in subs.index:
		cids=subs[outcome]
		s_idx=_rows_from_cids(cids)
		if s_idx is None:
			continue
		sap=_AP_MAT[s_idx]
		ma=mahalanobis_to_group(X,sap)
		ka=knn_score_to_group(X,sap)
		out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
		out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)
	return out

def run_stage1_parallel_memmap(ap, unique_aids, cid_lists, all_cid_lists, mahalanobis_to_group, knn_score_to_group, n_jobs=16, backend="loky", batch_size=1, tmp_dir=None):
	if tmp_dir is None:
		tmp_dir=tempfile.mkdtemp(prefix="stage1_memmap_")
	ap_mat_path=os.path.join(tmp_dir,"ap_mat.joblib")
	cid2row_path=os.path.join(tmp_dir,"cid2row.joblib")
	ap_mat=ap.to_numpy(dtype=np.float32,copy=True)
	joblib.dump(ap_mat,ap_mat_path,compress=0)
	cid2row={cid:i for i,cid in enumerate(ap.index.to_list())}
	joblib.dump(cid2row,cid2row_path,compress=3)
	_init_globals(ap_mat_path,cid2row_path)
	with tqdm(total=len(unique_aids),desc="stage1",mininterval=1.0) as pbar:
		def _wrapped(i):
			r=process_stage1_aid_memmap(i,unique_aids,cid_lists,all_cid_lists,mahalanobis_to_group,knn_score_to_group)
			pbar.update(1)
			return r
		res=Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size)(
			delayed(_wrapped)(i) for i in range(len(unique_aids))
		)
	return [r for r in res if r is not None]

stage1_results=run_stage1_parallel_memmap(ap,unique_aids,cid_lists,all_cid_lists,mahalanobis_to_group,knn_score_to_group,n_jobs=75,backend="loky",batch_size=1,tmp_dir=None)
import os
import tempfile
import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import joblib.parallel

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

_AP_MAT=None
_CID2ROW=None
_AP_MAT_PATH=None
_CID2ROW_PATH=None

_UNIQUE_AIDS=None
_CID_LISTS=None
_ALL_CID_LISTS=None

def _init_globals(ap_mat_path, cid2row_path, unique_aids, cid_lists, all_cid_lists):
	global _AP_MAT_PATH,_CID2ROW_PATH,_UNIQUE_AIDS,_CID_LISTS,_ALL_CID_LISTS,_AP_MAT,_CID2ROW
	_AP_MAT_PATH=ap_mat_path
	_CID2ROW_PATH=cid2row_path
	_UNIQUE_AIDS=unique_aids
	_CID_LISTS=cid_lists
	_ALL_CID_LISTS=all_cid_lists
	_AP_MAT=None
	_CID2ROW=None

def _ensure_loaded():
	global _AP_MAT,_CID2ROW
	if _AP_MAT is None:
		_AP_MAT=joblib.load(_AP_MAT_PATH,mmap_mode="r")
	if _CID2ROW is None:
		_CID2ROW=joblib.load(_CID2ROW_PATH)

def _rows_from_cids(cids):
	_ensure_loaded()
	idx=[]
	get=_CID2ROW.get
	for c in cids:
		j=get(c)
		if j is not None:
			idx.append(j)
	if len(idx)==0:
		return None
	return np.asarray(idx,dtype=np.int64)

def process_stage1_aid_memmap(aid_index):
	_ensure_loaded()
	aid=_UNIQUE_AIDS[aid_index]
	subs=_CID_LISTS[aid]
	tcds=_ALL_CID_LISTS[aid]
	t_idx=_rows_from_cids(tcds)
	if t_idx is None:
		return None
	X=_AP_MAT[t_idx]
	out={"AID":np.full(len(t_idx),aid,dtype=object),"CID":np.asarray(tcds,dtype=object)}
	for outcome in subs.index:
		cids=subs[outcome]
		s_idx=_rows_from_cids(cids)
		if s_idx is None:
			continue
		sap=_AP_MAT[s_idx]
		ma=mahalanobis_to_group(X,sap)
		ka=knn_score_to_group(X,sap)
		out[f"{outcome}_mahal_mean"]=np.asarray(ma,dtype=np.float32)
		out[f"{outcome}_knn_mean"]=np.asarray(ka,dtype=np.float32)
	return out

@contextmanager
def tqdm_joblib(tqdm_object):
	original_callback = joblib.parallel.BatchCompletionCallBack
	class TqdmBatchCompletionCallback(original_callback):
		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return original_callback.__call__(self, *args, **kwargs)
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = original_callback
		tqdm_object.close()

def run_stage1_parallel_memmap(ap, unique_aids, cid_lists, all_cid_lists, n_jobs=16, backend="loky", batch_size=1, tmp_dir=None):
	if tmp_dir is None:
		tmp_dir=tempfile.mkdtemp(prefix="stage1_memmap_")
	ap_mat_path=os.path.join(tmp_dir,"ap_mat.joblib")
	cid2row_path=os.path.join(tmp_dir,"cid2row.joblib")
	ap_mat=ap.to_numpy(dtype=np.float32,copy=True)
	joblib.dump(ap_mat,ap_mat_path,compress=0)
	cid2row={cid:i for i,cid in enumerate(ap.index.to_list())}
	joblib.dump(cid2row,cid2row_path,compress=3)
	_init_globals(ap_mat_path,cid2row_path,unique_aids,cid_lists,all_cid_lists)
	with tqdm_joblib(tqdm(total=len(unique_aids),desc="stage1",mininterval=1.0)):
		res=Parallel(n_jobs=n_jobs,backend=backend,batch_size=batch_size)(
			delayed(process_stage1_aid_memmap)(i) for i in range(len(unique_aids))
		)
	return [r for r in res if r is not None]


stage1_results=run_stage1_parallel_memmap(ap,unique_aids,cid_lists,all_cid_lists,n_jobs=75,backend="loky",batch_size=1,tmp_dir=None)

testapdf = apdf[apdf.CID.isin(testdf.CID)]

testapdf[testapdf.Active==1].iloc[:,2:6].mean()

testapdf[testapdf.Inactive==1].iloc[:,2:6].mean()

# inplace=True를 사용하면 기존 데이터프레임에 바로 반영되어 메모리를 아낄 수 있습니다.
testapdf.eval("""
    cosine_ratio = (cosine_active + 0.01) / (cosine_inactive + 0.01)
    wttani_ratio = (wttani_active + 0.01) / (wttani_inactive + 0.01)
""", inplace=True)


testapdf[testapdf.Active==1].iloc[:,2:].mean()

testapdf[testapdf.Inactive==1].iloc[:,2:].mean()



testfpdf = fpdf[fpdf.CID.isin(testdf.CID)]

testfpdf[testfpdf.Active==1].iloc[:,2:6].mean()

testfpdf[testfpdf.Inactive==1].iloc[:,2:6].mean()

# inplace=True를 사용하면 기존 데이터프레임에 바로 반영되어 메모리를 아낄 수 있습니다.
testfpdf.eval("""
    cosine_ratio = (cosine_active + 0.01) / (cosine_inactive + 0.01)
    tani_ratio = (tani_active + 0.01) / (tani_inactive + 0.01)
""", inplace=True)


testfpdf[testfpdf.Active==1].iloc[:,2:].mean()

testfpdf[testfpdf.Inactive==1].iloc[:,2:].mean()

testapdf[testapdf.Active==1].iloc[:,2:].mean()

testapdf[testapdf.Inactive==1].iloc[:,2:].mean()


resdf = pd.read_pickle(f'{adir}/ap_cent_mean.pkl')

fresdf=  fresdf.reset_index(drop=True)
testresdf = fresdf[fresdf.CID.isin(testdf.CID)]
testresdf= testresdf[testresdf.index.isin(testresdf.iloc[:,[0,1,4,5,8,9]].dropna().index)]

ff =final_df[(final_df.AID.isin(testresdf.AID)&final_df.CID.isin(testresdf.CID))]
testresdf = testresdf.sort_values(['AID','CID'])

yyy = pd.concat([testresdf.reset_index(drop=True), ff.reset_index(drop=True).iloc[:,2:]],axis=1)


yyy[yyy.Active==1].iloc[:,2:].mean()

yyy[yyy.Inactive==1].iloc[:,2:].mean()

# inplace=True를 사용하면 기존 데이터프레임에 바로 반영되어 메모리를 아낄 수 있습니다.
yyy.eval("""
    maha_ratio = (Active_mahal_mean + 0.01) / (Inactive_mahal_mean + 0.01)
    knn_ratio = (Active_knn_mean + 0.01) / (Inactive_knn_mean + 0.01)
""", inplace=True)
yyy.eval("""
    maha_rratio = (Inactive_mahal_mean + 0.1) / (Active_mahal_mean + 0.1)
""", inplace=True)

yyy.eval("""
    maha_diff = (Active_mahal_mean ) - (Inactive_mahal_mean )
""", inplace=True)


testfpdf[testfpdf.AID.isin(yyy.AID)][testfpdf[testfpdf.AID.isin(yyy.AID)].Active==1].iloc[:,2:].mean()

testfpdf[testfpdf.AID.isin(yyy.AID)][testfpdf[testfpdf.AID.isin(yyy.AID)].Inactive==1].iloc[:,2:].mean()



## protein 변환

acc = list(set(bio2['Protein Accessions']))
unis = list(set(bio2['UniProts IDs']))

import requests

def fetch_ncbi_protein_sequence(acc):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "protein",
        "id": acc,
        "rettype": "fasta",
        "retmode": "text"
    }
    r = requests.get(url, params=params)
    if not r.ok:
        return None
    return "".join(
        line.strip() for line in r.text.splitlines()
        if not line.startswith(">")
    )

seqs = []
for i, ac in enumerate(accs[1053:]):
    seq = fetch_ncbi_protein_sequence(ac)
    seqs.append([ac,seq])
    print(i)
     
xx = pd.read_pickle(f'{adir}/ptn_acc.pkl')
aa = pd.DataFrame(seqs)
aa.columns = xx.columns
dfdf = pd.concat([xx,aa])
dfdf.to_pickle(f'{adir}/ptn_acc.pkl')



# import requests,time

# def ncbi_to_uniprot(ncbi_acc):
#     # 1) submit mapping job
#     url = "https://rest.uniprot.org/idmapping/run"
#     r = requests.post(url, data={
#         "from": "RefSeq_Protein",
#         "to": "UniProtKB",
#         "ids": ncbi_acc
#     })
#     if not r.ok:
#         return None
#     job_id = r.json()["jobId"]
#     # 2) wait for result
#     status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
#     while True:
#         s = requests.get(status_url).json()
#         if s["jobStatus"] == "FINISHED":
#             break
#         time.sleep(1)
#     # 3) fetch result
#     result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
#     res = requests.get(result_url).json()
#     if "results" not in res or len(res["results"]) == 0:
#         return None
#     return res["results"][0]["to"]  # UniProt ID


# def fetch_pdb_chain_sequence(pdb_id, chain_id):
#     pdb_id = pdb_id.upper()
#     url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
#     r = requests.get(url)
#     if not r.ok:
#         return None
#     seq = ""
#     record = False
#     for line in r.text.splitlines():
#         if line.startswith(">"):
#             record = f"Chains {chain_id}" in line
#             seq = ""
#         elif record:
#             seq += line.strip()
#     return seq

# seq = fetch_pdb_chain_sequence("1Y7V", "A")

# import requests,time

def refseq_to_uniprot(refseq_id):
    # submit job
    url = "https://rest.uniprot.org/idmapping/run"
    r = requests.post(url, data={
        "from": "RefSeq_Protein",
        "to": "UniProtKB",
        "ids": refseq_id
    })
    if not r.ok:
        return None
    job_id = r.json()["jobId"]
    # poll status
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        s = requests.get(status_url).json()
        if s.get("jobStatus") == "FINISHED":
            break
        time.sleep(0.5)
    # fetch result
    result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
    res = requests.get(result_url).json()
    if "results" not in res or len(res["results"]) == 0:
        return None
    return res["results"][0]["to"]


import requests

def pdb_chain_to_uniprot(pdb_id, chain_id):
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    r = requests.get(url)
    if not r.ok:
        return None
    data = r.json()
    if pdb_id not in data:
        return None
    for up_id, info in data[pdb_id]["UniProt"].items():
        chains = info.get("chains", [])
        for ch in chains:
            if ch["chain_id"] == chain_id:
                return up_id
    return None

uid = pdb_chain_to_uniprot("1Y7V", "A")
print(uid)  # 예: P00533 (EGFR)

# UniProt



### PDB

import requests
import re

def fetch_pdb_chain_sequence(pdb_id, chain_id):
    pdb_id = pdb_id.upper()
    chain_id = chain_id.upper()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    r = requests.get(url)
    if not r.ok:
        return None
    seq = ""
    record = False
    for line in r.text.splitlines():
        if line.startswith(">"):
            record = False
            seq = ""
            # Chains part 추출
            m = re.search(r"Chains?\s+([A-Za-z0-9,\s]+)", line)
            if m:
                chains_str = m.group(1)
                # "A, B" / "A,B" / "A and B" → ["A","B"]
                chains_str = chains_str.replace("and", ",")
                chains = [c.strip() for c in chains_str.split(",") if c.strip()]
                if chain_id in chains:
                    record = True
        elif record:
            seq += line.strip()
    return seq if seq else None

seqq = []

for i, ac in enumerate(accs):
     print(i)
     if not pd.isna(ac):
        if is_pdb_chain(ac):
            seqq.append([ac, fetch_pdb_chain_sequence(ac[0:4], ac[5])])



seq = fetch_pdb_chain_sequence("1Y7V", "A")

import re

PDB_CHAIN_PATTERN = re.compile(r"^[0-9A-Za-z]{4}_[A-Za-z0-9]$")

NP_CHAIN_PATTERN = re.compile(r"^NP_[0-9]+")


UNIPROT_PATTERN = re.compile(
    r"^(?:[A-Z][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{8})$"
)
def is_pdb_chain(acc):
    if acc is None:
        return False
    return bool(
        NP_CHAIN_PATTERN.match(acc)
        or PDB_CHAIN_PATTERN.match(acc)
        or UNIPROT_PATTERN.match(acc)
    )

df[df.iloc[:, 1].astype(str).str.startswith("Error")]


cc3= [a for a in acc2 if not is_pdb_chain(a)]
len(cc3)


import requests

def fetch_ncbi_protein_sequence(acc, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "protein",
        "id": acc,
        "rettype": "fasta",
        "retmode": "text"
    }
    if api_key is not None:
        params["api_key"] = api_key
    r = requests.get(url, params=params, timeout=10)
    if not r.ok:
        return None
    seq = "".join(
        line.strip()
        for line in r.text.splitlines()
        if not line.startswith(">")
    )
    return seq or None


### AAA

adir = '/spstorage/USERS/gina/Project/FD/assay/'


adf = pd.read_pickle(f'{adir}/all_simdf.pkl')

testdf = pd.read_pickle(f'{adir}/test_biodf.pkl')

X_train = adf[~adf.CID.isin(testdf.CID)].iloc[:,list(range(2,18))].sample(frac=1, random_state=7)
y_train = adf[~adf.CID.isin(testdf.CID)]['Active']

X_test = adf[adf.CID.isin(testdf.CID)].iloc[:,list(range(2,18))].sample(frac=1, random_state=7)
y_test = adf[adf.CID.isin(testdf.CID)]['Active']


import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

feature_names = X_train.columns.tolist()
# X_train, y_train, X_test, y_test, feature_names 전제
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_names)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42
}

lgbm = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    #early_stopping_rounds=50,
    #verbose_eval=50
)

y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)
auc = roc_auc_score(y_test, y_pred)
print("LightGBM AUC:", auc)

# feature importance (gain 기준 추천)
gain_importance = lgbm.feature_importance(importance_type="gain")
split_importance = lgbm.feature_importance(importance_type="split")

fi_df = pd.DataFrame({
    "feature": feature_names,
    "gain_importance": gain_importance,
    "split_importance": split_importance
}).sort_values("gain_importance", ascending=False)

print(fi_df)

with open(f'{adir}/temp.pkl', 'wb') as f:
    pickle.dump(accs, f)

with open(f'{adir}/temp.pkl', 'rb') as f:
    accs = pickle.load(f)






accdf = pd.read_pickle(f'{adir}/')

nbiodf2 = bio2[pd.isna(bio2.iloc[:,3])]
abiodf2 = bio2[~pd.isna(bio2.iloc[:,3])]

abiodf2 = bio2[~pd.isna(bio2.iloc[:,3])]

adfdf = pd.merge(abiodf2, a1, left_on = 'Protein Accession', right_on = 'ACC')

abiodf2 = bio2[~pd.isna(bio2.iloc[:,3])]
nbiodf2 = bio2[pd.isna(bio2.iloc[:,3])]

biodf2 = pd.read_pickle(f'{adir}/sub_biodf2.pkl')

bb=bio2[~pd.isna(bio2.iloc[:,3])]


bbb=bb[bb.iloc[:,3].isin(a1.ACC)]
accs = list(set(bb.iloc[:,3])-set(bbb.iloc[:,3]))

e1 = adf[adf.iloc[:, 1].astype(str).str.startswith("Error")]
e2 = adf[pd.isna(adf.iloc[:, 1])]
ne = adf[~(adf.iloc[:, 1].astype(str).str.startswith("Error")) & ~pd.isna(adf.iloc[:, 1])]

ne.to_pickle(f'{adir}/alldf_pivot.pkl')


####

# biodf2 =  
biodf2 = pd.read_pickle(f'{adir}/sub_biodf2.pkl')
aid2seq = pd.read_pickle(f'{adir}/alldf_pivot.pkl')

abio2 = bio2[~pd.isna(bio2['Protein Accessions'])]

laids = list(set(abiodf2['Protein Accession']) - set(aid2seq['ACC']))

pd.merge(bio2, aid2seq, left_on = 'Protein Accessions', right_on = 'ACC')

### data preprocessing

biodf2 = pd.read_pickle(f'{adir}/sub_biodf2.pkl')
aid2seq = pd.read_pickle(f'{adir}/alldf_pivot.pkl')
apdf = pd.read_pickle(f'{adir}/pubchem_ap_opt.pkl')
smidf = pd.read_pickle(f'{adir}/pubchem_smiles_opt.pkl')



# 1) biodf2: pair / label 테이블
biodf2 = pd.read_pickle(f'{adir}/sub_biodf2.pkl')

# 2) aid2seq: Protein Accession -> SEQ
aid2seq = pd.read_pickle(f'{adir}/alldf_pivot.pkl')
aid2seq.columns =['Protein Accession','SEQ']

aid2seq = aid2seq.set_index("Protein Accession")  # 나중에 .loc로 바로 검색

# 3) apdf: CID index, 550 AP columns
apdf = pd.read_pickle(f'{adir}/pubchem_ap_opt.pkl')
# apdf.index: CID

# 4) smidf: CID index, smiles column
smidf = pd.read_pickle(f'{adir}/pubchem_smiles_opt.pkl')
# smidf.index: CID
valid_cids = apdf.index.intersection(smidf.index)
biodf2 = biodf2[biodf2["CID"].isin(valid_cids)].reset_index(drop=True)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Chem2AssayDataset(Dataset):
    def __init__(self,biodf2,aid2seq,apdf,smidf,assay_text_df=None):
        self.df=biodf2.reset_index(drop=True)
        self.aid2seq=aid2seq.set_index("Protein Accession") if "Protein Accession" in aid2seq.columns else aid2seq
        self.apdf=apdf
        self.smidf=smidf
        self.assay_text_df=assay_text_df
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        cid=r["CID"]
        aid=r["AID"]
        prot=r["Protein Accession"]
        smiles=self.smidf.loc[cid,"smiles"]
        ap_vec=self.apdf.loc[cid].values.astype(np.float32)
        assay_text=self.assay_text_df.loc[aid,"assay_text"] if self.assay_text_df is not None and aid in self.assay_text_df.index else f"Assay {aid}"
        target_seq=self.aid2seq.loc[prot,"SEQ"] if prot in self.aid2seq.index else None
        return {"smiles":smiles,"assay_text":assay_text,"target_seq":target_seq,"ap_vec":ap_vec,"assay_cont":None,"label":float(r["Activity Outcome"]),"aux_sim":None}


chem_tok = AutoTokenizer.from_pretrained(args.chemberta_name)
assay_tok = AutoTokenizer.from_pretrained(args.assay_model_name)

collate_fn = make_collate_fn(
    chem_tok,
    assay_tok,
    use_assay_cont=False,          # assay_cont가 없으니까
    use_aux_sim=args.use_aux_sim_head
)

dataset = Chem2AssayDataset(
    biodf2=biodf2,
    aid2seq=aid2seq,
    apdf=apdf,
    smidf=smidf,
    assay_text_df=None   # 나중에 AID→설명 df 생기면 여기 넣기
)

sampler = DistributedSampler(dataset, shuffle=True)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_fn
)


with autocast():
    out = model(
        smiles_inputs=batch["smiles_inputs"],
        ap_vec=batch["ap_vec"],
        assay_inputs=batch["assay_inputs"],
        target_seqs=batch["target_seqs"],
        assay_cont_vec=batch["assay_cont_vec"],
        labels=batch["labels"],
        aux_similarity_targets=batch["aux_similarity_targets"],
        return_embeddings=False
    )



biodf2=pd.read_pickle(f'{adir}/sub_biodf2.pkl')
aid2seq=pd.read_pickle(f'{adir}/alldf_pivot.pkl')
apdf=pd.read_pickle(f'{adir}/pubchem_ap_opt.pkl')
smidf=pd.read_pickle(f'{adir}/pubchem_smiles_opt.pkl')
valid_cids=apdf.index.intersection(smidf.index)
biodf2=biodf2[biodf2["CID"].isin(valid_cids)].reset_index(drop=True)
dataset=Chem2AssayDataset(biodf2,aid2seq,apdf,smidf)
print(len(dataset))
print(dataset[0])



## SMIlES similarity
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import time

def smiles_similarity_matrix(smiles_list, radius=2, n_bits=2048):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        for m in mols
    ]
    n = len(fps)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_matrix[i, :] = sims
    return sim_matrix


smidf = pd.read_pickle(f'{adir}/pubchem_smiles.pkl')

smiles_list = unique(smidf['scaffold'].tolist())

smidf = smidf[~(smidf.scaffold=='[c-]1cccc1')]


testscf = smidf[smidf.index.isin(testdf.CID)]['scaffold'].unique()

trainscf = smidf[~smidf.index.isin(testdf.CID)]['scaffold'].unique()

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np

def smiles_similarity_matrix_1v2(
    smiles_list_1,
    smiles_list_2,
    radius=2,
    n_bits=2048,
    dtype=np.float32,
):
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols1 = [Chem.MolFromSmiles(s) for s in smiles_list_1]
    mols2 = [Chem.MolFromSmiles(s) for s in smiles_list_2]
    fps1 = [gen.GetFingerprint(m) for m in mols1]
    fps2 = [gen.GetFingerprint(m) for m in mols2]
    n1, n2 = len(fps1), len(fps2)
    sim = np.zeros((n1, n2), dtype=dtype)
    for i, fp in enumerate(fps1):
        sim[i, :] = DataStructs.BulkTanimotoSimilarity(fp, fps2)
    return sim


st = time.time()

sim_1v2 = smiles_similarity_matrix_1v2(trainscf, testscf)
print("Time taken:", time.time() - st)


pd.DataFrame(sim_1v2, index=trainscf, columns=testscf).to_pickle(f'{adir}/smiles_sim_train_test.pkl')


st = time.time()
sim_matrix = smiles_similarity_matrix(smiles_list)
print("Time taken:", time.time() - st)


# from rdkit import Chem, DataStructs
# from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# import numpy as np

# def smiles_similarity_matrix(
#     smiles_list,
#     radius=2,
#     n_bits=2048,
#     use_chirality=False
# ):
#     mols = [Chem.MolFromSmiles(s) for s in smiles_list]
#     generator = GetMorganGenerator(
#         radius=radius,
#         fpSize=n_bits,
#         useChirality=use_chirality
#     )
#     fps = [generator.GetFingerprint(m) for m in mols]
#     n = len(fps)
#     sim_matrix = np.zeros((n, n), dtype=float)
#     for i in range(n):
#         sim_matrix[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
#     return sim_matrix
