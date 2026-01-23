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
# df_cid_lists = cid_lists.reset_index()
# df_cid_lists.columns = ['AID', 'Activity Outcome', 'CID_list']


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
# df_cid_lists = cid_lists2.reset_index()
# df_cid_lists.columns = ['AID', 'Activity Outcome', 'CID_list']


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



final_ratios=[aid_current_random_counts[aid]/aid_total_counts[aid] for aid in aid_total_counts if aid_total_counts[aid]>0]
         

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
rrr.to_pickle(f'{adir}/test_random_biodf.pkl')

fbiodf2.to_pickle(f'{adir}/fbiodf2.pkl')
sub_f = final_scf_df[final_scf_df.scaffold.isin(global_test_scaffolds)]
fff =fbiodf2[fbiodf2.CID.isin(sub_f.index)]
fff.to_pickle(f'{adir}/test_biodf.pkl')

fbiodf2= pd.read_pickle(f'{adir}/fbiodf2.pkl')

ap = pd.read_pickle(f'{adir}/pubchem_fp_opt.pkl')

fff = pd.read_pickle(f'{adir}/test_biodf.pkl')

traindf = fbiodf2[~fbiodf2.index.isin(fff.index)]

# 1. AID와 Activity Outcome별로 CID를 리스트로 묶기
cid_lists = traindf.groupby(['AID', 'Activity Outcome'])['CID'].apply(list)





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
            
            means_df = numeric_grouped.xs('mean', level=1, axis=1).iloc[:,1:551]
            # [수정] NaN은 0으로 채우는 것이 안전함
            ap_std_mean_series = numeric_grouped.xs('std', level=1, axis=1).iloc[:,1:551].mean(axis=1).fillna('-')
            
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


# 3. 그룹별 Mean과 Std 계산
    # 각 피처별로 mean과 std를 먼저 구합니다.
    grouped = merged.groupby(['AID', 'Activity Outcome'])[ap_cols].agg(['mean', 'std'])
    
    # 4. 결과 가공
    # (1) AP 피처들의 Mean 값들
    
    
    # (2) AP 피처들의 Std 값들의 평균 (Mean of Std)
    # 각 피처의 std를 구한 결과(level=1의 'std')에 대해 가로 방향(axis=1) 평균을 냅니다.
    stds_part = grouped.xs('std', level=1, axis=1).mean(axis=1).to_frame('ap_std_mean')
    
    # 두 결과를 합쳐서 저장
    chunk_result = pd.concat([means_part, stds_part], axis=1)
    all_results.append(chunk_result)
    
    i = end_idx
    pbar.update(len(target_aids))


# 모든 청크 합치기


unique_aids = fbiodf2['AID'].unique()
chunk_size = 10000  # 한 번에 처리할 AID 개수
all_chunks = []
all_stds = []
all_Ns = []

for i in range(0, len(unique_aids), chunk_size):
    print(f"Processing AIDs {i} to {i+chunk_size}...")
    target_aids = unique_aids[i:i+chunk_size]
    
    # 해당 AID에 속하는 fbiodf2 부분 추출
    temp_cbio = fbiodf2[fbiodf2['AID'].isin(target_aids)]
    
    # ap_df와 조인 후 평균 계산
    temp_df = temp_cbio.join(ap, on='CID').groupby(['AID', 'Activity Outcome'])
    temp_mean = temp_df.mean()
    all_chunks.append(temp_mean)
    


# 최종 합치기
final_means = pd.concat(all_chunks)