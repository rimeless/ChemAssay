### 연습 

import sys
src_dir = '/spstorage/USERS/gina/source' 
sys.path.append(src_dir)
from basic import *
from CC import *
from operator import itemgetter, add
import glob
from collections import OrderedDict

adir = '/spstorage/USERS/gina/Project/FD/assay/'

with open('/spstorage/USERS/gina/Project/FD/assay/hh2_processpool_results_except_test.pkl','rb') as f:
    actss, inactss, fp_actss, fp_inactss = pickle.load(f)


actdf= [a[0] for a in actss]
actdf2 = pd.DataFrame(actdf)

iactdf= [a[0] for a in inactss]
iactdf2 = pd.DataFrame(iactdf)


ap = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_ap.pkl')
tap = ap[ap.index.isin(ttbiodf.CID)]


actdf= [a[0] for a in fp_actss]
actdf2 = pd.DataFrame(actdf)

iactdf= [a[0] for a in fp_inactss]
iactdf2 = pd.DataFrame(iactdf)


fp = pd.read_pickle('/spstorage/USERS/gina/Project/FD/assay/pubchem_fp.pkl')
tfp = fp[fp.index.isin(ttbiodf.CID)]


## mean UMAP plot

import numpy as np
import umap
import matplotlib.pyplot as plt

X1=actdf2.to_numpy(dtype=np.float32)
X2=iactdf2.to_numpy(dtype=np.float32)
X3=tfp.sample(frac=1).iloc[0:1800,:].to_numpy(dtype=np.float32)

m1=np.isfinite(X1).all(axis=1)
m2=np.isfinite(X2).all(axis=1)
m3=np.isfinite(X3).all(axis=1)

X1=X1[m1]
X2=X2[m2]
X3=X3[m3]

X=np.vstack([X1,X2,X3])
labels=np.array(["active"]*X1.shape[0]+["inactive"]*X2.shape[0]+["test"]*X3.shape[0],dtype=object)

reducer=umap.UMAP(n_neighbors=30,min_dist=0.3,n_components=2,metric="cosine",random_state=0)
emb=reducer.fit_transform(X)

plt.figure(figsize=(7.5,6.5),dpi=200)
for name in ["active","inactive","test"]:
    idx=labels==name
    plt.scatter(emb[idx,0],emb[idx,1],s=10,label=name)



plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP (active vs inactive vs test)")
plt.legend(frameon=False,markerscale=2)
plt.tight_layout()

plt.show()





from colorsys import hsv_to_rgb



def _srgb_to_linear(c):
    c=np.asarray(c)
    return np.where(c<=0.04045, c/12.92, ((c+0.055)/1.055)**2.4)

def _rgb_to_xyz(rgb):
    M=np.array([[0.4124564,0.3575761,0.1804375],
                [0.2126729,0.7151522,0.0721750],
                [0.0193339,0.1191920,0.9503041]],dtype=float)
    return rgb@M.T

def _xyz_to_lab(xyz):
    white=np.array([0.95047,1.0,1.08883],dtype=float)
    x=xyz/white
    eps=216/24389
    kappa=24389/27
    f=np.where(x>eps, np.cbrt(x), (kappa*x+16)/116)
    L=116*f[...,1]-16
    a=500*(f[...,0]-f[...,1])
    b=200*(f[...,1]-f[...,2])
    return np.stack([L,a,b],axis=-1)

def _rgb_to_lab(rgb):
    rgb=np.clip(rgb,0,1)
    lin=_srgb_to_linear(rgb)
    xyz=_rgb_to_xyz(lin)
    lab=_xyz_to_lab(xyz)
    return lab

def glasbey_like_colors(n, seed=0, pool=8000):
    rng=np.random.default_rng(seed)
    hs=rng.random(pool)
    ss=rng.uniform(0.55,0.95,pool)
    vs=rng.uniform(0.65,0.95,pool)
    rgb=np.array([hsv_to_rgb(h,s,v) for h,s,v in zip(hs,ss,vs)],dtype=float)
    lab=_rgb_to_lab(rgb)
    chosen=[rng.integers(pool)]
    dist=np.linalg.norm(lab-lab[chosen[0]],axis=1)
    for _ in range(1,n):
        idx=int(np.argmax(dist))
        chosen.append(idx)
        dist=np.minimum(dist, np.linalg.norm(lab-lab[idx],axis=1))
    return rgb[np.array(chosen)]

rng=np.random.default_rng(0)

X1=actdf2.to_numpy(dtype=np.float32)
X2=iactdf2.to_numpy(dtype=np.float32)

m1=np.isfinite(X1).all(axis=1)
m2=np.isfinite(X2).all(axis=1)

idx1=np.flatnonzero(m1)
idx2=np.flatnonzero(m2)

n=30
s1=idx1[rng.choice(idx1.size,size=min(n,idx1.size),replace=False)]
s2=idx2[rng.choice(idx2.size,size=min(n,idx2.size),replace=False)]

A=X1[s1]
B=X2[s2]

X=np.vstack([A,B])
emb=umap.UMAP(n_neighbors=15,min_dist=0.3,n_components=2,metric="cosine",random_state=0).fit_transform(X)

pair_cols=glasbey_like_colors(n, seed=1, pool=12000)

plt.figure(figsize=(7.5,6.5),dpi=200)

for i in range(A.shape[0]):
    plt.scatter(emb[i,0],emb[i,1],s=80,marker="o",c=[pair_cols[i]],edgecolors="k",linewidths=0.55)



for i in range(B.shape[0]):
    j=A.shape[0]+i
    plt.scatter(emb[j,0],emb[j,1],s=80,marker="s",c=[pair_cols[i]],edgecolors="k",linewidths=0.55)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP: paired colors (active=o, inactive=s), 30 pairs")
plt.tight_layout()
plt.show()


import numpy as np
import umap
import matplotlib.pyplot as plt

X1=actdf2.to_numpy(dtype=np.float32)
X2=iactdf2.to_numpy(dtype=np.float32)
X3=tactdf2.to_numpy(dtype=np.float32)

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
plt.savefig("actdf2_iactdf2_tactdf2_umap.png",dpi=300)
plt.close()


