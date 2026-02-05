"""
sdf_featurizer_np.py
SDF 파일에서 CompoundTransformer에 들어가는 입력 feature들을 numpy로만 생성.
리턴 형식:
- atom_features               : (num_atoms, atom_feat_dim)
- pharmacophore_features      : (num_pharma, pharma_feat_dim)
- atom_to_pharmacophore       : (num_atoms,)
- pair_features               : (num_atoms, num_atoms, pair_feat_dim)
- pharmacophore_pair_features : (num_pharma, num_pharma, 64 + num_distance_bins)
- atom_mask                   : (num_atoms,)
conformer 단위 dict로 묶고, molecule 단위로 list[list[dict]] 형태로 리턴.
"""
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig
PHARMA_FAMILIES = ["Donor","Acceptor","Aromatic","PosIonizable","NegIonizable","Hydrophobe","ZnBinder"]
_PHARMA_FAMILY_TO_IDX = {name:i for i,name in enumerate(PHARMA_FAMILIES)}
def _get_feature_factory() -> ChemicalFeatures.FreeChemicalFeatureFactory:
    fdef_name = os.path.join(RDConfig.RDDataDir,"BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    return factory


FEATURE_FACTORY = _get_feature_factory()

def featurize_atoms(mol: Chem.Mol) -> np.ndarray:
    feats = []
    for atom in mol.GetAtoms():
        f = []
        Z = atom.GetAtomicNum()
        degree = atom.GetTotalDegree()
        formal_charge = atom.GetFormalCharge()
        aromatic = int(atom.GetIsAromatic())
        num_H = atom.GetTotalNumHs(includeNeighbors=True)
        f.append(Z/100.0)
        f.append(float(degree))
        f.append(float(formal_charge))
        f.append(float(aromatic))
        f.append(float(num_H))
        hyb = atom.GetHybridization()
        hyb_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP2D,
            Chem.rdchem.HybridizationType.SP3D,
        ]
        hyb_one_hot = [float(hyb==h) for h in hyb_types]
        f.extend(hyb_one_hot)
        feats.append(f)
    return np.asarray(feats,dtype=np.float32)


def _rbf_encode(dist: np.ndarray,num_basis: int=32,d_min: float=0.0,d_max: float=20.0) -> np.ndarray:
    centers = np.linspace(d_min,d_max,num_basis)
    gamma = 1.0/((centers[1]-centers[0])**2+1e-8)
    diff = dist[...,None]-centers[None,...]
    rbf = np.exp(-gamma*diff**2)
    return rbf.astype(np.float32)


def featurize_atom_pairs(mol: Chem.Mol,conf_id: int,num_distance_basis: int=32) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)],dtype=np.float32)
    diff = coords[:,None,:]-coords[None,:,:]
    dists = np.linalg.norm(diff,axis=-1)
    rbf = _rbf_encode(dists,num_basis=num_distance_basis,d_min=0.0,d_max=20.0)
    bond_feat = np.zeros((num_atoms,num_atoms,4),dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt==Chem.BondType.SINGLE:
            idx = 0
        elif bt==Chem.BondType.DOUBLE:
            idx = 1
        elif bt==Chem.BondType.TRIPLE:
            idx = 2
        elif bt==Chem.BondType.AROMATIC:
            idx = 3
        else:
            idx = None
        if idx is not None:
            bond_feat[i,j,idx] = 1.0
            bond_feat[j,i,idx] = 1.0
    pair_features = np.concatenate([rbf,bond_feat],axis=-1)
    return pair_features


def featurize_pharmacophores(mol: Chem.Mol,conf_id: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer(conf_id)
    feats_rdkit = FEATURE_FACTORY.GetFeaturesForMol(mol)
    centers = []
    types = []
    for f in feats_rdkit:
        fam = f.GetFamily()
        if fam not in _PHARMA_FAMILY_TO_IDX:
            continue
        atom_ids = list(f.GetAtomIds())
        if not atom_ids:
            continue
        coords = np.array([list(conf.GetAtomPosition(aid)) for aid in atom_ids],dtype=np.float32)
        center = coords.mean(axis=0)
        centers.append(center)
        types.append(fam)
    if len(centers)==0:
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)],dtype=np.float32)
        center = coords.mean(axis=0)
        centers = [center]
        types = ["Hydrophobe"]
    centers = np.stack(centers,axis=0)
    num_pharma = centers.shape[0]
    pharma_features = []
    for t,c in zip(types,centers):
        one_hot = np.zeros(len(PHARMA_FAMILIES),dtype=np.float32)
        one_hot[_PHARMA_FAMILY_TO_IDX[t]] = 1.0
        f = np.concatenate([one_hot,c.astype(np.float32)],axis=0)
        pharma_features.append(f)
    pharma_features = np.stack(pharma_features,axis=0)
    atom_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)],dtype=np.float32)
    diff = atom_coords[:,None,:]-centers[None,:,:]
    dists = np.linalg.norm(diff,axis=-1)
    atom_to_pharma = np.argmin(dists,axis=-1).astype(np.int64)
    return pharma_features,centers,atom_to_pharma

def featurize_pharmacophore_pairs(pharma_centers: np.ndarray,num_distance_bins: int,max_dist: float=20.0) -> np.ndarray:
    num_pharma = pharma_centers.shape[0]
    diff = pharma_centers[:,None,:]-pharma_centers[None,:,:]
    dists = np.linalg.norm(diff,axis=-1)
    bin_edges = np.linspace(0.0,max_dist,num_distance_bins+1)
    bin_indices = np.digitize(dists,bin_edges)-1
    bin_indices = np.clip(bin_indices,0,num_distance_bins-1)
    dist_one_hot = np.zeros((num_pharma,num_pharma,num_distance_bins),dtype=np.float32)
    for i in range(num_pharma):
        for j in range(num_pharma):
            idx = bin_indices[i,j]
            dist_one_hot[i,j,idx] = 1.0
    zeros = np.zeros((num_pharma,num_pharma,64),dtype=np.float32)
    pharma_pair_features = np.concatenate([zeros,dist_one_hot],axis=-1)
    return pharma_pair_features

def featurize_mol_single_conformer(mol: Chem.Mol,conf_id: Optional[int],num_distance_bins: int,num_pair_distance_basis: int=32) -> Dict[str,np.ndarray]:
    if conf_id is None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
        conf_id = 0
    num_atoms = mol.GetNumAtoms()
    atom_feats_np = featurize_atoms(mol)
    pair_feats_np = featurize_atom_pairs(mol,conf_id=conf_id,num_distance_basis=num_pair_distance_basis)
    pharma_feats_np,pharma_centers_np,atom_to_pharma_np = featurize_pharmacophores(mol,conf_id)
    pharma_pair_feats_np = featurize_pharmacophore_pairs(pharma_centers_np,num_distance_bins=num_distance_bins,max_dist=20.0)
    atom_mask_np = np.ones((num_atoms,),dtype=np.float32)
    return {
        "atom_features":atom_feats_np,
        "pharmacophore_features":pharma_feats_np,
        "atom_to_pharmacophore":atom_to_pharma_np,
        "pair_features":pair_feats_np,
        "pharmacophore_pair_features":pharma_pair_feats_np,
        "atom_mask":atom_mask_np,
    }

def featurize_mol_all_conformers(mol: Chem.Mol,num_distance_bins: int,max_conformers: Optional[int]=None,num_pair_distance_basis: int=32) -> List[Dict[str,np.ndarray]]:
    if mol is None:
        return []
    if mol.GetNumConformers()==0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    if max_conformers is not None:
        conf_ids = conf_ids[:max_conformers]
    conformer_batch = []
    for cid in conf_ids:
        conf_data = featurize_mol_single_conformer(mol,conf_id=cid,num_distance_bins=num_distance_bins,num_pair_distance_basis=num_pair_distance_basis)
        conformer_batch.append(conf_data)
    return conformer_batch

def featurize_sdf_file(sdf_path: str,num_distance_bins: int,max_conformers_per_mol: Optional[int]=None,sanitize: bool=True) -> List[List[Dict[str,np.ndarray]]]:
    suppl = Chem.SDMolSupplier(sdf_path,removeHs=False,sanitize=sanitize)
    mol_conformer_batches: List[List[Dict[str,np.ndarray]]] = []
    for mol in suppl:
        if mol is None:
            continue
        conf_batch = featurize_mol_all_conformers(mol,num_distance_bins=num_distance_bins,max_conformers=max_conformers_per_mol)
        if len(conf_batch)==0:
            continue
        mol_conformer_batches.append(conf_batch)
    return mol_conformer_batches

if __name__=="__main__":
    sdf_path = "example.sdf"
    num_distance_bins = 32
    mol_conformer_batches = featurize_sdf_file(sdf_path,num_distance_bins=num_distance_bins,max_conformers_per_mol=4)
    print("Num molecules in SDF:",len(mol_conformer_batches))
    if len(mol_conformer_batches)>0:
        first_conf = mol_conformer_batches[0][0]
        for k,v in first_conf.items():
            print(k,":",v.shape)
