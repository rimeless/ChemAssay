"""
dataset.py — PU-aware Dataset for Assay-Compound Contrastive Learning
=====================================================================
Handles PubChem BioAssay data with multi-fidelity label mapping:
  (assay_type × outcome) → {P, P_soft, RN, RN_soft, U} with confidence weights.

Label encoding (internal):
    +1  = Positive (P or P_soft)
    -1  = Reliable Negative (RN or RN_soft)
     0  = Unlabeled (U)

Each sample carries:
    alpha_i : positive contribution weight  (attraction strength)
    beta_i  : negative contribution weight  (repulsion strength)
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARNING] RDKit not available. Using hash-based APM fallback.")

# ───────────────────────────────────────────────────────────────
# 1.  PU Label Mapping Configuration
# ───────────────────────────────────────────────────────────────

# (assay_type, outcome) → (pu_label, alpha, beta)
# pu_label: +1 = P, -1 = RN, 0 = U
# alpha: positive contribution weight (attraction)
# beta:  negative contribution weight (repulsion)

DEFAULT_CONFIRMATION_RATE = 0.36  # avg. fraction of screening actives confirmed
DEFAULT_FN_RATE = 0.01           # false negative rate among screening inactives

PU_LABEL_MAP = {
    # Confirmatory assays — high confidence
    ("confirmatory", "active"):        (+1, 0.95, 0.00),
    ("confirmatory", "probe"):         (+1, 1.00, 0.00),
    ("confirmatory", "inactive"):      (-1, 0.00, 0.90),
    ("confirmatory", "inconclusive"):  ( 0, None, None),  # will use prior
    ("confirmatory", "unspecified"):   ( 0, None, None),

    # Screening (primary HTS) — noisy labels
    ("screening", "active"):           (+1, DEFAULT_CONFIRMATION_RATE, 1.0 - DEFAULT_CONFIRMATION_RATE),
    ("screening", "inactive"):         (-1, DEFAULT_FN_RATE, 1.0 - DEFAULT_FN_RATE),
    ("screening", "inconclusive"):     ( 0, None, None),
    ("screening", "unspecified"):      ( 0, None, None),

    # Other / Summary — moderate confidence
    ("other", "active"):               (+1, 0.60, 0.40),
    ("other", "inactive"):             (-1, 0.05, 0.85),
    ("other", "inconclusive"):         ( 0, None, None),
    ("other", "unspecified"):          ( 0, None, None),
    ("summary", "active"):             (+1, 0.70, 0.30),
    ("summary", "inactive"):           (-1, 0.03, 0.87),
    ("summary", "inconclusive"):       ( 0, None, None),
    ("summary", "unspecified"):        ( 0, None, None),
}


def assign_pu_label(
    assay_type: str,
    outcome: str,
    assay_prior: float = 0.01,
    confirmation_rate: Optional[float] = None,
) -> Tuple[int, float, float]:
    """
    Map (assay_type, outcome) → (pu_label, alpha, beta).

    For Unlabeled samples (inconclusive / unspecified / untested),
    alpha = pi_assay, beta = 1 - pi_assay  (assay-specific prior).

    Parameters
    ----------
    assay_type : str
        One of 'confirmatory', 'screening', 'other', 'summary'.
    outcome : str
        One of 'active', 'inactive', 'inconclusive', 'unspecified', 'probe'.
    assay_prior : float
        Estimated P(active | assay) — used for unlabeled samples.
    confirmation_rate : float, optional
        Assay-specific confirmation rate overriding the default 0.36
        for screening actives (computed from linked confirmatory assay).

    Returns
    -------
    (pu_label, alpha, beta)
    """
    assay_type = assay_type.lower().strip()
    outcome = outcome.lower().strip()

    key = (assay_type, outcome)
    if key not in PU_LABEL_MAP:
        # Fallback: treat as unlabeled
        return (0, assay_prior, 1.0 - assay_prior)

    pu_label, alpha, beta = PU_LABEL_MAP[key]

    # Override screening active confirmation rate if available
    if assay_type == "screening" and outcome == "active" and confirmation_rate is not None:
        alpha = confirmation_rate
        beta = 1.0 - confirmation_rate

    # Unlabeled: use assay prior
    if alpha is None:
        alpha = assay_prior
        beta = 1.0 - assay_prior

    return (pu_label, float(alpha), float(beta))


# ───────────────────────────────────────────────────────────────
# 2.  Atom Pair Map (APM) Computation
# ───────────────────────────────────────────────────────────────

# Pharmacophore feature types
PHARM_FEATURES = ["Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable"]


def compute_apm_hash(smiles: str, max_atoms: int = 64, n_features: int = 8) -> Optional[np.ndarray]:
    """
    Hash-based APM fallback when RDKit is not available.
    Produces a deterministic pseudo-APM from SMILES string.
    NOT chemically meaningful — for pipeline testing only.
    """
    rng = np.random.RandomState(hash(smiles) % (2**31))
    n_atoms = min(len(smiles) // 2 + 1, max_atoms)
    n_atoms = max(n_atoms, 2)
    apm = np.zeros((max_atoms, max_atoms, n_features), dtype=np.float32)
    for i in range(n_atoms):
        for j in range(n_atoms):
            apm[i, j, :] = rng.randn(n_features).astype(np.float32) * 0.3
            # Add distance-like decay
            apm[i, j, 4] = abs(i - j) / max(n_atoms, 1)
    return apm


def compute_apm(smiles: str, max_atoms: int = 64, n_features: int = 8) -> Optional[np.ndarray]:
    """
    Compute Atom Pair Map: a (max_atoms, max_atoms, n_features) tensor encoding
    pairwise atomic properties and spatial/topological distances.

    Features per atom pair (i, j):
        0: atomic number of atom i  (normalized)
        1: atomic number of atom j  (normalized)
        2: Gasteiger partial charge of atom i
        3: Gasteiger partial charge of atom j
        4: topological distance (shortest path)
        5: 3D Euclidean distance  (from ETKDG conformer, 0 if generation fails)
        6: is_same_ring  (1 if i,j share a ring)
        7: combined pharmacophore compatibility score

    Returns None if SMILES is invalid.
    """
    if not HAS_RDKIT:
        return compute_apm_hash(smiles, max_atoms, n_features)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()

    if n_atoms == 0 or n_atoms > max_atoms:
        return None

    # --- Gasteiger charges ---
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in range(n_atoms)]
        charges = [0.0 if np.isnan(c) or np.isinf(c) else c for c in charges]
    except Exception:
        charges = [0.0] * n_atoms

    # --- Atomic numbers (normalized by max common = 53 for Iodine) ---
    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() / 53.0 for i in range(n_atoms)]

    # --- Topological distance matrix ---
    dist_matrix = Chem.GetDistanceMatrix(mol)

    # --- 3D distance (ETKDG) ---
    conf_3d = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()
            for i in range(n_atoms):
                pi = conf.GetAtomPosition(i)
                for j in range(i + 1, n_atoms):
                    pj = conf.GetAtomPosition(j)
                    d = pi.Distance(pj)
                    conf_3d[i, j] = d
                    conf_3d[j, i] = d
    except Exception:
        pass  # 3D distances remain 0

    # --- Ring membership ---
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRingSizes()

    def share_ring(i, j):
        for ring in ring_info.AtomRings():
            if i in ring and j in ring:
                return 1.0
        return 0.0

    # --- Build APM tensor ---
    apm = np.zeros((max_atoms, max_atoms, n_features), dtype=np.float32)

    for i in range(n_atoms):
        for j in range(n_atoms):
            apm[i, j, 0] = atomic_nums[i]
            apm[i, j, 1] = atomic_nums[j]
            apm[i, j, 2] = charges[i]
            apm[i, j, 3] = charges[j]
            apm[i, j, 4] = dist_matrix[i, j] / max(dist_matrix.max(), 1.0)  # normalized
            apm[i, j, 5] = conf_3d[i, j] / max(conf_3d.max(), 1.0) if conf_3d.max() > 0 else 0.0
            apm[i, j, 6] = share_ring(i, j)
            # Feature 7: simple compatibility (same type similarity)
            apm[i, j, 7] = 1.0 if mol.GetAtomWithIdx(i).GetAtomicNum() == mol.GetAtomWithIdx(j).GetAtomicNum() else 0.0

    return apm


def compute_apm_fast(smiles: str, max_atoms: int = 64, n_features: int = 8) -> Optional[np.ndarray]:
    """
    Lightweight APM without 3D conformer generation — much faster for large-scale training.
    Uses topological distance only (feature[5] = 0).
    """
    if not HAS_RDKIT:
        return compute_apm_hash(smiles, max_atoms, n_features)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_atoms > max_atoms:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in range(n_atoms)]
        charges = [0.0 if np.isnan(c) or np.isinf(c) else c for c in charges]
    except Exception:
        charges = [0.0] * n_atoms

    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() / 53.0 for i in range(n_atoms)]
    dist_matrix = Chem.GetDistanceMatrix(mol)
    max_dist = max(dist_matrix.max(), 1.0)

    ring_info = mol.GetRingInfo()

    def share_ring(i, j):
        for ring in ring_info.AtomRings():
            if i in ring and j in ring:
                return 1.0
        return 0.0

    apm = np.zeros((max_atoms, max_atoms, n_features), dtype=np.float32)
    for i in range(n_atoms):
        for j in range(n_atoms):
            apm[i, j, 0] = atomic_nums[i]
            apm[i, j, 1] = atomic_nums[j]
            apm[i, j, 2] = charges[i]
            apm[i, j, 3] = charges[j]
            apm[i, j, 4] = dist_matrix[i, j] / max_dist
            apm[i, j, 5] = 0.0  # no 3D
            apm[i, j, 6] = share_ring(i, j)
            apm[i, j, 7] = 1.0 if mol.GetAtomWithIdx(i).GetAtomicNum() == mol.GetAtomWithIdx(j).GetAtomicNum() else 0.0

    return apm


# ───────────────────────────────────────────────────────────────
# 3.  Assay Text + Metadata Processing
# ───────────────────────────────────────────────────────────────

# Vocabulary indices for structured metadata fields
ASSAY_TYPE_VOCAB = {"screening": 0, "confirmatory": 1, "summary": 2, "other": 3, "<unk>": 4}
ORGANISM_VOCAB_SIZE = 200   # placeholder — built from data
CELL_LINE_VOCAB_SIZE = 500
TARGET_TYPE_VOCAB_SIZE = 100
DETECTION_VOCAB_SIZE = 50

FIELD_CONFIGS = {
    "assay_type": len(ASSAY_TYPE_VOCAB),
    "organism": ORGANISM_VOCAB_SIZE,
    "cell_line": CELL_LINE_VOCAB_SIZE,
    "target_type": TARGET_TYPE_VOCAB_SIZE,
    "detection_method": DETECTION_VOCAB_SIZE,
}


def tokenize_assay_text(text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Tokenize assay description text.
    In production, uses PubChemDeBERTa tokenizer.
    Here, returns character-level placeholder for self-contained code.
    """
    # Placeholder: simple word-level tokenization with vocabulary
    # Replace with: tokenizer = AutoTokenizer.from_pretrained("your-pubchem-deberta")
    words = text.lower().split()[:max_length]
    # Encode as integer ids (hash-based for demo)
    ids = [hash(w) % 30000 for w in words]
    ids = ids + [0] * (max_length - len(ids))  # pad
    attention_mask = [1] * min(len(words), max_length) + [0] * (max_length - min(len(words), max_length))

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def encode_structured_metadata(metadata: Dict) -> Dict[str, torch.Tensor]:
    """
    Encode structured assay metadata fields as integer indices.
    """
    encoded = {}
    for field, vocab_size in FIELD_CONFIGS.items():
        val = metadata.get(field, "<unk>")
        if field == "assay_type":
            idx = ASSAY_TYPE_VOCAB.get(str(val).lower(), ASSAY_TYPE_VOCAB["<unk>"])
        else:
            idx = hash(str(val)) % vocab_size
        encoded[field] = torch.tensor(idx, dtype=torch.long)
    return encoded


# ───────────────────────────────────────────────────────────────
# 4.  Dataset Class
# ───────────────────────────────────────────────────────────────

class AssayCompoundPUDataset(Dataset):
    """
    PU-aware dataset for assay-compound contrastive learning.

    Expected data format (DataFrame or CSV):
        - smiles: str           SMILES string of compound
        - assay_id: str         Unique assay identifier
        - assay_type: str       'screening' | 'confirmatory' | 'summary' | 'other'
        - outcome: str          'active' | 'inactive' | 'inconclusive' | 'unspecified' | 'probe'
        - assay_description: str    Free-text protocol description
        - organism: str         (optional) Target organism
        - cell_line: str        (optional) Cell line used
        - target_type: str      (optional) e.g., 'protein', 'gene', 'pathway'
        - detection_method: str (optional) e.g., 'fluorescence', 'luminescence'
        - n_tested: int         (optional) Number of compounds tested in this assay
        - active_ratio: float   (optional) Fraction of actives in this assay
        - confirmation_rate: float (optional) Assay-specific confirmation rate

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with columns above.
    max_atoms : int
        Maximum number of atoms for APM (compounds with more are skipped).
    use_3d : bool
        Whether to compute 3D conformers for APM (slower but richer).
    precompute_apm : bool
        If True, compute all APMs upfront. If False, compute on-the-fly.
    artifact_smiles : set, optional
        Set of SMILES to exclude (aggregators, PAINS, frequent hitters).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        max_atoms: int = 64,
        n_apm_features: int = 8,
        use_3d: bool = False,
        precompute_apm: bool = False,
        artifact_smiles: Optional[set] = None,
    ):
        self.max_atoms = max_atoms
        self.n_apm_features = n_apm_features
        self.use_3d = use_3d
        self.apm_fn = compute_apm if use_3d else compute_apm_fast

        # --- Artifact filtering ---
        if artifact_smiles:
            before = len(data)
            data = data[~data["smiles"].isin(artifact_smiles)].reset_index(drop=True)
            print(f"[Dataset] Filtered {before - len(data)} artifact compounds")

        # --- Compute PU labels & weights ---
        pu_labels, alphas, betas = [], [], []
        for _, row in data.iterrows():
            assay_prior = row.get("active_ratio", 0.01)
            if pd.isna(assay_prior) or assay_prior <= 0:
                assay_prior = 0.01
            conf_rate = row.get("confirmation_rate", None)
            if pd.notna(conf_rate):
                conf_rate = float(conf_rate)
            else:
                conf_rate = None

            label, alpha, beta = assign_pu_label(
                assay_type=str(row.get("assay_type", "other")),
                outcome=str(row.get("outcome", "unspecified")),
                assay_prior=assay_prior,
                confirmation_rate=conf_rate,
            )
            pu_labels.append(label)
            alphas.append(alpha)
            betas.append(beta)

        data = data.copy()
        data["pu_label"] = pu_labels
        data["alpha"] = alphas
        data["beta"] = betas

        self.data = data.reset_index(drop=True)

        # --- Precompute APMs if requested ---
        self.apm_cache = {}
        if precompute_apm:
            print("[Dataset] Precomputing APMs...")
            unique_smiles = self.data["smiles"].unique()
            for i, smi in enumerate(unique_smiles):
                apm = self.apm_fn(smi, max_atoms=max_atoms, n_features=n_apm_features)
                if apm is not None:
                    self.apm_cache[smi] = apm
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(unique_smiles)} APMs computed")
            print(f"[Dataset] Cached {len(self.apm_cache)} APMs")

        # --- Build assay-level metadata index ---
        self.assay_meta = {}
        for aid in self.data["assay_id"].unique():
            subset = self.data[self.data["assay_id"] == aid].iloc[0]
            self.assay_meta[aid] = {
                "description": str(subset.get("assay_description", "")),
                "assay_type": str(subset.get("assay_type", "other")),
                "organism": str(subset.get("organism", "<unk>")),
                "cell_line": str(subset.get("cell_line", "<unk>")),
                "target_type": str(subset.get("target_type", "<unk>")),
                "detection_method": str(subset.get("detection_method", "<unk>")),
                "n_tested": int(subset.get("n_tested", 0)),
                "active_ratio": float(subset.get("active_ratio", 0.01)),
            }

        self._print_stats()

    def _print_stats(self):
        labels = self.data["pu_label"]
        n_p = (labels == 1).sum()
        n_rn = (labels == -1).sum()
        n_u = (labels == 0).sum()
        print(f"[Dataset] Total samples: {len(self.data)}")
        print(f"  P (positive):          {n_p}  ({100*n_p/len(self.data):.1f}%)")
        print(f"  RN (reliable negative): {n_rn}  ({100*n_rn/len(self.data):.1f}%)")
        print(f"  U  (unlabeled):         {n_u}  ({100*n_u/len(self.data):.1f}%)")
        print(f"  Unique assays:          {self.data['assay_id'].nunique()}")
        print(f"  Unique compounds:       {self.data['smiles'].nunique()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        assay_id = row["assay_id"]

        # --- Compound: APM ---
        if smiles in self.apm_cache:
            apm = self.apm_cache[smiles]
        else:
            apm = self.apm_fn(smiles, max_atoms=self.max_atoms, n_features=self.n_apm_features)
            if apm is None:
                # Fallback: zero tensor
                apm = np.zeros((self.max_atoms, self.max_atoms, self.n_apm_features), dtype=np.float32)

        apm_tensor = torch.from_numpy(apm)  # (max_atoms, max_atoms, n_features)

        # --- Assay: text + structured metadata ---
        meta = self.assay_meta[assay_id]
        text_tokens = tokenize_assay_text(meta["description"])
        struct_meta = encode_structured_metadata(meta)

        # --- Contextual features ---
        ctx_features = torch.tensor([
            np.log1p(meta["n_tested"]) / 15.0,  # log-normalized
            meta["active_ratio"],
        ], dtype=torch.float32)

        # --- PU labels & weights ---
        pu_label = torch.tensor(row["pu_label"], dtype=torch.long)
        alpha = torch.tensor(row["alpha"], dtype=torch.float32)
        beta = torch.tensor(row["beta"], dtype=torch.float32)

        return {
            "apm": apm_tensor,
            "text_input_ids": text_tokens["input_ids"],
            "text_attention_mask": text_tokens["attention_mask"],
            "struct_meta": struct_meta,
            "ctx_features": ctx_features,
            "pu_label": pu_label,
            "alpha": alpha,
            "beta": beta,
            "assay_id": assay_id,
            "smiles": smiles,
        }


# ───────────────────────────────────────────────────────────────
# 5.  Assay-Quality Weighted Sampler
# ───────────────────────────────────────────────────────────────

def build_quality_weighted_sampler(dataset: AssayCompoundPUDataset) -> WeightedRandomSampler:
    """
    Build a weighted sampler that up-weights:
      1. Confirmatory assay samples (higher quality)
      2. Positive samples (rare but critical)
      3. Assays with many tested compounds (more reliable prior)
    """
    weights = []
    for idx in range(len(dataset)):
        row = dataset.data.iloc[idx]
        w = 1.0

        # Assay type bonus
        atype = str(row.get("assay_type", "other")).lower()
        if atype == "confirmatory":
            w *= 2.0
        elif atype == "screening":
            w *= 1.0
        else:
            w *= 0.5

        # Positive up-weighting (handle class imbalance)
        if row["pu_label"] == 1:
            w *= 5.0
        elif row["pu_label"] == 0:
            w *= 0.5

        # Compound count reliability
        n_tested = row.get("n_tested", 50)
        if pd.notna(n_tested) and n_tested > 0:
            w *= min(np.log1p(n_tested) / 10.0, 1.5)

        weights.append(w)

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ───────────────────────────────────────────────────────────────
# 6.  Collate Function
# ───────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for variable-length metadata."""
    apm = torch.stack([b["apm"] for b in batch])
    text_ids = torch.stack([b["text_input_ids"] for b in batch])
    text_mask = torch.stack([b["text_attention_mask"] for b in batch])
    ctx = torch.stack([b["ctx_features"] for b in batch])
    pu_label = torch.stack([b["pu_label"] for b in batch])
    alpha = torch.stack([b["alpha"] for b in batch])
    beta = torch.stack([b["beta"] for b in batch])

    # Structured metadata — stack per field
    struct_meta = {}
    for field in batch[0]["struct_meta"]:
        struct_meta[field] = torch.stack([b["struct_meta"][field] for b in batch])

    return {
        "apm": apm,
        "text_input_ids": text_ids,
        "text_attention_mask": text_mask,
        "struct_meta": struct_meta,
        "ctx_features": ctx,
        "pu_label": pu_label,
        "alpha": alpha,
        "beta": beta,
        "assay_ids": [b["assay_id"] for b in batch],
        "smiles_list": [b["smiles"] for b in batch],
    }


# ───────────────────────────────────────────────────────────────
# 7.  Synthetic Data Generator (for testing)
# ───────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_assays: int = 50,
    n_compounds_per_assay: int = 200,
    active_ratio: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic PubChem-like data for testing the pipeline.
    """
    rng = np.random.RandomState(seed)

    # Simple valid SMILES pool
    smiles_pool = [
        "c1ccccc1", "CC(=O)O", "CCO", "CC(=O)Nc1ccc(O)cc1",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C1CCCCC1", "c1ccncc1",
        "CC(=O)OC1=CC=CC=C1C(=O)O", "OC(=O)C1=CC=CC=C1O", "CCN(CC)CC",
        "CC1=CC=C(C=C1)NC(=O)C", "C1=CC=C(C=C1)C(=O)O", "CC(C)O",
        "C(C(=O)O)N", "C1=CC=C(C=C1)O", "CC=O", "CCCC", "C(CO)O",
        "CC(=O)C", "C1=CC=NC=C1", "c1ccc(cc1)N", "CC(C)(C)O",
        "OC(=O)c1cccc(O)c1", "Clc1ccccc1", "FC(F)(F)c1ccccc1",
        "c1ccc2c(c1)cccc2", "CC(=O)NC1=CC=C(C=C1)Cl",
        "OC(=O)CCc1ccccc1", "NCC(=O)O", "CC(N)C(=O)O",
    ]

    records = []
    assay_types = ["screening", "screening", "screening", "confirmatory", "other"]
    outcomes_screening = ["inactive"] * 95 + ["active"] * 3 + ["inconclusive"] + ["unspecified"]
    outcomes_confirm = ["inactive"] * 65 + ["active"] * 30 + ["inconclusive"] * 3 + ["unspecified"] * 2

    for a_idx in range(n_assays):
        assay_id = f"AID_{a_idx:05d}"
        atype = rng.choice(assay_types)
        outcomes = outcomes_confirm if atype == "confirmatory" else outcomes_screening
        n_compounds = rng.randint(50, n_compounds_per_assay + 1)

        for c_idx in range(n_compounds):
            smi = rng.choice(smiles_pool)
            outcome = rng.choice(outcomes)
            records.append({
                "smiles": smi,
                "assay_id": assay_id,
                "assay_type": atype,
                "outcome": outcome,
                "assay_description": f"This is a {atype} assay targeting protein kinase {a_idx % 10} "
                                     f"using {rng.choice(['fluorescence', 'luminescence', 'absorbance'])} "
                                     f"detection in {rng.choice(['HEK293', 'HeLa', 'MCF7'])} cells.",
                "organism": rng.choice(["Homo sapiens", "Mus musculus", "Rattus norvegicus"]),
                "cell_line": rng.choice(["HEK293", "HeLa", "MCF7", "A549", "<unk>"]),
                "target_type": rng.choice(["protein", "gene", "pathway"]),
                "detection_method": rng.choice(["fluorescence", "luminescence", "absorbance"]),
                "n_tested": n_compounds,
                "active_ratio": 0.03 if atype == "screening" else 0.30,
            })

    return pd.DataFrame(records)


# ───────────────────────────────────────────────────────────────
# Quick test
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating synthetic data ===")
    df = generate_synthetic_data(n_assays=10, n_compounds_per_assay=50)
    print(f"Generated {len(df)} records\n")

    print("=== Building dataset ===")
    ds = AssayCompoundPUDataset(df, max_atoms=32, use_3d=False, precompute_apm=False)

    print(f"\n=== Testing __getitem__ ===")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    print(f"\n=== Testing DataLoader ===")
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: {{" + ", ".join(f"{kk}: {vv.shape}" for kk, vv in v.items()) + "}")
        else:
            print(f"  {k}: list[{len(v)}]")
