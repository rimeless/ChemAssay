"""
main.py — PU-Aware Contrastive Learning for Assay-Compound Virtual Screening
=============================================================================
End-to-end pipeline: data → model → 3-phase training → evaluation → screening

Usage:
    python main.py                          # Run with synthetic data (demo)
    python main.py --data path/to/data.csv  # Run with real PubChem data
"""

import argparse
import os
import json
import time

import numpy as np
import pandas as pd
import torch

from dataset import (
    AssayCompoundPUDataset,
    FIELD_CONFIGS,
    generate_synthetic_data,
    collate_fn,
)
from model import PUContrastiveVSModel
from loss import CurriculumPULoss
from trainer import PUContrastiveTrainer, evaluate, virtual_screen


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Data
    "max_atoms": 32,
    "n_apm_features": 8,
    "use_3d": False,
    "train_ratio": 0.8,

    # Model
    "d_embed": 256,
    "d_text": 256,
    "d_meta": 128,
    "d_ctx": 64,
    "conv_channels": 64,
    "lstm_hidden": 128,
    "n_attn_heads": 4,
    "gate_type": "bilinear",
    "pi_min": 0.001,
    "pi_max": 0.30,
    "dropout": 0.1,

    # Training
    "n_epochs": 25,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "tau": 0.07,
    "lambda_rn": 1.0,
    "lambda_u": 1.0,
    "prior_reg_weight": 0.1,
    "phase1_end": 5,
    "phase2_end": 15,
    "bbe_interval": 5,
    "scheduler_T0": 10,
    "num_workers": 0,

    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "seed": 42,
}


def set_seed(seed: int):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_assay(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple:
    """
    Split data by assay (not by compound) to test cross-assay generalization.
    All compounds from a given assay go into either train or val.
    """
    rng = np.random.RandomState(seed)
    assay_ids = df["assay_id"].unique()
    rng.shuffle(assay_ids)

    n_train = int(len(assay_ids) * train_ratio)
    train_aids = set(assay_ids[:n_train])
    val_aids = set(assay_ids[n_train:])

    df_train = df[df["assay_id"].isin(train_aids)].reset_index(drop=True)
    df_val = df[df["assay_id"].isin(val_aids)].reset_index(drop=True)

    print(f"[Split] Train: {len(df_train)} samples, {len(train_aids)} assays")
    print(f"[Split] Val:   {len(df_val)} samples, {len(val_aids)} assays")

    return df_train, df_val


def run_demo(config: dict):
    """Run full pipeline with synthetic data."""
    print("\n" + "=" * 70)
    print("  PU-Aware Contrastive VS — Demo with Synthetic Data")
    print("=" * 70)

    set_seed(config["seed"])

    # ─── 1. Generate Data ───
    print("\n[1/5] Generating synthetic PubChem-like data...")
    df = generate_synthetic_data(n_assays=30, n_compounds_per_assay=100, seed=config["seed"])
    print(f"  Generated {len(df)} total records")
    print(f"  Assay types: {df['assay_type'].value_counts().to_dict()}")
    print(f"  Outcomes:    {df['outcome'].value_counts().to_dict()}")

    # ─── 2. Split ───
    print("\n[2/5] Splitting by assay...")
    df_train, df_val = split_by_assay(df, train_ratio=config["train_ratio"], seed=config["seed"])

    # ─── 3. Build Datasets ───
    print("\n[3/5] Building PU-aware datasets...")
    train_ds = AssayCompoundPUDataset(
        df_train,
        max_atoms=config["max_atoms"],
        n_apm_features=config["n_apm_features"],
        use_3d=config["use_3d"],
    )
    val_ds = AssayCompoundPUDataset(
        df_val,
        max_atoms=config["max_atoms"],
        n_apm_features=config["n_apm_features"],
        use_3d=config["use_3d"],
    )

    # Verify a sample
    sample = train_ds[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    print(f"  APM shape:   {sample['apm'].shape}")
    print(f"  PU label:    {sample['pu_label'].item()}")
    print(f"  Alpha:       {sample['alpha'].item():.3f}")
    print(f"  Beta:        {sample['beta'].item():.3f}")

    # ─── 4. Build Model ───
    print("\n[4/5] Initializing model...")
    model = PUContrastiveVSModel(
        field_configs=FIELD_CONFIGS,
        max_atoms=config["max_atoms"],
        n_apm_features=config["n_apm_features"],
        d_embed=config["d_embed"],
        d_text=config["d_text"],
        d_meta=config["d_meta"],
        d_ctx=config["d_ctx"],
        conv_channels=config["conv_channels"],
        lstm_hidden=config["lstm_hidden"],
        n_attn_heads=config["n_attn_heads"],
        gate_type=config["gate_type"],
        pi_min=config["pi_min"],
        pi_max=config["pi_max"],
        dropout=config["dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # ─── 5. Train ───
    print("\n[5/5] Starting training...")
    trainer = PUContrastiveTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
    )

    history = trainer.train()

    # ─── Final Evaluation ───
    print("\n[Final] Loading best model and evaluating...")
    ckpt_path = os.path.join(config["save_dir"], "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=config["device"], weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best model from epoch {ckpt['epoch']}")

    device = torch.device(config["device"])
    model.to(device)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, collate_fn=collate_fn,
    )

    final_metrics = evaluate(model, val_loader, device)
    print("\n  Final Validation Metrics:")
    for k, v in sorted(final_metrics.items()):
        print(f"    {k:25s}: {v:.4f}")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print("  Pipeline Summary")
    print(f"{'='*70}")
    print(f"  Data:   {len(df)} records, {df['assay_id'].nunique()} assays")
    print(f"  Model:  {total_params:,} params")
    print(f"  Best:   epoch {trainer.best_epoch}, BEDROC={trainer.best_metric:.4f}")
    print(f"  EF1%:   {final_metrics.get('mean_ef1', 0):.2f}")
    print(f"  AUROC:  {final_metrics.get('mean_auroc', 0):.4f}")
    print(f"  BEDROC: {final_metrics.get('mean_bedroc', 0):.4f}")

    return model, history, final_metrics


def run_with_data(data_path: str, config: dict):
    """Run pipeline with real data from CSV."""
    print(f"\n[Data] Loading from {data_path}")
    df = pd.read_csv(data_path)

    required_cols = ["smiles", "assay_id", "assay_type", "outcome"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Loaded {len(df)} records, {df['assay_id'].nunique()} assays")

    # Fill optional columns
    defaults = {
        "assay_description": "Unspecified bioassay",
        "organism": "<unk>",
        "cell_line": "<unk>",
        "target_type": "<unk>",
        "detection_method": "<unk>",
        "n_tested": 100,
        "active_ratio": 0.01,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Same pipeline as demo
    set_seed(config["seed"])
    df_train, df_val = split_by_assay(df, config["train_ratio"], config["seed"])

    train_ds = AssayCompoundPUDataset(df_train, max_atoms=config["max_atoms"],
                                       n_apm_features=config["n_apm_features"], use_3d=config["use_3d"])
    val_ds = AssayCompoundPUDataset(df_val, max_atoms=config["max_atoms"],
                                     n_apm_features=config["n_apm_features"], use_3d=config["use_3d"])

    model = PUContrastiveVSModel(
        field_configs=FIELD_CONFIGS,
        max_atoms=config["max_atoms"], n_apm_features=config["n_apm_features"],
        d_embed=config["d_embed"], d_text=config["d_text"], d_meta=config["d_meta"],
        d_ctx=config["d_ctx"], conv_channels=config["conv_channels"],
        lstm_hidden=config["lstm_hidden"], n_attn_heads=config["n_attn_heads"],
        gate_type=config["gate_type"], pi_min=config["pi_min"], pi_max=config["pi_max"],
        dropout=config["dropout"],
    )

    trainer = PUContrastiveTrainer(model=model, train_dataset=train_ds, val_dataset=val_ds, config=config)
    history = trainer.train()
    return model, history


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="PU-Aware Contrastive VS Training")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_embed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    # Override with CLI args
    for key in ["n_epochs", "batch_size", "lr", "d_embed", "device", "save_dir", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    if args.data:
        model, history = run_with_data(args.data, config)
    else:
        model, history, metrics = run_demo(config)
