"""
trainer.py — Training & Evaluation Pipeline
============================================
Implements:
    1. 3-Phase Curriculum Training (InfoNCE → PU warm-up → Full CW-SemiPU)
    2. BBE Prior Estimation (periodic re-estimation)
    3. Evaluation: EF1%, BEDROC, AUROC, Hit Rate @K
    4. Retrieval-based Virtual Screening
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import PUContrastiveVSModel
from loss import CurriculumPULoss
from dataset import AssayCompoundPUDataset, collate_fn, build_quality_weighted_sampler


# ═══════════════════════════════════════════════════════════════
# 1.  BBE Prior Estimator
# ═══════════════════════════════════════════════════════════════

def estimate_prior_bbe(
    model: PUContrastiveVSModel,
    dataloader: DataLoader,
    device: torch.device,
    n_bins: int = 20,
) -> Dict[str, float]:
    """
    Best Bin Estimation (Garg et al., NeurIPS 2021) for per-assay class prior.
    Returns: {assay_id: estimated_prior}
    """
    model.eval()
    assay_scores = defaultdict(lambda: {"pos": [], "unl": []})

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else
                ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in batch.items()
            }

            outputs = model(batch_dev)
            sim_diag = (outputs["h_assay"] * outputs["h_compound"]).sum(dim=-1)

            for idx in range(len(batch["pu_label"])):
                aid = batch["assay_ids"][idx]
                score = sim_diag[idx].item()
                label = batch["pu_label"][idx].item()

                if label == 1:
                    assay_scores[aid]["pos"].append(score)
                elif label == 0:
                    assay_scores[aid]["unl"].append(score)

    priors = {}
    for aid, scores in assay_scores.items():
        pos_scores = np.array(scores["pos"])
        unl_scores = np.array(scores["unl"])

        if len(pos_scores) < 5 or len(unl_scores) < 10:
            priors[aid] = 0.01
            continue

        bins = np.linspace(
            min(pos_scores.min(), unl_scores.min()) - 0.01,
            max(pos_scores.max(), unl_scores.max()) + 0.01,
            n_bins + 1,
        )
        p_hist, _ = np.histogram(pos_scores, bins=bins, density=True)
        u_hist, _ = np.histogram(unl_scores, bins=bins, density=True)

        best_bin = np.argmax(p_hist)
        if p_hist[best_bin] > 0:
            pi_est = u_hist[best_bin] / (p_hist[best_bin] + 1e-8)
        else:
            pi_est = 0.01

        priors[aid] = float(np.clip(pi_est, 0.001, 0.3))

    model.train()
    return priors


# ═══════════════════════════════════════════════════════════════
# 2.  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════

def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01) -> float:
    """Enrichment Factor at given fraction (e.g., EF1% for fraction=0.01)."""
    n = len(y_true)
    n_actives = y_true.sum()
    if n_actives == 0 or n == 0:
        return 0.0

    k = max(1, int(n * fraction))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    hits_in_top_k = y_true[top_k_idx].sum()

    expected = n_actives * fraction
    return float(hits_in_top_k / max(expected, 1e-8))


def bedroc_score(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, 2007). Exponentially weights early enrichment."""
    n = len(y_true)
    n_actives = int(y_true.sum())
    if n_actives == 0 or n == 0:
        return 0.0

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    active_ranks = np.where(y_sorted == 1)[0] + 1
    ra = n_actives / n

    sum_exp = np.sum(np.exp(-alpha * active_ranks / n))
    ri_max = np.sum(np.exp(-alpha * np.arange(1, n_actives + 1) / n))
    ri_min = np.sum(np.exp(-alpha * np.arange(n - n_actives + 1, n + 1) / n))
    ri_rand = (ra * (1 - np.exp(-alpha))) / (np.exp(alpha / n) - 1)

    bedroc = (sum_exp - ri_rand) / (ri_max - ri_rand + 1e-8)
    return float(np.clip(bedroc, 0.0, 1.0))


def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    """Fraction of true actives in top-k."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Simple AUROC without sklearn."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    tp = 0
    auc = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            auc += tp
    return auc / max(n_pos * n_neg, 1)


# ═══════════════════════════════════════════════════════════════
# 3.  Evaluator
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: PUContrastiveVSModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model: per-assay retrieval metrics."""
    model.eval()

    all_assay_embs = []
    all_compound_embs = []
    all_labels = []
    all_assay_ids = []

    for batch in dataloader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else
            ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in batch.items()
        }

        h_assay = model.encode_assay(batch_dev)
        h_compound = model.encode_compound(batch_dev)

        all_assay_embs.append(h_assay.cpu())
        all_compound_embs.append(h_compound.cpu())
        all_labels.append(batch["pu_label"])
        all_assay_ids.extend(batch["assay_ids"])

    all_assay_embs = torch.cat(all_assay_embs, dim=0)
    all_compound_embs = torch.cat(all_compound_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_assay_ids = np.array(all_assay_ids)

    unique_assays = np.unique(all_assay_ids)
    metrics_per_assay = defaultdict(list)

    for aid in unique_assays:
        mask = all_assay_ids == aid
        if mask.sum() < 10:
            continue

        assay_emb = all_assay_embs[mask].mean(dim=0, keepdim=True)
        compound_embs = all_compound_embs[mask]
        labels = all_labels[mask]

        # Binary: P → 1, RN → 0, U → exclude from validation
        valid = (labels != 0)
        if valid.sum() < 5:
            continue

        compound_embs_valid = compound_embs[valid]
        labels_valid = (labels[valid] == 1).astype(np.float32)

        if labels_valid.sum() == 0 or labels_valid.sum() == len(labels_valid):
            continue

        scores = (assay_emb * compound_embs_valid).sum(dim=-1).numpy()

        metrics_per_assay["ef1"].append(enrichment_factor(labels_valid, scores, 0.01))
        metrics_per_assay["ef5"].append(enrichment_factor(labels_valid, scores, 0.05))
        metrics_per_assay["bedroc"].append(bedroc_score(labels_valid, scores, alpha=20.0))
        metrics_per_assay["auroc"].append(compute_auroc(labels_valid, scores))
        metrics_per_assay["hit_rate_100"].append(hit_rate_at_k(labels_valid, scores, k=100))

    results = {}
    for metric_name, values in metrics_per_assay.items():
        if values:
            results[f"mean_{metric_name}"] = float(np.mean(values))
            results[f"median_{metric_name}"] = float(np.median(values))
            results[f"std_{metric_name}"] = float(np.std(values))
    results["n_evaluated_assays"] = len(unique_assays)

    model.train()
    return results


# ═══════════════════════════════════════════════════════════════
# 4.  Trainer
# ═══════════════════════════════════════════════════════════════

class PUContrastiveTrainer:
    """
    3-Phase Curriculum Trainer for PU-Aware Contrastive VS.

    Phase 1: InfoNCE warm-up, text encoder frozen
    Phase 2: PU loss warm-up, BBE priors, PU Gate activation
    Phase 3: Full CW-SemiPU with learnable priors
    """

    def __init__(
        self,
        model: PUContrastiveVSModel,
        train_dataset: AssayCompoundPUDataset,
        val_dataset: AssayCompoundPUDataset,
        config: Dict,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)

        # Loss
        self.criterion = CurriculumPULoss(
            phase1_end=config.get("phase1_end", 5),
            phase2_end=config.get("phase2_end", 15),
            tau=config.get("tau", 0.07),
            lambda_rn=config.get("lambda_rn", 1.0),
            lambda_u=config.get("lambda_u", 1.0),
            prior_reg_weight=config.get("prior_reg_weight", 0.1),
        )

        # Optimizer with differential LR
        text_params = list(model.assay_encoder.text_encoder.parameters())
        text_ids = set(id(p) for p in text_params)
        other_params = [p for p in model.parameters() if id(p) not in text_ids]

        self.optimizer = optim.AdamW([
            {"params": other_params, "lr": config.get("lr", 1e-4)},
            {"params": text_params, "lr": config.get("lr", 1e-4) * 0.1},
        ], weight_decay=config.get("weight_decay", 0.01))

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.get("scheduler_T0", 10), T_mult=2,
        )

        # Data loaders
        sampler = build_quality_weighted_sampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.get("batch_size", 128),
            sampler=sampler, collate_fn=collate_fn,
            num_workers=config.get("num_workers", 0), pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.get("batch_size", 128),
            shuffle=False, collate_fn=collate_fn,
            num_workers=config.get("num_workers", 0), pin_memory=True,
        )

        # BBE priors
        self.bbe_priors = {}
        self.bbe_interval = config.get("bbe_interval", 5)

        # Logging
        self.history = {"train": [], "val": []}
        self.best_metric = -float("inf")
        self.best_epoch = 0

    def _move_batch(self, batch: Dict) -> Dict:
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                moved[k] = {kk: vv.to(self.device, non_blocking=True) for kk, vv in v.items()}
            else:
                moved[k] = v
        return moved

    def _get_external_priors(self, assay_ids: List[str]) -> Optional[torch.Tensor]:
        if not self.bbe_priors:
            return None
        priors = [self.bbe_priors.get(aid, 0.01) for aid in assay_ids]
        return torch.tensor(priors, dtype=torch.float32, device=self.device)

    def _freeze_text_encoder(self, freeze: bool = True):
        for param in self.model.assay_encoder.text_encoder.parameters():
            param.requires_grad = not freeze

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        phase = self.criterion.get_phase(epoch)

        if phase == 1:
            self._freeze_text_encoder(freeze=True)
        else:
            self._freeze_text_encoder(freeze=False)

        epoch_metrics = defaultdict(float)
        n_batches = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            pi_ext = self._get_external_priors(batch["assay_ids"])

            loss, metrics = self.criterion(
                sim_matrix=outputs["sim_matrix"],
                pu_labels=batch["pu_label"],
                alphas=batch["alpha"],
                betas=batch["beta"],
                pi_a=outputs["pi_a"],
                pu_weights=outputs["pu_weights"],
                epoch=epoch,
                pi_a_external=pi_ext,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] += v
            n_batches += 1

        self.scheduler.step()

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)
        epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        epoch_metrics["epoch"] = epoch
        return dict(epoch_metrics)

    def train(self):
        """Full training loop with 3-phase curriculum."""
        n_epochs = self.config.get("n_epochs", 30)
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  PU-Aware Contrastive VS Training")
        print(f"  Phase 1 (InfoNCE):     epochs 0-{self.config.get('phase1_end',5)-1}")
        print(f"  Phase 2 (PU warm-up):  epochs {self.config.get('phase1_end',5)}-{self.config.get('phase2_end',15)-1}")
        print(f"  Phase 3 (Full PU):     epochs {self.config.get('phase2_end',15)}-{n_epochs-1}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")

        for epoch in range(n_epochs):
            t0 = time.time()
            phase = self.criterion.get_phase(epoch)

            # BBE Prior Re-estimation (Phase 2+)
            if phase >= 2 and epoch % self.bbe_interval == 0 and epoch > 0:
                print(f"  [BBE] Re-estimating priors at epoch {epoch}...")
                self.bbe_priors = estimate_prior_bbe(self.model, self.train_loader, self.device)
                mean_pi = np.mean(list(self.bbe_priors.values())) if self.bbe_priors else 0
                print(f"  [BBE] {len(self.bbe_priors)} assays, mean π = {mean_pi:.4f}")

            # Train
            train_metrics = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            # Validate
            val_metrics = evaluate(self.model, self.val_loader, self.device)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            # Log
            val_ef1 = val_metrics.get("mean_ef1", 0)
            val_auroc = val_metrics.get("mean_auroc", 0)
            val_bedroc = val_metrics.get("mean_bedroc", 0)
            pu_w = train_metrics.get("pu_weight", 0)

            print(
                f"  Epoch {epoch:3d} | Phase {phase} | PU_w={pu_w:.2f} | "
                f"loss={train_metrics.get('loss_total', 0):.4f} | "
                f"EF1%={val_ef1:.2f} AUROC={val_auroc:.3f} BEDROC={val_bedroc:.3f} | "
                f"{elapsed:.1f}s"
            )

            # Best model checkpoint
            target_metric = val_bedroc
            if target_metric > self.best_metric:
                self.best_metric = target_metric
                self.best_epoch = epoch
                ckpt_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": self.best_metric,
                    "bbe_priors": self.bbe_priors,
                    "config": self.config,
                }, ckpt_path)
                print(f"    ★ New best BEDROC={self.best_metric:.4f} → {ckpt_path}")

        print(f"\n{'='*70}")
        print(f"  Training complete. Best epoch: {self.best_epoch}, Best BEDROC: {self.best_metric:.4f}")
        print(f"{'='*70}")

        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        return self.history


# ═══════════════════════════════════════════════════════════════
# 5.  Retrieval-based Virtual Screening
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def virtual_screen(
    model: PUContrastiveVSModel,
    assay_batch: Dict,
    compound_loader: DataLoader,
    device: torch.device,
    top_k: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Screen a compound library against a target assay.

    Returns:
        scores, indices, smiles_list  (all sorted by descending score)
    """
    model.eval()

    assay_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else
        ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
        for k, v in assay_batch.items()
    }
    h_assay = model.encode_assay(assay_dev)
    pi_a = model.prior_network(h_assay)

    all_scores = []
    all_smiles = []

    for batch in compound_loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else
            ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in batch.items()
        }
        h_compound = model.encode_compound(batch_dev)
        scores = (h_assay * h_compound).sum(dim=-1)
        pu_w = model.pu_gate(h_assay, h_compound, pi_a)
        adjusted = scores * pu_w.squeeze(0)

        all_scores.append(adjusted.cpu().numpy())
        all_smiles.extend(batch.get("smiles_list", []))

    all_scores = np.concatenate(all_scores)
    top_idx = np.argsort(all_scores)[::-1][:top_k]
    return all_scores[top_idx], top_idx, [all_smiles[i] for i in top_idx]
