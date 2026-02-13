"""
loss.py — Confidence-Weighted Semi-PU Contrastive Losses
=========================================================
Implements:
    1. CWSemiPUContrastiveLoss   : Full PU-aware contrastive loss with (α, β) weights
    2. StandardInfoNCELoss       : Vanilla InfoNCE for Phase 1 warm-up / ablation
    3. PriorRegularizationLoss   : Aligns learned π_a with BBE estimates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CWSemiPUContrastiveLoss(nn.Module):
    """
    Confidence-Weighted Semi-PU InfoNCE Loss.

    Combines:
        1. Du Plessis / Kiryo (NeurIPS 2017): nnPU risk estimator
        2. Chuang et al. (NeurIPS 2020): Debiased Contrastive Learning
        3. Sakai et al. (ICML 2017): Semi-PU with reliable negatives

    Each sample carries per-sample weights:
        alpha_i : positive contribution weight  (attraction to assay anchor)
        beta_i  : negative contribution weight  (repulsion from assay anchor)

    The loss for each assay anchor a_i:

        L_i = L_pos_i  +  λ_RN · L_neg_confirmed_i  +  λ_U · L_neg_debiased_i

    Where:
        L_pos  = -Σ_{j∈P} α_j · log[exp(sim(a_i, c_j)/τ) / Z]
        L_neg_confirmed = Σ_{j∈RN} β_j · exp(sim(a_i, c_j)/τ) / Z
        L_neg_debiased  = max(0, (E_U - π_i · E_P) / (1 - π_i))  [nnPU]

    Parameters
    ----------
    tau : float
        Temperature for contrastive scaling.
    lambda_rn : float
        Weight for confirmed negative loss term.
    lambda_u : float
        Weight for debiased unlabeled loss term.
    beta_floor : float
        Non-negative floor for nnPU (Kiryo et al.). Default=0.
    """

    def __init__(
        self,
        tau: float = 0.07,
        lambda_rn: float = 1.0,
        lambda_u: float = 1.0,
        beta_floor: float = 0.0,
    ):
        super().__init__()
        self.tau = tau
        self.lambda_rn = lambda_rn
        self.lambda_u = lambda_u
        self.beta_floor = beta_floor

    def forward(
        self,
        sim_matrix: torch.Tensor,
        pu_labels: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        pi_a: torch.Tensor,
        pu_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            sim_matrix: (B, B) cosine similarity between assay[i] and compound[j]
            pu_labels:  (B,)   PU label per sample: +1 (P), -1 (RN), 0 (U)
            alphas:     (B,)   positive contribution weight per sample
            betas:      (B,)   negative contribution weight per sample
            pi_a:       (B,)   estimated class prior per assay
            pu_weights: (B, B) PU Gate confidence weights

        Returns:
            loss: scalar
            metrics: dict of component losses for logging
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        # Scale similarities by temperature
        logits = sim_matrix / self.tau  # (B, B)

        # Masks
        pos_mask = (pu_labels == 1)   # (B,) which samples are Positive
        rn_mask = (pu_labels == -1)   # (B,) which samples are Reliable Negative
        u_mask = (pu_labels == 0)     # (B,) which samples are Unlabeled

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_pos_sum = 0.0
        loss_rn_sum = 0.0
        loss_u_sum = 0.0
        n_valid = 0

        for i in range(B):
            # --- For assay anchor i, compute loss over all compound j ≠ i ---

            # Identify diagonal (self-pair)
            mask_other = torch.ones(B, dtype=torch.bool, device=device)
            mask_other[i] = False

            logits_i = logits[i]  # (B,)

            # Log-sum-exp denominator (all compounds except self)
            log_Z = torch.logsumexp(logits_i[mask_other], dim=0)

            # === Positive Loss ===
            # Attract toward compounds with high α
            pos_idx = mask_other & pos_mask
            if pos_idx.any():
                pos_logits = logits_i[pos_idx]
                pos_alphas = alphas[pos_idx]
                # Weighted positive log-probability
                L_pos_i = -(pos_alphas * (pos_logits - log_Z)).mean()
            else:
                L_pos_i = torch.tensor(0.0, device=device)

            # === Confirmed Negative Loss (RN) ===
            # Repel from confirmed negatives, weighted by β and PU Gate
            rn_idx = mask_other & rn_mask
            if rn_idx.any():
                rn_logits = logits_i[rn_idx]
                rn_betas = betas[rn_idx]
                rn_gate_w = pu_weights[i][rn_idx]
                # Higher weight → stronger repulsion
                L_rn_i = (rn_betas * rn_gate_w * torch.exp(rn_logits - log_Z)).mean()
            else:
                L_rn_i = torch.tensor(0.0, device=device)

            # === Debiased Unlabeled Loss (nnPU) ===
            u_idx = mask_other & u_mask
            if u_idx.any() and pos_idx.any():
                pi_i = pi_a[i].clamp(min=1e-4, max=0.5)

                # E_U: weighted expected similarity over unlabeled
                u_logits = logits_i[u_idx]
                u_gate_w = pu_weights[i][u_idx]
                E_U = (u_gate_w * torch.exp(u_logits - log_Z)).mean()

                # E_P: expected similarity over positives (for debiasing)
                E_P = torch.exp(logits_i[pos_idx] - log_Z).mean()

                # Debiased negative estimation
                debiased = (E_U - pi_i * E_P) / (1.0 - pi_i + 1e-8)

                # Non-negative PU constraint (nnPU)
                L_u_i = torch.max(debiased, torch.tensor(self.beta_floor, device=device))
            else:
                L_u_i = torch.tensor(0.0, device=device)

            # --- Combine ---
            L_i = L_pos_i + self.lambda_rn * L_rn_i + self.lambda_u * L_u_i
            total_loss = total_loss + L_i

            loss_pos_sum += L_pos_i.item()
            loss_rn_sum += L_rn_i.item()
            loss_u_sum += L_u_i.item()
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        metrics = {
            "loss_total": total_loss.item(),
            "loss_pos": loss_pos_sum / max(n_valid, 1),
            "loss_rn": loss_rn_sum / max(n_valid, 1),
            "loss_u_debiased": loss_u_sum / max(n_valid, 1),
            "mean_pi_a": pi_a.mean().item(),
        }

        return total_loss, metrics


class StandardInfoNCELoss(nn.Module):
    """
    Standard InfoNCE loss (for Phase 1 warm-up and ablation comparison).
    Ignores PU structure entirely: all non-matched pairs are treated as negatives.
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        sim_matrix: torch.Tensor,
        **kwargs,  # accepts but ignores PU-specific args
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Standard symmetric InfoNCE: diagonal pairs are positives, off-diag are negatives.

        Args:
            sim_matrix: (B, B) cosine similarity
        Returns:
            loss, metrics dict
        """
        B = sim_matrix.size(0)
        logits = sim_matrix / self.tau
        labels = torch.arange(B, device=sim_matrix.device)

        # Symmetric loss: assay→compound + compound→assay
        loss_a2c = F.cross_entropy(logits, labels)
        loss_c2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2c + loss_c2a) / 2.0

        # Accuracy (for monitoring)
        with torch.no_grad():
            pred_a2c = logits.argmax(dim=1)
            acc = (pred_a2c == labels).float().mean().item()

        metrics = {
            "loss_total": loss.item(),
            "loss_a2c": loss_a2c.item(),
            "loss_c2a": loss_c2a.item(),
            "accuracy": acc,
        }

        return loss, metrics


class PriorRegularizationLoss(nn.Module):
    """
    Regularizes learned π_a to stay close to externally estimated priors (e.g., BBE).
    L_prior = MSE(π_a_learned, π_a_external)
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pi_a_learned: torch.Tensor,
        pi_a_external: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pi_a_learned: (B,) from AssayPriorNetwork
            pi_a_external: (B,) from BBE or empirical estimation
        Returns:
            weighted MSE loss
        """
        return self.weight * F.mse_loss(pi_a_learned, pi_a_external)


# ═══════════════════════════════════════════════════════════════
# Combined Loss with Curriculum Schedule
# ═══════════════════════════════════════════════════════════════

class CurriculumPULoss(nn.Module):
    """
    Wraps InfoNCE and CW-SemiPU losses with curriculum scheduling.

    Phase 1 (epoch < phase1_end): Pure InfoNCE warm-up
    Phase 2 (phase1_end ≤ epoch < phase2_end): Linear warm-up of PU components
    Phase 3 (epoch ≥ phase2_end): Full CW-SemiPU Loss

    Parameters
    ----------
    phase1_end : int  — end of InfoNCE warm-up phase
    phase2_end : int  — end of PU ramp-up phase
    tau : float       — temperature
    lambda_rn : float — confirmed negative weight (at full strength)
    lambda_u  : float — debiased unlabeled weight (at full strength)
    prior_reg_weight : float — prior regularization weight
    """

    def __init__(
        self,
        phase1_end: int = 5,
        phase2_end: int = 15,
        tau: float = 0.07,
        lambda_rn: float = 1.0,
        lambda_u: float = 1.0,
        prior_reg_weight: float = 0.1,
    ):
        super().__init__()
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end

        self.infonce = StandardInfoNCELoss(tau=tau)
        self.pu_loss = CWSemiPUContrastiveLoss(
            tau=tau, lambda_rn=lambda_rn, lambda_u=lambda_u,
        )
        self.prior_reg = PriorRegularizationLoss(weight=prior_reg_weight)

    def get_phase(self, epoch: int) -> int:
        if epoch < self.phase1_end:
            return 1
        elif epoch < self.phase2_end:
            return 2
        else:
            return 3

    def get_pu_weight(self, epoch: int) -> float:
        """Linear warm-up of PU loss weight during Phase 2."""
        if epoch < self.phase1_end:
            return 0.0
        elif epoch < self.phase2_end:
            progress = (epoch - self.phase1_end) / max(self.phase2_end - self.phase1_end, 1)
            return progress
        else:
            return 1.0

    def forward(
        self,
        sim_matrix: torch.Tensor,
        pu_labels: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        pi_a: torch.Tensor,
        pu_weights: torch.Tensor,
        epoch: int,
        pi_a_external: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            epoch: current training epoch (determines curriculum phase)
            pi_a_external: (B,) optional BBE-estimated priors for regularization
        """
        phase = self.get_phase(epoch)
        pu_w = self.get_pu_weight(epoch)

        # Phase 1: Pure InfoNCE
        loss_infonce, metrics_infonce = self.infonce(sim_matrix)

        if phase == 1:
            metrics_infonce["phase"] = 1
            metrics_infonce["pu_weight"] = 0.0
            return loss_infonce, metrics_infonce

        # Phase 2-3: Add PU loss
        loss_pu, metrics_pu = self.pu_loss(
            sim_matrix, pu_labels, alphas, betas, pi_a, pu_weights,
        )

        # Blended loss
        total_loss = (1.0 - pu_w) * loss_infonce + pu_w * loss_pu

        # Prior regularization (Phase 3 only)
        loss_prior_reg = torch.tensor(0.0, device=sim_matrix.device)
        if phase == 3 and pi_a_external is not None:
            loss_prior_reg = self.prior_reg(pi_a, pi_a_external)
            total_loss = total_loss + loss_prior_reg

        metrics = {
            "phase": phase,
            "pu_weight": pu_w,
            "loss_total": total_loss.item(),
            "loss_infonce": loss_infonce.item(),
            "loss_pu": loss_pu.item(),
            "loss_prior_reg": loss_prior_reg.item(),
            **{f"pu_{k}": v for k, v in metrics_pu.items()},
        }

        return total_loss, metrics
