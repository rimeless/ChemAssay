"""
Confidence-weighted Semi-PU Assay–Compound Model (with InfoNCE)
===============================================================

- Positive / Negative / Inconclusive / Unspecified 라벨 구조
- Negative: confidence-weighted supervised BCE
- Inconclusive: true unlabeled → PU loss
- Assay-specific prior (π) 추정
- Activity value 기반 soft label (IC50/EC50 등)
- Assay-aware InfoNCE contrastive loss (compound–assay)
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

# Optional (validation/vis); comment out if not installed
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# =============================================================================
# Label & Meta 정의
# =============================================================================

class OriginalLabel(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1
    INCONCLUSIVE = 2
    UNSPECIFIED = 3


class AssayType(IntEnum):
    SCREENING = 0
    CONFIRMATORY = 1
    UNKNOWN = 2


@dataclass
class LabelWithConfidence:
    # Soft labels: P(positive), P(negative), P(unknown)
    p_positive: float
    p_negative: float
    p_unknown: float  # Inconclusive에 해당

    include_in_training: bool = True   # 학습에 포함할지
    apply_pu: bool = False            # PU learning 적용할지 (true unlabeled만)


# =============================================================================
# Model Config
# =============================================================================

@dataclass
class ModelConfig:
    # Encoder output dims
    compound_dim: int = 768      # ChemBERTa CLS dim
    assay_text_dim: int = 768    # SciBERT CLS dim
    target_dim: int = 1280       # ESM-2 pooled dim

    # Latent
    latent_dim: int = 512
    num_prototypes: int = 256

    # Prototype / prior 기타
    prototype_momentum: float = 0.9

    # Contrastive (InfoNCE)
    temperature: float = 0.07
    contrastive_weight: float = 0.2   # 전체 loss 중 대조학습 비중


# =============================================================================
# Label Confidence Assigner
# =============================================================================

class LabelConfidenceAssigner:
    """
    원본 라벨 + 메타데이터를 기반으로 confidence 할당
    """

    def __init__(
        self,
        # Confirmatory assay confidence
        confirmatory_neg_with_value: float = 0.95,
        confirmatory_neg_without_value: float = 0.85,
        # Screening assay confidence
        screening_neg_with_value: float = 0.70,
        screening_neg_without_value: float = 0.45,
        # Activity value thresholds for soft labeling
        strong_negative_threshold: float = 100.0,  # IC50 > 100μM
        weak_negative_threshold: float = 10.0,     # IC50 > 10μM
    ):
        self.conf_neg_with_val = confirmatory_neg_with_value
        self.conf_neg_without_val = confirmatory_neg_without_value
        self.screen_neg_with_val = screening_neg_with_value
        self.screen_neg_without_val = screening_neg_without_value
        self.strong_neg_thresh = strong_negative_threshold
        self.weak_neg_thresh = weak_negative_threshold

    def assign_confidence(
        self,
        original_label: int,
        assay_type: int,
        has_activity_value: bool,
        activity_value: Optional[float] = None,
        multi_dose_tested: bool = False
    ) -> LabelWithConfidence:
        """
        샘플별 confidence 할당
        """

        # Unspecified → Exclude
        if original_label == OriginalLabel.UNSPECIFIED:
            return LabelWithConfidence(
                p_positive=0.0,
                p_negative=0.0,
                p_unknown=1.0,
                include_in_training=False,
                apply_pu=False
            )

        # Positive → High confidence positive
        if original_label == OriginalLabel.POSITIVE:
            return LabelWithConfidence(
                p_positive=1.0,
                p_negative=0.0,
                p_unknown=0.0,
                include_in_training=True,
                apply_pu=False
            )

        # Inconclusive → True unlabeled (PU learning)
        if original_label == OriginalLabel.INCONCLUSIVE:
            return LabelWithConfidence(
                p_positive=0.0,
                p_negative=0.0,
                p_unknown=1.0,
                include_in_training=True,
                apply_pu=True
            )

        # Negative → confidence depends on assay type & activity value
        if original_label == OriginalLabel.NEGATIVE:
            return self._assign_negative_confidence(
                assay_type, has_activity_value, activity_value, multi_dose_tested
            )

        raise ValueError(f"Unknown label: {original_label}")

    def _assign_negative_confidence(
        self,
        assay_type: int,
        has_activity_value: bool,
        activity_value: Optional[float],
        multi_dose_tested: bool
    ) -> LabelWithConfidence:
        """
        Negative 샘플의 confidence 계산
        """
        is_confirmatory = (assay_type == AssayType.CONFIRMATORY)

        # Base confidence by assay type and value availability
        if is_confirmatory:
            base_conf = self.conf_neg_with_val if has_activity_value else self.conf_neg_without_val
        else:  # Screening or Unknown
            base_conf = self.screen_neg_with_val if has_activity_value else self.screen_neg_without_val

        # Activity value로 조정 (있을 때만)
        if has_activity_value and activity_value is not None:
            if activity_value > self.strong_neg_thresh:
                value_boost = 0.10
            elif activity_value > self.weak_neg_thresh:
                value_boost = 0.05
            else:
                value_boost = -0.10

            base_conf = min(0.98, max(0.2, base_conf + value_boost))

        # Multi-dose면 조금 더 신뢰
        if multi_dose_tested:
            base_conf = min(0.98, base_conf + 0.05)

        # Soft labels
        p_negative = base_conf
        p_positive = (1 - base_conf) * 0.3
        p_unknown = (1 - base_conf) * 0.7

        return LabelWithConfidence(
            p_positive=p_positive,
            p_negative=p_negative,
            p_unknown=p_unknown,
            include_in_training=True,
            apply_pu=False
        )


# =============================================================================
# Confidence-weighted BCE Loss
# =============================================================================

class ConfidenceWeightedBCELoss(nn.Module):
    """
    Confidence에 따라 가중치를 부여하는 BCE Loss
    """

    def __init__(self, min_weight: float = 0.1):
        super().__init__()
        self.min_weight = min_weight

    def forward(
        self,
        logits: torch.Tensor,          # (B,)
        soft_labels: torch.Tensor,     # (B, 2) [p_neg, p_pos]
        confidences: torch.Tensor      # (B,)
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        p_pos = soft_labels[:, 1]
        p_neg = soft_labels[:, 0]

        eps = 1e-7
        loss = -(p_pos * torch.log(probs + eps) + p_neg * torch.log(1 - probs + eps))

        weights = torch.clamp(confidences, min=self.min_weight)
        weighted_loss = loss * weights

        return weighted_loss.mean()


# =============================================================================
# Semi-supervised PU Loss
# =============================================================================

class SemiPULoss(nn.Module):
    """
    Semi-supervised PU Loss
    - Supervised: positive + confident negative
    - PU: inconclusive (true unlabeled)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.supervised_loss = ConfidenceWeightedBCELoss(min_weight=0.1)
        self.pu_beta = 0.0
        self.pu_gamma = 1.0

    def forward(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,      # (B, 2)
        confidences: torch.Tensor,      # (B,)
        apply_pu_mask: torch.Tensor,    # (B,) bool
        assay_priors: torch.Tensor      # (B,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}

        supervised_mask = ~apply_pu_mask
        pu_mask = apply_pu_mask

        total_loss = torch.tensor(0.0, device=logits.device)

        # 1) Supervised
        if supervised_mask.sum() > 0:
            sup_loss = self.supervised_loss(
                logits[supervised_mask],
                soft_labels[supervised_mask],
                confidences[supervised_mask]
            )
            total_loss = total_loss + sup_loss
            metrics['supervised_loss'] = sup_loss.item()

        # 2) PU
        if pu_mask.sum() > 0:
            pu_loss = self._compute_pu_loss(
                logits, soft_labels, pu_mask, supervised_mask, assay_priors
            )
            total_loss = total_loss + pu_loss
            metrics['pu_loss'] = pu_loss.item()

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

    def _compute_pu_loss(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
        pu_mask: torch.Tensor,
        supervised_mask: torch.Tensor,
        assay_priors: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        positive_mask = supervised_mask & (soft_labels[:, 1] > 0.5)

        if positive_mask.sum() > 0:
            pos_loss = F.binary_cross_entropy(
                probs[positive_mask],
                torch.ones_like(probs[positive_mask]),
                reduction='mean'
            )
        else:
            pos_loss = torch.tensor(0.0, device=logits.device)

        if pu_mask.sum() > 0:
            unlabeled_neg_loss = F.binary_cross_entropy(
                probs[pu_mask],
                torch.zeros_like(probs[pu_mask]),
                reduction='mean'
            )
        else:
            unlabeled_neg_loss = torch.tensor(0.0, device=logits.device)

        if positive_mask.sum() > 0:
            pos_neg_loss = F.binary_cross_entropy(
                probs[positive_mask],
                torch.zeros_like(probs[positive_mask]),
                reduction='mean'
            )
        else:
            pos_neg_loss = torch.tensor(0.0, device=logits.device)

        mean_prior = assay_priors[pu_mask].mean() if pu_mask.sum() > 0 else torch.tensor(0.1, device=logits.device)

        neg_risk = unlabeled_neg_loss - mean_prior * pos_neg_loss

        if neg_risk < self.pu_beta:
            pu_loss = mean_prior * pos_loss - self.pu_gamma * neg_risk
        else:
            pu_loss = mean_prior * pos_loss + neg_risk

        return pu_loss


# =============================================================================
# Activity Value → Soft Label Encoder
# =============================================================================

class ActivityValueEncoder(nn.Module):
    """
    Activity value(IC50, EC50 등)를 soft label로 변환
    """

    def __init__(
        self,
        positive_threshold: float = 1.0,    # IC50 < 1μM → positive
        negative_threshold: float = 10.0,   # IC50 > 10μM → negative
        log_transform: bool = True
    ):
        super().__init__()
        self.pos_thresh = positive_threshold
        self.neg_thresh = negative_threshold
        self.log_transform = log_transform

    def forward(
        self,
        activity_values: torch.Tensor,  # (B,) in μM
        has_value: torch.Tensor         # (B,) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = activity_values.size(0)
        device = activity_values.device

        soft_labels = torch.full((B, 2), 0.5, device=device)
        confidence = torch.zeros(B, device=device)

        if has_value.sum() == 0:
            return soft_labels, confidence

        values = activity_values[has_value]

        if self.log_transform:
            values = torch.log10(values.clamp(min=1e-3))
            pos_thresh = np.log10(self.pos_thresh)
            neg_thresh = np.log10(self.neg_thresh)
        else:
            pos_thresh = self.pos_thresh
            neg_thresh = self.neg_thresh

        normalized = (values - pos_thresh) / (neg_thresh - pos_thresh + 1e-6)
        normalized = normalized.clamp(0, 1)

        p_positive = 1 - normalized
        p_negative = normalized

        conf = 2 * torch.abs(normalized - 0.5)

        soft_labels[has_value, 0] = p_negative
        soft_labels[has_value, 1] = p_positive
        confidence[has_value] = conf

        return soft_labels, confidence


# =============================================================================
# Encoders (ChemBERTa / SciBERT / ESM-2)
# =============================================================================

class CompoundEncoder(nn.Module):
    """
    Compound structure encoder using pretrained ChemBERTa
    """

    def __init__(self, config: ModelConfig, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projector = nn.Sequential(
            nn.Linear(config.compound_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        compound_repr = outputs.last_hidden_state[:, 0, :]
        return self.projector(compound_repr)


class AssayTextEncoder(nn.Module):
    """
    Assay description encoder using SciBERT
    """

    def __init__(self, config: ModelConfig, model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projector = nn.Sequential(
            nn.Linear(config.assay_text_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        assay_repr = outputs.last_hidden_state[:, 0, :]
        return self.projector(assay_repr)


class TargetEncoder(nn.Module):
    """
    Protein target encoder using ESM-2
    - has_target=False → learnable no_target embedding 사용
    """

    def __init__(self, config: ModelConfig, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projector = nn.Sequential(
            nn.Linear(config.target_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        self.no_target_embedding = nn.Parameter(torch.randn(config.latent_dim))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        has_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if has_target is None:
            batch_size = input_ids.size(0)
            return self.no_target_embedding.unsqueeze(0).expand(batch_size, -1)

        B = has_target.size(0)
        device = has_target.device
        out = torch.zeros(B, self.no_target_embedding.size(0), device=device)

        if has_target.any():
            target_outputs = self.encoder(
                input_ids=input_ids[has_target],
                attention_mask=attention_mask[has_target]
            )
            target_repr = target_outputs.last_hidden_state.mean(dim=1)
            out[has_target] = self.projector(target_repr)

        if (~has_target).any():
            out[~has_target] = self.no_target_embedding.unsqueeze(0).expand((~has_target).sum(), -1)

        return out


# =============================================================================
# Assay Context Aggregator + Prototype Layer
# =============================================================================

class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(dim * 2, dim)

    def forward(self, text_repr: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([text_repr, target_repr], dim=-1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * text_repr + (1 - gate) * transformed


class AssayContextAggregator(nn.Module):
    """
    Assay text + target representation → unified assay_repr
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.gated_fusion = GatedFusion(config.latent_dim)
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        assay_text_repr: torch.Tensor,  # (B, D)
        target_repr: torch.Tensor,      # (B, D)
        has_target: torch.Tensor        # (B,)
    ) -> torch.Tensor:
        attn_output, _ = self.cross_attention(
            query=assay_text_repr.unsqueeze(1),
            key=target_repr.unsqueeze(1),
            value=target_repr.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)

        fused = self.gated_fusion(assay_text_repr, attn_output)

        mask = has_target.float().unsqueeze(-1)
        output = mask * fused + (1 - mask) * assay_text_repr

        return self.norm(output)


class AssayPrototypeLayer(nn.Module):
    """
    Learnable assay prototypes (long-tail용)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_prototypes = config.num_prototypes
        self.latent_dim = config.latent_dim

        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.latent_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.assignment_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.num_prototypes)
        )

    def forward(self, assay_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.assignment_net(assay_repr)
        assignments = F.softmax(logits / 0.1, dim=-1)
        prototype_repr = torch.matmul(assignments, self.prototypes)
        enhanced_repr = assay_repr + 0.5 * prototype_repr
        return enhanced_repr, assignments


# =============================================================================
# Assay Prior Estimator
# =============================================================================

class AssayPriorEstimator(nn.Module):
    """
    Assay별 class prior (π) 추정기
    - observed positive ratio
    - assay_repr 기반 prior predictor
    - learnable per-assay prior
    """

    def __init__(self, config: ModelConfig, num_assays: int):
        super().__init__()
        self.config = config
        self.num_assays = num_assays

        # observed stats
        self.register_buffer('observed_positive_ratio', torch.zeros(num_assays))
        self.register_buffer('assay_sample_counts', torch.zeros(num_assays))

        # assay_repr → prior
        self.prior_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # learnable per-assay prior
        self.learnable_logit_prior = nn.Parameter(
            torch.zeros(num_assays) - 2.0  # sigmoid(-2) ≈ 0.12
        )

        self.min_prior = 0.001
        self.max_prior = 0.5

        self.strategy_weights = nn.Parameter(torch.tensor([0.3, 0.4, 0.3]))

    def update_observed_ratios(self, assay_ids: torch.Tensor, pos_flags: torch.Tensor):
        """
        pos_flags: 1 = positive, 0 = not-positive (neg+inconc)
        """
        for aid, label in zip(assay_ids.tolist(), pos_flags.tolist()):
            if aid < self.num_assays:
                count = self.assay_sample_counts[aid].item()
                current_ratio = self.observed_positive_ratio[aid].item()
                new_count = count + 1
                new_ratio = (current_ratio * count + float(label)) / new_count
                self.observed_positive_ratio[aid] = new_ratio
                self.assay_sample_counts[aid] = new_count

    def forward(
        self,
        assay_ids: torch.Tensor,
        assay_repr: torch.Tensor,
        use_observed: bool = True
    ) -> torch.Tensor:
        observed_priors = self.observed_positive_ratio[assay_ids]
        predicted_priors = self.prior_predictor(assay_repr).squeeze(-1)
        learnable_priors = torch.sigmoid(self.learnable_logit_prior[assay_ids])

        weights = F.softmax(self.strategy_weights, dim=0)
        sample_counts = self.assay_sample_counts[assay_ids]
        confidence = torch.clamp(sample_counts / 1000.0, 0, 1)

        combined_prior = (
            weights[0] * observed_priors * confidence +
            weights[1] * predicted_priors +
            weights[2] * learnable_priors * confidence +
            (1 - confidence) * weights[0] * predicted_priors
        )

        combined_prior = torch.clamp(combined_prior, self.min_prior, self.max_prior)
        return combined_prior


# =============================================================================
# Assay-aware InfoNCE Loss
# =============================================================================

class AssayAwareContrastiveLoss(nn.Module):
    """
    Compound–Assay InfoNCE-style contrastive loss

    - 임베딩: compound_repr (B, D), assay_repr (B, D)
    - positive pair:
        같은 assay_id AND 같은 original_label
    - 나머지는 negative
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        compound_repr: torch.Tensor,   # (B, D)
        assay_repr: torch.Tensor,      # (B, D)
        labels: torch.Tensor,          # (B,) OriginalLabel 값 (0~3)
        assay_ids: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        device = compound_repr.device
        B = compound_repr.size(0)

        # L2 normalize
        c = F.normalize(compound_repr, dim=-1)
        a = F.normalize(assay_repr, dim=-1)

        # similarity matrix (compound_i vs assay_j)
        sim = torch.matmul(c, a.T) / self.temperature  # (B, B)

        # exp(sim)
        exp_sim = torch.exp(sim)

        # positive mask: same assay_id & same label
        assay_match = assay_ids.unsqueeze(0) == assay_ids.unsqueeze(1)   # (B, B)
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = assay_match & label_match

        # self-pair는 positive에서 제거 (i==j)
        positive_mask.fill_diagonal_(False)

        # 각 row마다 positive logit 합
        pos_sim = (exp_sim * positive_mask.float()).sum(dim=1)  # (B,)

        # denominator: 모든 j에 대한 exp(sim_ij)
        all_sim = exp_sim.sum(dim=1)  # (B,)

        eps = 1e-8
        # positive가 하나도 없는 row는 무시
        has_pos = positive_mask.sum(dim=1) > 0  # (B,)

        # InfoNCE: -log( sum_pos / sum_all )
        loss = -torch.log((pos_sim + eps) / (all_sim + eps))

        if has_pos.sum() == 0:
            # batch 내에 positive pair가 전혀 없으면 0 반환
            return torch.tensor(0.0, device=device)

        return loss[has_pos].mean()


# =============================================================================
# Main Model with Confidence-weighted Semi-PU + InfoNCE
# =============================================================================

class AssayCompoundModelWithConfidencePU(nn.Module):
    """
    Confidence-weighted Semi-PU Learning 통합 모델 (InfoNCE 포함)
    """

    def __init__(self, config: ModelConfig, num_assays: int):
        super().__init__()
        self.config = config

        # Encoders & aggregator
        self.compound_encoder = CompoundEncoder(config)
        self.assay_text_encoder = AssayTextEncoder(config)
        self.target_encoder = TargetEncoder(config)
        self.assay_aggregator = AssayContextAggregator(config)
        self.prototype_layer = AssayPrototypeLayer(config)

        # Interaction
        self.interaction = nn.Sequential(
            nn.Linear(config.latent_dim * 3, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.GELU()
        )

        # Binary classifier (positive logit)
        self.classifier = nn.Linear(config.latent_dim // 2, 1)

        # Contrastive Projection Head (for InfoNCE)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, 128)   # 128-d projection
        )

        # Confidence & Prior
        self.label_assigner = LabelConfidenceAssigner()
        self.activity_encoder = ActivityValueEncoder()
        self.prior_estimator = AssayPriorEstimator(config, num_assays)

        # Loss
        self.loss_fn = SemiPULoss(config)

    def forward(
        self,
        compound_input_ids: torch.Tensor,
        compound_attention_mask: torch.Tensor,
        assay_input_ids: torch.Tensor,
        assay_attention_mask: torch.Tensor,
        assay_ids: torch.Tensor,
        original_labels: Optional[torch.Tensor] = None,
        assay_types: Optional[torch.Tensor] = None,
        activity_values: Optional[torch.Tensor] = None,
        has_activity_value: Optional[torch.Tensor] = None,
        multi_dose_tested: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        has_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B = compound_input_ids.size(0)
        device = compound_input_ids.device

        # Encoding
        compound_repr = self.compound_encoder(compound_input_ids, compound_attention_mask)
        assay_text_repr = self.assay_text_encoder(assay_input_ids, assay_attention_mask)

        if has_target is None:
            has_target = torch.zeros(B, dtype=torch.bool, device=device)
        target_repr = self.target_encoder(target_input_ids, target_attention_mask, has_target)

        assay_repr = self.assay_aggregator(assay_text_repr, target_repr, has_target)
        assay_repr, proto_assignments = self.prototype_layer(assay_repr)

        # Interaction & logits
        interaction_features = torch.cat([
            compound_repr,
            assay_repr,
            compound_repr * assay_repr
        ], dim=-1)
        interaction_repr = self.interaction(interaction_features)
        logits = self.classifier(interaction_repr).squeeze(-1)

        # Contrastive projection (InfoNCE 용)
        compound_proj = self.contrastive_proj(compound_repr)   # (B, 128)
        assay_proj = self.contrastive_proj(assay_repr)         # (B, 128)

        outputs: Dict[str, torch.Tensor] = {
            'logits': logits,
            'probs': torch.sigmoid(logits),
            'compound_repr': compound_repr,
            'assay_repr': assay_repr,
            'compound_proj': compound_proj,
            'assay_proj': assay_proj,
            'proto_assignments': proto_assignments
        }

        if original_labels is not None:
            loss_outputs = self._compute_loss(
                logits=logits,
                original_labels=original_labels,
                assay_ids=assay_ids,
                assay_repr=assay_repr,
                assay_types=assay_types,
                activity_values=activity_values,
                has_activity_value=has_activity_value,
                multi_dose_tested=multi_dose_tested,
                device=device
            )
            outputs.update(loss_outputs)

        return outputs

    def _compute_loss(
        self,
        logits: torch.Tensor,
        original_labels: torch.Tensor,
        assay_ids: torch.Tensor,
        assay_repr: torch.Tensor,
        assay_types: Optional[torch.Tensor],
        activity_values: Optional[torch.Tensor],
        has_activity_value: Optional[torch.Tensor],
        multi_dose_tested: Optional[torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:

        B = logits.size(0)

        if assay_types is None:
            assay_types = torch.full((B,), AssayType.UNKNOWN, device=device)
        if has_activity_value is None:
            has_activity_value = torch.zeros(B, dtype=torch.bool, device=device)
        if activity_values is None:
            activity_values = torch.zeros(B, device=device)
        if multi_dose_tested is None:
            multi_dose_tested = torch.zeros(B, dtype=torch.bool, device=device)

        # Activity value가 있는 샘플의 soft label
        activity_soft_labels, activity_confidence = self.activity_encoder(
            activity_values, has_activity_value
        )

        soft_labels = torch.zeros(B, 2, device=device)
        confidences = torch.zeros(B, device=device)
        apply_pu_mask = torch.zeros(B, dtype=torch.bool, device=device)
        include_mask = torch.ones(B, dtype=torch.bool, device=device)

        # prior estimator의 observed ratio 업데이트 (positive vs not-positive)
        pos_flags = (original_labels == OriginalLabel.POSITIVE).long()
        self.prior_estimator.update_observed_ratios(assay_ids, pos_flags)

        for i in range(B):
            label_conf = self.label_assigner.assign_confidence(
                original_label=int(original_labels[i].item()),
                assay_type=int(assay_types[i].item()),
                has_activity_value=bool(has_activity_value[i].item()),
                activity_value=float(activity_values[i].item()) if has_activity_value[i] else None,
                multi_dose_tested=bool(multi_dose_tested[i].item())
            )

            if not label_conf.include_in_training:
                include_mask[i] = False
                continue

            if has_activity_value[i] and original_labels[i] == OriginalLabel.NEGATIVE:
                soft_labels[i] = activity_soft_labels[i]
                confidences[i] = activity_confidence[i]
            else:
                soft_labels[i, 0] = label_conf.p_negative
                soft_labels[i, 1] = label_conf.p_positive
                confidences[i] = 1.0 - label_conf.p_unknown

            apply_pu_mask[i] = label_conf.apply_pu

        assay_priors = self.prior_estimator(assay_ids, assay_repr)

        if include_mask.sum() == 0:
            return {
                'loss': torch.tensor(0.0, device=device),
                'metrics': {'skipped': True}
            }

        loss, metrics = self.loss_fn(
            logits=logits[include_mask],
            soft_labels=soft_labels[include_mask],
            confidences=confidences[include_mask],
            apply_pu_mask=apply_pu_mask[include_mask],
            assay_priors=assay_priors[include_mask]
        )

        metrics['n_supervised'] = int((~apply_pu_mask[include_mask]).sum().item())
        metrics['n_pu'] = int(apply_pu_mask[include_mask].sum().item())
        metrics['n_excluded'] = int((~include_mask).sum().item())
        metrics['avg_confidence'] = float(confidences[include_mask].mean().item())

        return {'loss': loss, 'metrics': metrics}


# =============================================================================
# Dataset
# =============================================================================

class AssayCompoundDataset(torch.utils.data.Dataset):
    """
    메타데이터를 포함한 Dataset
    - 길이 N인 리스트들로 초기화
    """

    def __init__(
        self,
        compound_smiles: List[str],
        assay_descriptions: List[str],
        assay_ids: List[int],
        labels: List[int],
        assay_types: List[int],
        activity_values: Optional[List[Optional[float]]] = None,
        target_sequences: Optional[List[Optional[str]]] = None,
        multi_dose_info: Optional[List[bool]] = None,
        compound_tokenizer=None,
        assay_tokenizer=None,
        target_tokenizer=None,
        max_compound_len: int = 128,
        max_assay_len: int = 512,
        max_target_len: int = 1024,
    ):
        assert len(compound_smiles) == len(assay_descriptions) == len(assay_ids) == len(labels)

        self.compound_smiles = compound_smiles
        self.assay_descriptions = assay_descriptions
        self.assay_ids = assay_ids
        self.labels = labels
        self.assay_types = assay_types

        N = len(labels)
        if activity_values is None:
            activity_values = [None] * N
        if target_sequences is None:
            target_sequences = [None] * N
        if multi_dose_info is None:
            multi_dose_info = [False] * N

        self.activity_values = activity_values
        self.target_sequences = target_sequences
        self.multi_dose_info = multi_dose_info

        self.compound_tokenizer = compound_tokenizer
        self.assay_tokenizer = assay_tokenizer
        self.target_tokenizer = target_tokenizer

        self.max_compound_len = max_compound_len
        self.max_assay_len = max_assay_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item: Dict[str, torch.Tensor] = {}

        # 원본 메타
        item['original_label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['assay_id'] = torch.tensor(self.assay_ids[idx], dtype=torch.long)
        item['assay_type'] = torch.tensor(self.assay_types[idx], dtype=torch.long)

        # activity
        val = self.activity_values[idx]
        has_val = (val is not None)
        item['has_activity_value'] = torch.tensor(has_val, dtype=torch.bool)
        item['activity_value'] = torch.tensor(float(val) if has_val else 0.0, dtype=torch.float)

        # multi-dose
        item['multi_dose_tested'] = torch.tensor(bool(self.multi_dose_info[idx]), dtype=torch.bool)

        # target
        tgt_seq = self.target_sequences[idx]
        has_target = tgt_seq is not None
        item['has_target'] = torch.tensor(has_target, dtype=torch.bool)

        # Tokenize compound
        c_tok = self.compound_tokenizer(
            self.compound_smiles[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_compound_len,
            return_tensors='pt'
        )
        item['compound_input_ids'] = c_tok['input_ids'].squeeze(0)
        item['compound_attention_mask'] = c_tok['attention_mask'].squeeze(0)

        # Tokenize assay
        a_tok = self.assay_tokenizer(
            self.assay_descriptions[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_assay_len,
            return_tensors='pt'
        )
        item['assay_input_ids'] = a_tok['input_ids'].squeeze(0)
        item['assay_attention_mask'] = a_tok['attention_mask'].squeeze(0)

        # Tokenize target if present
        if has_target:
            t_tok = self.target_tokenizer(
                tgt_seq,
                padding='max_length',
                truncation=True,
                max_length=self.max_target_len,
                return_tensors='pt'
            )
            item['target_input_ids'] = t_tok['input_ids'].squeeze(0)
            item['target_attention_mask'] = t_tok['attention_mask'].squeeze(0)

        return item


# =============================================================================
# Trainer
# =============================================================================

class ConfidencePUTrainer:
    """
    Confidence-weighted Semi-PU + InfoNCE contrastive Training
    """

    def __init__(self, model: AssayCompoundModelWithConfidencePU, config: ModelConfig):
        self.model = model
        self.config = config

        # Contrastive loss
        self.contrastive_loss_fn = AssayAwareContrastiveLoss(temperature=config.temperature)
        self.contrastive_weight = config.contrastive_weight

        self.optimizer = torch.optim.AdamW([
            {'params': model.compound_encoder.parameters(), 'lr': 1e-5},
            {'params': model.assay_text_encoder.parameters(), 'lr': 1e-5},
            {'params': model.target_encoder.parameters(), 'lr': 1e-5},
            {'params': model.assay_aggregator.parameters(), 'lr': 5e-5},
            {'params': model.prototype_layer.parameters(), 'lr': 5e-5},
            {'params': model.interaction.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-4},
            {'params': model.prior_estimator.parameters(), 'lr': 1e-5},
            {'params': model.activity_encoder.parameters(), 'lr': 1e-4},
            {'params': model.contrastive_proj.parameters(), 'lr': 1e-4},
        ], weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )

    def train_epoch(self, dataloader: DataLoader, device) -> Dict[str, float]:
        self.model.train()
        total_metrics: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            batch = self._move_to_device(batch, device)

            outputs = self.model(
                compound_input_ids=batch['compound_input_ids'],
                compound_attention_mask=batch['compound_attention_mask'],
                assay_input_ids=batch['assay_input_ids'],
                assay_attention_mask=batch['assay_attention_mask'],
                assay_ids=batch['assay_id'],
                original_labels=batch['original_label'],
                assay_types=batch['assay_type'],
                activity_values=batch['activity_value'],
                has_activity_value=batch['has_activity_value'],
                multi_dose_tested=batch['multi_dose_tested'],
                target_input_ids=batch.get('target_input_ids'),
                target_attention_mask=batch.get('target_attention_mask'),
                has_target=batch['has_target'],
            )

            # 1) Semi-PU loss
            semi_pu_loss = outputs['loss']

            # 2) Contrastive InfoNCE loss (assay-aware)
            labels = batch['original_label']
            assay_ids = batch['assay_id']
            valid_mask = labels != OriginalLabel.UNSPECIFIED

            if valid_mask.sum() > 1:  # 최소 2개 이상 있어야 contrastive 의미 있음
                contr_loss = self.contrastive_loss_fn(
                    compound_repr=outputs['compound_proj'][valid_mask],
                    assay_repr=outputs['assay_proj'][valid_mask],
                    labels=labels[valid_mask],
                    assay_ids=assay_ids[valid_mask],
                )
            else:
                contr_loss = torch.tensor(0.0, device=device)

            total_loss = semi_pu_loss + self.contrastive_weight * contr_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # metrics 정리
            metrics = outputs.get('metrics', {}).copy()
            metrics['semi_pu_loss'] = float(semi_pu_loss.item())
            metrics['contrastive_loss'] = float(contr_loss.item())
            metrics['total_loss'] = float(total_loss.item())

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

        self.scheduler.step()

        return {k: v / num_batches for k, v in total_metrics.items()} if num_batches > 0 else {}

    def _move_to_device(self, batch, device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


# =============================================================================
# Evaluator (Validation)
# =============================================================================

class ConfidencePUEvaluator:
    """
    Validation / Test용 evaluator
    - Binary: POSITIVE vs (NEGATIVE + INCONCLUSIVE)
    - UNSPECIFIED는 평가에서 제외
    """

    def __init__(self):
        pass

    @torch.no_grad()
    def evaluate(
        self,
        model: AssayCompoundModelWithConfidencePU,
        dataloader: DataLoader,
        device: torch.device,
        per_assay: bool = False,
    ) -> Dict[str, float]:
        model.eval()

        all_scores = []
        all_binary_labels = []
        all_original_labels = []
        all_assay_types = []
        all_assay_ids = []

        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(
                compound_input_ids=batch["compound_input_ids"],
                compound_attention_mask=batch["compound_attention_mask"],
                assay_input_ids=batch["assay_input_ids"],
                assay_attention_mask=batch["assay_attention_mask"],
                assay_ids=batch["assay_id"],
                original_labels=batch["original_label"],
                assay_types=batch["assay_type"],
                activity_values=batch["activity_value"],
                has_activity_value=batch["has_activity_value"],
                multi_dose_tested=batch["multi_dose_tested"],
                target_input_ids=batch.get("target_input_ids"),
                target_attention_mask=batch.get("target_attention_mask"),
                has_target=batch["has_target"],
            )

            probs = outputs["probs"].detach().cpu().numpy()  # P(positive)
            labels = batch["original_label"].detach().cpu().numpy()
            assay_types = batch["assay_type"].detach().cpu().numpy()
            assay_ids = batch["assay_id"].detach().cpu().numpy()

            # UNSPECIFIED 제외
            mask_valid = labels != OriginalLabel.UNSPECIFIED
            if mask_valid.sum() == 0:
                continue

            probs = probs[mask_valid]
            labels = labels[mask_valid]
            assay_types = assay_types[mask_valid]
            assay_ids = assay_ids[mask_valid]

            # Binary label: POSITIVE vs (NEGATIVE + INCONCLUSIVE)
            bin_labels = (labels == OriginalLabel.POSITIVE).astype(np.int64)

            all_scores.append(probs)
            all_binary_labels.append(bin_labels)
            all_original_labels.append(labels)
            all_assay_types.append(assay_types)
            all_assay_ids.append(assay_ids)

        if len(all_scores) == 0:
            return {"note": "no valid samples for evaluation"}

        all_scores = np.concatenate(all_scores, axis=0)
        all_binary_labels = np.concatenate(all_binary_labels, axis=0)
        all_original_labels = np.concatenate(all_original_labels, axis=0)
        all_assay_types = np.concatenate(all_assay_types, axis=0)
        all_assay_ids = np.concatenate(all_assay_ids, axis=0)

        metrics: Dict[str, float] = {}

        # 전체 binary metrics
        try:
            metrics["roc_auc"] = roc_auc_score(all_binary_labels, all_scores)
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["average_precision"] = average_precision_score(all_binary_labels, all_scores)
        except ValueError:
            metrics["average_precision"] = float("nan")

        # threshold 0.5
        preds = (all_scores >= 0.5).astype(np.int64)
        metrics["accuracy"] = accuracy_score(all_binary_labels, preds)
        metrics["balanced_accuracy"] = balanced_accuracy_score(all_binary_labels, preds)
        metrics["macro_f1"] = f1_score(all_binary_labels, preds, average="macro")
        metrics["pos_ratio_pred"] = float(preds.mean())
        metrics["pos_ratio_true"] = float(all_binary_labels.mean())

        # Confirmatory / Screening subset
        conf_mask = all_assay_types == AssayType.CONFIRMATORY
        scr_mask = all_assay_types == AssayType.SCREENING

        if conf_mask.sum() > 10:
            preds_c = preds[conf_mask]
            labels_c = all_binary_labels[conf_ma_]()_
