"""
model.py — PU-Aware Contrastive Learning Model for Assay-Compound VS
=====================================================================
Architecture:
    1. AssayEncoder    : Hierarchical (Text + StructuredMeta + Context) → h_assay
    2. CompoundEncoder : APM → Conv1D + BiLSTM + Attention → h_compound
    3. AssayPriorNet   : h_assay → π_a  (assay-level class prior)
    4. PUGate          : (h_assay, h_compound, π_a) → confidence weights
    5. ProjectionHead  : Shared projection into contrastive embedding space
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1.  Compound Encoder  (APM-based)
# ═══════════════════════════════════════════════════════════════

class APMEncoder(nn.Module):
    """
    Atom Pair Map Encoder.
    Input:  (B, max_atoms, max_atoms, n_features)  — APM tensor
    Output: (B, d_compound)

    Architecture:
        1. Flatten atom-pair features → (B, max_atoms*max_atoms, n_features)
        2. Conv1D blocks for local pattern extraction
        3. BiLSTM for sequential dependencies
        4. Multi-head Self-Attention for global interactions
        5. Mean pooling → compound embedding
    """

    def __init__(
        self,
        max_atoms: int = 64,
        n_features: int = 8,
        conv_channels: int = 128,
        lstm_hidden: int = 256,
        n_attn_heads: int = 4,
        d_compound: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.n_features = n_features

        # Conv1D blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Multi-head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # bidirectional
            num_heads=n_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden * 2)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, d_compound),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_compound, d_compound),
        )

    def forward(self, apm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            apm: (B, max_atoms, max_atoms, n_features)
        Returns:
            h_compound: (B, d_compound)
        """
        B = apm.size(0)

        # Reshape: flatten atom pairs → sequence
        # (B, max_atoms, max_atoms, F) → (B, max_atoms*max_atoms, F)
        x = apm.view(B, -1, self.n_features)

        # Conv1D: (B, seq_len, F) → (B, F, seq_len) → conv → (B, C, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)  # (B, seq_len, C)

        # Adaptive pooling to reduce sequence length for LSTM efficiency
        # (B, seq_len, C) → (B, max_atoms, C)  via mean pooling over atom pairs
        x = x.view(B, self.max_atoms, self.max_atoms, -1).mean(dim=2)  # (B, max_atoms, C)

        # BiLSTM
        x, _ = self.lstm(x)  # (B, max_atoms, 2*lstm_hidden)

        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)  # residual + LayerNorm

        # Mean pooling over atoms
        # Create mask for padded atoms (all-zero APM rows)
        atom_mask = apm[:, :, 0, 0].abs().sum(dim=-1) if apm.dim() == 4 else None
        x = x.mean(dim=1)  # (B, 2*lstm_hidden)

        # Project
        h_compound = self.output_proj(x)  # (B, d_compound)

        return h_compound


# ═══════════════════════════════════════════════════════════════
# 2.  Assay Encoder  (Hierarchical)
# ═══════════════════════════════════════════════════════════════

class StructuredMetadataEncoder(nn.Module):
    """
    Encodes structured assay metadata fields via field-wise embeddings
    + cross-field Transformer attention.
    """

    def __init__(self, field_configs: Dict[str, int], d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.field_names = list(field_configs.keys())
        self.field_embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_size + 1, d_model)  # +1 for <unk>
            for field, vocab_size in field_configs.items()
        })
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            dropout=dropout, batch_first=True,
        )
        self.cross_field_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, field_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            field_ids: {field_name: (B,) tensor of indices}
        Returns:
            (B, d_model)
        """
        tokens = []
        for field_name in self.field_names:
            if field_name in field_ids:
                emb = self.field_embeddings[field_name](field_ids[field_name])  # (B, d_model)
                tokens.append(emb)

        if not tokens:
            return torch.zeros(field_ids[self.field_names[0]].size(0), 256)

        token_seq = torch.stack(tokens, dim=1)  # (B, n_fields, d_model)
        attended = self.cross_field_attn(token_seq)  # (B, n_fields, d_model)
        return attended.mean(dim=1)  # (B, d_model)


class TextEncoder(nn.Module):
    """
    Lightweight text encoder placeholder.
    In production, replace with PubChemDeBERTa:
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("your-pubchem-deberta")
    """

    def __init__(self, vocab_size: int = 30000, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, d_out: int = 768, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            (B, d_out)
        """
        x = self.embedding(input_ids)  # (B, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Create padding mask for transformer
        # TransformerEncoder expects: True = ignore
        src_key_padding_mask = (attention_mask == 0)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.output_proj(x)  # (B, d_out)


class AssayEncoder(nn.Module):
    """
    Hierarchical Assay Encoder:
        h_text (768) + h_meta (256) + h_ctx (128) → AssayProjector → h_assay (d_embed)
    """

    def __init__(
        self,
        field_configs: Dict[str, int],
        d_text: int = 768,
        d_meta: int = 256,
        d_ctx_input: int = 2,
        d_ctx: int = 128,
        d_embed: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(d_out=d_text)
        self.meta_encoder = StructuredMetadataEncoder(field_configs, d_model=d_meta)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(d_ctx_input, d_ctx),
            nn.BatchNorm1d(d_ctx),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ctx, d_ctx),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_text + d_meta + d_ctx, d_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed, d_embed),
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        struct_meta: Dict[str, torch.Tensor],
        ctx_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns: (B, d_embed) assay embedding"""
        h_text = self.text_encoder(text_input_ids, text_attention_mask)  # (B, d_text)
        h_meta = self.meta_encoder(struct_meta)  # (B, d_meta)
        h_ctx = self.ctx_encoder(ctx_features)  # (B, d_ctx)

        h_concat = torch.cat([h_text, h_meta, h_ctx], dim=-1)
        h_assay = self.projector(h_concat)  # (B, d_embed)
        return h_assay


# ═══════════════════════════════════════════════════════════════
# 3.  Assay Prior Network
# ═══════════════════════════════════════════════════════════════

class AssayPriorNetwork(nn.Module):
    """
    Predicts assay-level class prior π_a from assay embedding.
    Output ∈ [pi_min, pi_max].
    """

    def __init__(self, d_embed: int = 512, pi_min: float = 0.001, pi_max: float = 0.30):
        super().__init__()
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.head = nn.Sequential(
            nn.Linear(d_embed, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_assay: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_assay: (B, d_embed)
        Returns:
            pi_a: (B,) estimated class prior per assay
        """
        raw = self.head(h_assay).squeeze(-1)  # (B,)
        pi = self.pi_min + (self.pi_max - self.pi_min) * raw
        return pi


# ═══════════════════════════════════════════════════════════════
# 4.  PU Gate
# ═══════════════════════════════════════════════════════════════

class PUGate(nn.Module):
    """
    Dynamic confidence weighting for (assay, compound) pairs.
    Outputs weight w ∈ [0, 1]:
        w → 1.0 : high confidence this compound is true negative for this assay
        w → 0.0 : this compound may be a hidden positive (false negative)
    """

    def __init__(self, d_embed: int = 512, gate_type: str = "bilinear"):
        super().__init__()
        self.gate_type = gate_type

        if gate_type == "bilinear":
            self.bilinear = nn.Bilinear(d_embed, d_embed, 1)
            self.temperature = nn.Parameter(torch.tensor(1.0))
        elif gate_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_embed * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        elif gate_type == "similarity":
            self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_assay: torch.Tensor,
        h_compound: torch.Tensor,
        pi_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_assay: (B_a, d)
            h_compound: (B_c, d)
            pi_a: (B_a,) class prior per assay
        Returns:
            weights: (B_a, B_c) confidence weights
        """
        B_a, d = h_assay.shape
        B_c = h_compound.size(0)

        if self.gate_type == "bilinear":
            # Expand for pairwise computation
            a_exp = h_assay.unsqueeze(1).expand(-1, B_c, -1)  # (B_a, B_c, d)
            c_exp = h_compound.unsqueeze(0).expand(B_a, -1, -1)
            raw = self.bilinear(
                a_exp.reshape(-1, d),
                c_exp.reshape(-1, d)
            ).reshape(B_a, B_c)
            raw_weights = torch.sigmoid(raw / self.temperature.abs().clamp(min=0.1))

        elif self.gate_type == "mlp":
            a_exp = h_assay.unsqueeze(1).expand(-1, B_c, -1)
            c_exp = h_compound.unsqueeze(0).expand(B_a, -1, -1)
            pair_feat = torch.cat([a_exp, c_exp], dim=-1)  # (B_a, B_c, 2d)
            raw_weights = torch.sigmoid(self.mlp(pair_feat).squeeze(-1))

        elif self.gate_type == "similarity":
            sim = F.cosine_similarity(
                h_assay.unsqueeze(1), h_compound.unsqueeze(0), dim=-1
            )
            # High similarity → low weight (potential false negative)
            raw_weights = 1.0 - torch.sigmoid((sim - 0.3) / self.temperature.abs().clamp(min=0.1))

        # Prior adjustment: higher π_a → more hidden positives → lower weights overall
        prior_adj = (1.0 - pi_a).unsqueeze(1)  # (B_a, 1)
        weights = raw_weights * prior_adj

        return weights.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 5.  Full Model
# ═══════════════════════════════════════════════════════════════

class PUContrastiveVSModel(nn.Module):
    """
    Full PU-Aware Contrastive Virtual Screening Model.

    Components:
        - AssayEncoder    → h_assay   (d_embed)
        - CompoundEncoder → h_compound (d_embed)
        - AssayPriorNet   → π_a
        - PUGate          → confidence weights
    """

    def __init__(
        self,
        field_configs: Dict[str, int],
        max_atoms: int = 64,
        n_apm_features: int = 8,
        d_embed: int = 512,
        d_text: int = 768,
        d_meta: int = 256,
        d_ctx: int = 128,
        conv_channels: int = 128,
        lstm_hidden: int = 256,
        n_attn_heads: int = 4,
        gate_type: str = "bilinear",
        pi_min: float = 0.001,
        pi_max: float = 0.30,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_embed = d_embed

        # Encoders
        self.assay_encoder = AssayEncoder(
            field_configs=field_configs,
            d_text=d_text,
            d_meta=d_meta,
            d_ctx=d_ctx,
            d_embed=d_embed,
            dropout=dropout,
        )
        self.compound_encoder = APMEncoder(
            max_atoms=max_atoms,
            n_features=n_apm_features,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            n_attn_heads=n_attn_heads,
            d_compound=d_embed,
            dropout=dropout,
        )

        # PU components
        self.prior_network = AssayPriorNetwork(d_embed=d_embed, pi_min=pi_min, pi_max=pi_max)
        self.pu_gate = PUGate(d_embed=d_embed, gate_type=gate_type)

    def encode_assay(self, batch: Dict) -> torch.Tensor:
        """Encode assay from batch → L2-normalized embedding."""
        h = self.assay_encoder(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            struct_meta=batch["struct_meta"],
            ctx_features=batch["ctx_features"],
        )
        return F.normalize(h, p=2, dim=-1)

    def encode_compound(self, batch: Dict) -> torch.Tensor:
        """Encode compound from batch → L2-normalized embedding."""
        h = self.compound_encoder(batch["apm"])
        return F.normalize(h, p=2, dim=-1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            h_assay:    (B, d_embed) L2-normalized
            h_compound: (B, d_embed) L2-normalized
            pi_a:       (B,) estimated class prior
            pu_weights: (B, B) pairwise confidence weights
            sim_matrix: (B, B) cosine similarity matrix
        """
        h_assay = self.encode_assay(batch)
        h_compound = self.encode_compound(batch)
        pi_a = self.prior_network(h_assay)
        pu_weights = self.pu_gate(h_assay, h_compound, pi_a)
        sim_matrix = torch.mm(h_assay, h_compound.T)  # cosine sim (already normalized)

        return {
            "h_assay": h_assay,
            "h_compound": h_compound,
            "pi_a": pi_a,
            "pu_weights": pu_weights,
            "sim_matrix": sim_matrix,
        }


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from dataset import FIELD_CONFIGS

    B = 4
    max_atoms = 32
    n_features = 8

    model = PUContrastiveVSModel(
        field_configs=FIELD_CONFIGS,
        max_atoms=max_atoms,
        n_apm_features=n_features,
        d_embed=256,
        d_text=256,
        d_meta=128,
        d_ctx=64,
        conv_channels=64,
        lstm_hidden=128,
        n_attn_heads=4,
    )

    # Dummy batch
    batch = {
        "apm": torch.randn(B, max_atoms, max_atoms, n_features),
        "text_input_ids": torch.randint(0, 1000, (B, 128)),
        "text_attention_mask": torch.ones(B, 128, dtype=torch.long),
        "struct_meta": {k: torch.randint(0, 10, (B,)) for k in FIELD_CONFIGS},
        "ctx_features": torch.randn(B, 2),
        "pu_label": torch.tensor([1, -1, 0, 1]),
        "alpha": torch.tensor([0.95, 0.0, 0.01, 0.36]),
        "beta": torch.tensor([0.0, 0.90, 0.99, 0.64]),
    }

    outputs = model(batch)
    print("Model outputs:")
    for k, v in outputs.items():
        print(f"  {k}: shape={v.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")
