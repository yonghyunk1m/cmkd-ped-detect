"""
Knowledge Distillation model for audio-based pedestrian counting.

Teacher:  Mask2Former (frozen) — pre-computed query embeddings [S, 100, D_teacher]
          + a frozen MLP head that maps [N, D_teacher] → [N, 2] binary logits.
Student:  ASPEDLightningModel (audio-only, seq2seq).

KD signal: configurable combination of losses (see kd_cfg):
  - LogitKD  : temperature-scaled KL-Div on binary logits
  - Cosine   : cosine similarity on projected features
  - CRD      : contrastive InfoNCE on projected features
  - RKD      : relational (distance + angle) on projected features

Loss weighting (kd_cfg.use_uncertainty):
  - True  : Homoscedastic uncertainty MTL (learnable log-variances per task)
  - False : Fixed equal weighting  total = CE + kd_weight * KD_loss(es)

Query aggregation:
  - 'gap'       : Global Average Pooling over 100 queries         → [S, D]
  - 'attention' : Learned scalar-attention pooling over 100 queries → [S, D]

4-class → binary projection for KD loss:
  P(ped present) = softmax(student_logits)[:, 1:].sum(-1)
  This binary distribution is then compared to the teacher's binary output.

Key kd_cfg flags:
  use_logit_kd   : bool  — include LogitKD loss          (default True)
  use_uncertainty: bool  — use homoscedastic MTL          (default True)
  kd_weight      : float — fixed KD weight when not MTL   (default 1.0)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .seq2seq import ASPEDLightningModel
from .losses import FocalLoss, LogitKDLoss, FeatureKDLoss, CRDLoss, RKDLoss


# ---------------------------------------------------------------------------
# Query aggregation modules
# ---------------------------------------------------------------------------

class GAPPooling(nn.Module):
    """Trivial mean pooling over the query dimension."""
    def forward(self, x):  # [B*S, N_q, D] → [B*S, D]
        return x.mean(dim=1)


class AttentionPooling(nn.Module):
    """Learns a scalar attention weight per query via a linear probe."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x):  # [B*S, N_q, D] → [B*S, D]
        scores = self.attn(x).squeeze(-1)          # [B*S, N_q]
        weights = F.softmax(scores, dim=-1)         # [B*S, N_q]
        return (weights.unsqueeze(-1) * x).sum(1)  # [B*S, D]


# ---------------------------------------------------------------------------
# KD model
# ---------------------------------------------------------------------------

class ASPEDKDModel(ASPEDLightningModel):
    """
    Audio-only pedestrian counting model trained with Knowledge Distillation
    from a frozen Mask2Former video teacher.

    ┌─────────────────────── Training loop ─────────────────────────────────┐
    │                                                                         │
    │  Batch = (audio [B,160k], emb [B,S,258], labels [B,S])                 │
    │                                                                         │
    │  ① Student forward  (audio only, full gradient)                         │
    │       raw waveform → VGGish backbone → TransformerEncoder               │
    │       → classifier → student_out [B, S, n_cls]                          │
    │       cls_loss = BCEWithLogitsLoss / CrossEntropy vs GT labels           │
    │                                                                         │
    │  ② Teacher forward  (no gradient, weights frozen)                       │
    │       emb [B,S,258] → compress_mlp [258→128] → classifier_exist [128→2] │
    │       teacher_logits [B*S, 2]  (binary: no-ped vs ped)                  │
    │                                                                         │
    │  ③ KD loss  (temperature-scaled KL-Divergence)                          │
    │       student_out → _student_to_binary_logits → [B*S, 2]                │
    │       kd_loss = T^2 * KLDiv(softmax(teacher/T) || softmax(student/T))         │
    │                                                                         │
    │  ④ Homoscedastic uncertainty weighting                                  │
    │       total = 0.5·exp(−σ_cls)·cls_loss + 0.5·σ_cls                     │
    │             + 0.5·exp(−σ_kd )·kd_loss  + 0.5·σ_kd                     │
    │       σ_cls, σ_kd are learnable log-variances                           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────── Validation / Test ──────────────────────────────┐
    │                                                                         │
    │  val_quick/macro_accuracy  = STUDENT predictions vs GT labels           │
    │  val_quick/loss            = total loss (cls + kd) on val set           │
    │  kd/teacher_student_agreement = fraction of steps where                 │
    │                                 argmax(teacher) == argmax(student)      │
    │                                                                         │
    │  Teacher is NOT used for accuracy evaluation — only for the KD signal.  │
    └─────────────────────────────────────────────────────────────────────────┘

    Logged metrics (WandB):
      train/loss/cls         — task loss (per step)
      train/loss/kd          — KL-Div distillation loss (per step)
      train/loss/total       — weighted total (per step)
      sigma/cls, sigma/kd    — uncertainty weights (per epoch)
      kd/teacher_student_agreement — agreement rate (per epoch)
      val_quick/loss         — val total loss (every 100 steps)
      val_quick/macro_accuracy     — student binary accuracy on val (every 100 steps)
      test/loss, test/macro_accuracy — final test metrics

    Args (kd_cfg dict):
      teacher_dim       : int   — embedding dim (258 for 258-dim, 256 for raw queries)
      query_agg_type    : str   — 'gap' | 'attention' (only used if emb is [B,S,N_q,D])
      teacher_ckpt      : str | None — path to LOSO teacher .ckpt
      kd_temperature    : float — softmax temperature T for KL-Div (default 3.0)
      focal_loss        : bool  — use FocalLoss for n_classes>2 (default True)
      focal_gamma       : float — focal gamma (default 2.0)
    """

    def __init__(self, kd_cfg: dict, **student_kwargs):
        # Initialise student backbone + classifier as usual
        super().__init__(**student_kwargs)
        self.save_hyperparameters()

        teacher_dim      = kd_cfg['teacher_dim']
        self.teacher_n_classes = kd_cfg.get('teacher_n_classes', 2)
        agg_type         = kd_cfg.get('query_agg_type', 'attention')
        teacher_ckpt     = kd_cfg.get('teacher_ckpt', None)
        temperature      = kd_cfg.get('kd_temperature', 3.0)
        use_focal        = kd_cfg.get('focal_loss', True)
        focal_gamma      = kd_cfg.get('focal_gamma', 2.0)
        # Loss control flags
        self.use_logit_kd    = kd_cfg.get('use_logit_kd', True)
        self.use_uncertainty = kd_cfg.get('use_uncertainty', True)
        self.kd_weight       = kd_cfg.get('kd_weight', 1.0)
        # Selective distillation
        self.use_selective_kd     = kd_cfg.get('use_selective_kd', False)
        self.confidence_threshold = kd_cfg.get('confidence_threshold', 0.5)
        self.teacher_dim_cfg      = teacher_dim  # store for confidence source logic
        # Phase 3: Soft label blending
        self.label_blend_alpha    = kd_cfg.get('label_blend_alpha', None)  # None=disabled, 0.7=blend
        # Phase 3: Continuous confidence weighting
        self.confidence_weighting = kd_cfg.get('confidence_weighting', 'none')  # 'none', 'continuous'
        # Phase 3: Class-weighted KD (upweight minority class KD signal)
        self.kd_class_weighting   = kd_cfg.get('kd_class_weighting', False)
        # Phase 3: Teacher un-freeze
        self.unfreeze_teacher     = kd_cfg.get('unfreeze_teacher', False)
        self.teacher_lr           = kd_cfg.get('teacher_lr', 1e-6)
        # Phase 4: Curriculum temperature (per-sample T based on teacher confidence)
        self.curriculum_temperature = kd_cfg.get('curriculum_temperature', False)
        self.T_base               = kd_cfg.get('kd_temperature', 3.0)
        self.T_range              = kd_cfg.get('T_range', 4.0)  # T goes from T_base to T_base+T_range
        # Phase 4: Teacher-GT agreement filtering
        self.filter_teacher_errors = kd_cfg.get('filter_teacher_errors', False)

        # TAD: Transferability-Aware Distillation
        self.use_tad = kd_cfg.get('use_tad', False)
        tad_mode = kd_cfg.get('tad_mode', 'sp')  # 'sp', 'cg', 'learned', 'full'
        if self.use_tad:
            # models/tad.py is not part of this release; the TAD configs are
            # exploratory and are not among the eleven reported in the paper.
            raise NotImplementedError(
                "use_tad is not supported in this release (models/tad.py is "
                "not included). None of the paper's reported configurations "
                "uses it.")
            from models.tad import TADModule
            self.tad = TADModule(
                mode=tad_mode,
                modality='audio',
                student_dim=student_kwargs.get('token_dim', 128),
                teacher_dim=teacher_dim,
                sp_alpha=kd_cfg.get('tad_sp_alpha', 100.0),
                sp_mu=kd_cfg.get('tad_sp_mu', 0.015),
            )
        # Phase 4: Wasserstein KD loss
        self.use_wasserstein_kd   = kd_cfg.get('use_wasserstein_kd', False)
        # Phase 5: Cross-disciplinary approaches
        self.immune_tolerance     = kd_cfg.get('immune_tolerance', False)
        self.immune_threshold     = kd_cfg.get('immune_threshold', 0.4)
        self.kd_shutoff_ratio     = kd_cfg.get('kd_shutoff_ratio', 0.0)  # 0.0=off, 0.8=shutoff last 20%
        self.therapeutic_alpha    = kd_cfg.get('therapeutic_alpha', False)
        # Competitor baselines
        self.use_logit_adjustment = kd_cfg.get('use_logit_adjustment', False)
        self.logit_adj_tau        = kd_cfg.get('logit_adj_tau', 1.0)
        self.use_focal_kd         = kd_cfg.get('use_focal_kd', False)
        self.focal_kd_gamma       = kd_cfg.get('focal_kd_gamma', 2.0)
        # Phase 6: Loss-scale normalization (EMA-based)
        self.loss_norm            = kd_cfg.get('loss_normalization', False)
        self.loss_norm_ema        = kd_cfg.get('loss_norm_ema', 0.999)  # EMA decay
        if self.loss_norm:
            self.register_buffer('_ema_cls', torch.ones(1))
            self.register_buffer('_ema_kd', torch.ones(1))
            self.register_buffer('_ema_feat', torch.ones(1))
            self.register_buffer('_ema_initialized', torch.zeros(1, dtype=torch.bool))

        # ------------------------------------------------------------------ #
        # 1. Query aggregation
        # ------------------------------------------------------------------ #
        if agg_type == 'attention':
            self.query_agg = AttentionPooling(teacher_dim)
        else:
            self.query_agg = GAPPooling()

        # ------------------------------------------------------------------ #
        # 2. Teacher MLP  (frozen)
        # Matches the SavedVideoBaseline checkpoint layout:
        #   compress_mlp : Linear(teacher_dim → 128) + BatchNorm1d(128)
        #   classifier_exist : Linear(128 → 2)
        # NOTE: The released checkpoint was trained on 258-dim embeddings
        #   (256 visual + 2 metadata).  Local embeddings_dataset/ files are
        #   256-dim, so set teacher_dim=256 when using local files (teacher
        #   weights will be random).  Set teacher_dim=258 if you have the
        #   external 258-dim files and a matching checkpoint.
        # ------------------------------------------------------------------ #
        self.compress_mlp      = nn.Sequential(
            nn.Linear(teacher_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Teacher classifier head: binary uses `classifier_exist`, 4-class uses `classifier_count`
        if self.teacher_n_classes == 2:
            self.teacher_head_name = 'classifier_exist'
            self.classifier_exist  = nn.Linear(128, 2)
        else:
            self.teacher_head_name = 'classifier_count'
            self.classifier_count  = nn.Linear(128, self.teacher_n_classes)

        if teacher_ckpt and os.path.exists(teacher_ckpt):
            print(f"[KD] Loading teacher weights from {teacher_ckpt} "
                  f"(teacher_n_classes={self.teacher_n_classes})")
            ckpt = torch.load(teacher_ckpt, map_location='cpu')
            sd   = ckpt.get('state_dict', ckpt)
            head = self.teacher_head_name
            missing, unexpected = self.load_state_dict(
                {k: v for k, v in sd.items()
                 if k.startswith('compress_mlp') or k.startswith(head)},
                strict=False
            )
            if missing:
                print(f"[KD] Teacher: missing keys  -> {missing}")
            if unexpected:
                print(f"[KD] Teacher: unexpected keys -> {unexpected}")
        else:
            print("[KD] ⚠️  No teacher checkpoint — using random teacher weights.")

        if self.unfreeze_teacher:
            print(f"[KD] Teacher MLP UN-FROZEN (lr={self.teacher_lr})")
        else:
            for p in self.compress_mlp.parameters():
                p.requires_grad = False
            for p in self._teacher_head().parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------ #
        # 3. KD loss
        # ------------------------------------------------------------------ #
        self.kd_loss_fn = LogitKDLoss(temperature=temperature)

        # ------------------------------------------------------------------ #
        # 4. Optionally replace classification loss with Focal Loss
        # FocalLoss uses cross_entropy, so it requires [N, C] logits.
        # Binary (num_classes=2) uses BCEWithLogitsLoss → keep as-is.
        # Focal is only useful for multi-class (4-class) task loss.
        # ------------------------------------------------------------------ #
        if use_focal and self.num_classes > 2:
            cw = self.class_weights  # registered buffer  [n_classes]
            self.criterion = FocalLoss(weight=cw, gamma=focal_gamma)

        # ------------------------------------------------------------------ #
        # 5. Feature-level KD (optional — aligns student hidden with teacher embed)
        # ------------------------------------------------------------------ #
        feat_kd_cfg      = kd_cfg.get('feature_kd', {})
        self.use_feat_kd = feat_kd_cfg.get('enabled', False)

        if self.use_feat_kd:
            feat_student_dim = feat_kd_cfg.get('student_feat_dim', 128)
            feat_proj_dim    = feat_kd_cfg.get('proj_dim', 256)
            self.feat_teacher_dim = feat_kd_cfg.get('teacher_feat_dim', 256)
            self.feat_projector = nn.Sequential(
                nn.Linear(feat_student_dim, feat_proj_dim),
                nn.ReLU(),
                nn.Linear(feat_proj_dim, feat_proj_dim),
            )
            feat_kd_type = feat_kd_cfg.get('feat_kd_type', 'cosine')
            if feat_kd_type == 'crd':
                crd_temp = feat_kd_cfg.get('crd_temperature', 0.07)
                self.feat_kd_loss_fn = CRDLoss(temperature=crd_temp)
            elif feat_kd_type == 'rkd':
                rkd_dist_w  = feat_kd_cfg.get('rkd_dist_weight', 1.0)
                rkd_angle_w = feat_kd_cfg.get('rkd_angle_weight', 2.0)
                self.feat_kd_loss_fn = RKDLoss(dist_weight=rkd_dist_w, angle_weight=rkd_angle_w)
            else:  # 'cosine' (default)
                self.feat_kd_loss_fn = FeatureKDLoss()
            print(f"[KD] Feature-level KD enabled ({feat_kd_type}): student {feat_student_dim} → proj {feat_proj_dim} ↔ teacher {self.feat_teacher_dim}")

        # ------------------------------------------------------------------ #
        # 6. Homoscedastic uncertainty — learnable log-variances
        #    Only allocated when use_uncertainty=True.
        # ------------------------------------------------------------------ #
        if self.use_uncertainty:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            if self.use_logit_kd:
                self.log_var_kd = nn.Parameter(torch.zeros(1))
            if self.use_feat_kd:
                self.log_var_feat = nn.Parameter(torch.zeros(1))

        # Print experiment config summary
        kd_types = []
        if self.use_logit_kd:
            kd_types.append('LogitKD')
        if self.use_feat_kd:
            feat_type = kd_cfg.get('feature_kd', {}).get('feat_kd_type', 'cosine')
            kd_types.append(f'FeatKD({feat_type})')
        if self.loss_norm:
            weighting = f'LossNorm(ema={self.loss_norm_ema}, λ={self.kd_weight})'
        elif self.use_uncertainty:
            weighting = 'MTL(uncertainty)'
        else:
            weighting = f'fixed(λ={self.kd_weight})'
        extras = []
        if self.use_selective_kd:
            extras.append(f'TFD-Conf(≥{self.confidence_threshold})')
        if self.label_blend_alpha is not None:
            extras.append(f'SoftBlend(α={self.label_blend_alpha})')
        if self.kd_class_weighting:
            extras.append('ClassWeightKD')
        if self.confidence_weighting == 'continuous':
            extras.append('ConfWeight')
        if self.unfreeze_teacher:
            extras.append(f'TeacherUnfreeze(lr={self.teacher_lr})')
        if self.curriculum_temperature:
            extras.append(f'CurriculumT({self.T_base}-{self.T_base+self.T_range})')
        if self.filter_teacher_errors:
            extras.append('FilterTeacherErrors')
        if self.use_wasserstein_kd:
            extras.append('WassersteinKD')
        if self.immune_tolerance:
            extras.append(f'TFD-Label(>{self.immune_threshold})')
        if self.kd_shutoff_ratio > 0:
            extras.append(f'Shutoff({self.kd_shutoff_ratio})')
        if self.therapeutic_alpha:
            extras.append('TherapeuticAlpha')
        extra_str = '  |  ' + ', '.join(extras) if extras else ''
        print(f"[KD] Losses: CE + {' + '.join(kd_types) if kd_types else '(none — CE only)'}  |  Weighting: {weighting}{extra_str}")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _aggregate_queries(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Handles two embedding formats:
          Pre-aggregated  [B, S, D]       (_embed_258.pt)  → reshape to [B*S, D]
          Raw queries     [B, S, N_q, D]  (local 256-dim)  → pool then [B*S, D]
        """
        if emb.dim() == 3:
            # Already per-second (e.g. 258-dim): skip query pooling
            B, S, D = emb.shape
            return emb.reshape(B * S, D)    # [B*S, D]
        else:
            B, S, N_q, D = emb.shape
            flat = emb.view(B * S, N_q, D)
            return self.query_agg(flat)     # [B*S, D]

    def _extract_teacher_confidence(self, emb: torch.Tensor,
                                       teacher_logits: torch.Tensor) -> torch.Tensor:
        """Extract per-sample teacher confidence for selective distillation.

        Returns [B*S] confidence tensor.
        - If teacher_dim==258: use dim 256 (raw Mask2Former confidence)
        - If teacher_dim==256: derive from teacher logits softmax
        """
        if self.teacher_dim_cfg >= 257 and emb.shape[-1] >= 257:
            # Confidence stored at dim 256
            if emb.dim() == 3:
                B, S, D = emb.shape
                return emb[:, :, 256].reshape(B * S)
            else:
                B, S, N_q, D = emb.shape
                return emb[:, :, :, 256].max(dim=2).values.reshape(B * S)
        else:
            # Derive from teacher logits
            probs = F.softmax(teacher_logits, dim=-1)
            return probs.max(dim=-1).values

    def _teacher_head(self) -> nn.Module:
        """Return the teacher classifier head (binary or 4-class)."""
        return getattr(self, self.teacher_head_name)

    def _student_to_teacher_logits(self, student_out: torch.Tensor) -> torch.Tensor:
        """Project student classifier output to teacher-space logits for KL-Div.

        teacher_n_classes=2  (binary teacher):
          Binary  student [B*S, 1] → [-z,  z]               (BCE-compatible)
          4-class student [B*S, 4] → log([P(0), P(1+2+3)])  (binary projection)
        teacher_n_classes=4  (4-class teacher):
          4-class student [B*S, 4] → student (identity)
          Binary  student [B*S, 1] → [-z, z, 0, 0]  (not a typical use case)

        Returns: [B*S, teacher_n_classes]
        """
        if self.teacher_n_classes == 2:
            if self.num_classes == 2:
                z = student_out  # [B*S, 1]
                return torch.cat([-z, z], dim=-1)  # [B*S, 2]
            else:
                probs   = F.softmax(student_out, dim=-1)          # [B*S, 4]
                p_none  = probs[:, 0:1]                           # P(0 pedestrians)
                p_ped   = probs[:, 1:].sum(dim=-1, keepdim=True)  # P(1+ pedestrians)
                binary  = torch.clamp(torch.cat([p_none, p_ped], dim=-1), min=1e-9)
                return torch.log(binary)                          # [B*S, 2]
        else:
            # 4-class teacher — expect matching student dimensionality
            if self.num_classes == self.teacher_n_classes:
                return student_out  # [B*S, 4]
            if self.num_classes == 2:
                z = student_out
                pad = torch.zeros(z.shape[0], self.teacher_n_classes - 2,
                                  device=z.device, dtype=z.dtype)
                return torch.cat([-z, z, pad], dim=-1)
            raise ValueError(
                f"Unsupported student/teacher pair: student={self.num_classes}, "
                f"teacher={self.teacher_n_classes}")

    # Backward-compat alias (still referenced by older callers / tests)
    def _student_to_binary_logits(self, student_out: torch.Tensor) -> torch.Tensor:
        return self._student_to_teacher_logits(student_out)

    def _cls_loss(self, student_out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the task classification loss (same logic as ASPEDLightningModel)."""
        if self.task_type == 'regression':
            return self.criterion(student_out.squeeze(-1).reshape(-1),
                                  labels.float().reshape(-1))
        elif self.num_classes == 2:
            preds   = student_out.squeeze(-1).reshape(-1)
            targets = (labels > 0).float().reshape(-1)
            return self.criterion(preds, targets)
        else:
            preds        = student_out.reshape(-1, self.num_classes)
            labels_long  = labels.long().reshape(-1)
            # FocalLoss uses internal class weights; WeightedOrdinalEMDLoss accepts sample_weights
            if isinstance(self.criterion, FocalLoss):
                return self.criterion(preds, labels_long)
            else:
                sample_w = self.class_weights[labels_long]
                return self.criterion(preds, labels_long, sample_weights=sample_w)

    # ---------------------------------------------------------------------- #
    # Forward pass (inherited from ASPEDLightningModel — no changes needed)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # Step logic
    # ---------------------------------------------------------------------- #

    def _kd_step(self, batch):
        """Shared computation for train/val/test with KD.

        Returns: (loss_dict, student_raw_out)
          student_raw_out : [B, S, n_classes]  (same shape as self.classifier output)
        """
        audio, emb, labels = batch

        # ---- Student forward ----
        if self.use_feat_kd:
            student_out, student_hidden = self(audio, return_features=True)
            # student_hidden: [B, S, token_dim]
        else:
            student_out = self(audio)

        # ---- Teacher forward (needed before cls_loss if blending) ----
        B, S = labels.shape
        teacher_head = self._teacher_head()
        if self.unfreeze_teacher:
            agg            = self._aggregate_queries(emb)
            teacher_logits = teacher_head(self.compress_mlp(agg))
        else:
            with torch.no_grad():
                agg            = self._aggregate_queries(emb)
                teacher_logits = teacher_head(self.compress_mlp(agg))

        # ---- Classification loss (with optional soft label blending) ----
        if self.label_blend_alpha is not None:
            from models.losses import soft_cross_entropy
            alpha = self.label_blend_alpha
            teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)  # [B*S, teacher_n_classes]
            labels_flat = labels.reshape(-1)
            valid = (labels_flat >= 0)
            target_classes = self.teacher_n_classes
            label_idx = labels_flat.clamp(min=0, max=target_classes - 1)
            hard_targets = F.one_hot(label_idx, target_classes).float()
            blended = alpha * hard_targets + (1 - alpha) * teacher_probs
            student_proj_for_blend = self._student_to_teacher_logits(
                student_out.reshape(B * S, -1))
            cls_loss = soft_cross_entropy(
                student_proj_for_blend[valid], blended[valid],
                weight=self.class_weights if hasattr(self, 'class_weights') else None)
        else:
            cls_loss = self._cls_loss(student_out, labels)

        # ---- Student logits projected into teacher space (used for logit-KD + agreement) ----
        student_flat = student_out.reshape(B * S, -1)
        student_2cls = self._student_to_teacher_logits(student_flat)
        valid_mask   = (labels.reshape(-1) >= 0)

        # ---- Trust-Filtered Distillation (TFD) mask construction ----
        # The paper reports only `immune_tolerance` (Eq. 4): on minority-class
        # samples, drop KD when the teacher's no-pedestrian probability exceeds
        # `immune_threshold` (0.4). NOTE this is a probability threshold, not an
        # argmax test: at 0.4 it also covers minority samples the teacher gets
        # right with P(ped) < 0.6.
        # `use_selective_kd` and `filter_teacher_errors` are exploratory
        # variants that are not reported.
        kd_mask = valid_mask.clone()
        if self.use_selective_kd:
            # TFD-Conf
            with torch.no_grad():
                conf = self._extract_teacher_confidence(emb, teacher_logits)
                kd_mask = kd_mask & (conf >= self.confidence_threshold)
        if self.filter_teacher_errors:
            with torch.no_grad():
                t_pred = teacher_logits.argmax(dim=-1)
                gt_flat = labels.reshape(-1).clamp(min=0)
                kd_mask = kd_mask & (t_pred == gt_flat)
        if self.immune_tolerance:
            # Eq. 4: suppress KD on minority samples where the teacher's target
            # leans to the majority class (sigma(z_t)_0 > tau, tau = 0.4).
            # Minority = any pedestrian present (gt>=1), majority = class 0 (no pedestrian).
            with torch.no_grad():
                t_probs = F.softmax(teacher_logits, dim=-1)
                gt_flat = labels.reshape(-1).clamp(min=0)
                suppress = (gt_flat >= 1) & (t_probs[:, 0] > self.immune_threshold)
                kd_mask = kd_mask & ~suppress

        # ---- KD shutoff: disable KD in final training phase ----
        if self.kd_shutoff_ratio > 0 and hasattr(self, 'trainer') and self.trainer is not None:
            progress = self.trainer.current_epoch / max(self.trainer.max_epochs, 1)
            if progress >= self.kd_shutoff_ratio:
                kd_mask = torch.zeros_like(kd_mask)

        # ---- Logit Adjustment (Menon 2021): debias teacher for minority ----
        if self.use_logit_adjustment:
            if self.teacher_n_classes != 2:
                raise NotImplementedError(
                    "use_logit_adjustment is only implemented for binary teachers; "
                    f"got teacher_n_classes={self.teacher_n_classes}")
            with torch.no_grad():
                adjustment = torch.zeros_like(teacher_logits)
                adjustment[:, 1] += self.logit_adj_tau
                teacher_logits = teacher_logits + adjustment

        # ---- Logit-level KD loss (optional) ----
        if self.use_logit_kd and kd_mask.any():
            if self.use_wasserstein_kd:
                # Wasserstein (EMD) for binary: |CDF_s - CDF_t|
                s_prob = torch.sigmoid(student_2cls[kd_mask, 1] - student_2cls[kd_mask, 0])
                t_prob_w = torch.sigmoid(teacher_logits[kd_mask, 1] - teacher_logits[kd_mask, 0])
                per_sample = (s_prob - t_prob_w.detach()).abs()  # [N]
            elif self.curriculum_temperature:
                # Per-sample temperature: easy(high conf)→low T, hard(low conf)→high T
                conf = self._extract_teacher_confidence(emb, teacher_logits)
                T_per = self.T_base + self.T_range * (1.0 - conf[kd_mask])  # [N]
                T_per = T_per.unsqueeze(-1)  # [N, 1] for broadcasting
                s_logits_m = student_2cls[kd_mask]
                t_logits_m = teacher_logits[kd_mask].detach()
                s_log = F.log_softmax(s_logits_m / T_per, dim=-1)
                t_prob = F.softmax(t_logits_m / T_per, dim=-1)
                per_sample = -(t_prob * s_log).sum(dim=-1) * (T_per.squeeze() ** 2)
            else:
                T = self.kd_loss_fn.T
                s_log = F.log_softmax(student_2cls[kd_mask] / T, dim=-1)
                t_prob = F.softmax(teacher_logits[kd_mask].detach() / T, dim=-1)
                per_sample = -(t_prob * s_log).sum(dim=-1) * (T ** 2)  # [N]

            # Per-sample weighting
            if self.therapeutic_alpha:
                # Class-conditional KD strength: high for majority, low for minority
                gt_masked = labels.reshape(-1)[kd_mask]
                # Majority (class 0): alpha=0.8, Minority (class 1): alpha=0.2
                alpha = torch.where(gt_masked == 0, 0.8, 0.2)
                kd_loss = (per_sample * alpha).mean()
            elif self.kd_class_weighting and hasattr(self, 'class_weights'):
                cw = self.class_weights[labels.reshape(-1)[kd_mask]]
                kd_loss = (per_sample * cw).mean()
            elif self.confidence_weighting == 'continuous':
                conf = self._extract_teacher_confidence(emb, teacher_logits)
                kd_loss = (per_sample * conf[kd_mask]).mean()
            elif self.use_tad:
                tau = self.tad(
                    student_input=audio.reshape(B * S, -1) if audio.dim() == 2 else audio,
                    teacher_logits=teacher_logits,
                    gt_labels=labels.reshape(-1),
                    student_features=student_hidden.reshape(B * S, -1),
                    teacher_features=emb[:, :, :self.tad.learned_estimator.net[0].in_features - student_hidden.size(-1)].reshape(B * S, -1) if hasattr(self.tad, 'learned_estimator') else None,
                )
                if tau.dim() > 0 and tau.shape[0] == B * S:
                    tau_masked = tau[kd_mask]
                else:
                    tau_masked = tau
                kd_loss = (per_sample * tau_masked).mean()
            else:
                kd_loss = per_sample.mean()
            # Focal-KD: upweight hard samples (low teacher confidence on predicted class)
            if self.use_focal_kd and kd_loss.item() > 0:
                with torch.no_grad():
                    t_prob_max = F.softmax(teacher_logits[kd_mask].detach(), dim=-1).max(dim=-1).values
                    focal_weight = (1.0 - t_prob_max) ** self.focal_kd_gamma
                kd_loss = (per_sample * focal_weight).mean()
        else:
            kd_loss = torch.tensor(0.0, device=self.device)

        # ---- Feature-level KD loss (optional) ----
        if self.use_feat_kd:
            student_feat_flat = student_hidden.reshape(B * S, -1)
            student_proj = self.feat_projector(student_feat_flat)

            with torch.no_grad():
                if emb.dim() == 4:
                    teacher_feat = self._aggregate_queries(emb)[:, :self.feat_teacher_dim]
                else:
                    teacher_feat = emb[:, :, :self.feat_teacher_dim].reshape(B * S, -1)

            if kd_mask.any():
                feat_loss = self.feat_kd_loss_fn(student_proj[kd_mask],
                                                 teacher_feat[kd_mask])
            else:
                feat_loss = torch.tensor(0.0, device=self.device)
        else:
            feat_loss = None

        # ---- Loss normalization (EMA-based scale matching) ----
        if self.loss_norm and self.training:
            with torch.no_grad():
                alpha = self.loss_norm_ema
                cls_val = cls_loss.detach().clamp(min=1e-8)
                if not self._ema_initialized:
                    # First step: initialize EMA with current values
                    self._ema_cls.fill_(cls_val)
                    if self.use_logit_kd:
                        self._ema_kd.fill_(kd_loss.detach().clamp(min=1e-8))
                    if feat_loss is not None:
                        self._ema_feat.fill_(feat_loss.detach().clamp(min=1e-8))
                    self._ema_initialized.fill_(True)
                else:
                    self._ema_cls.mul_(alpha).add_(cls_val, alpha=1 - alpha)
                    if self.use_logit_kd:
                        self._ema_kd.mul_(alpha).add_(kd_loss.detach().clamp(min=1e-8), alpha=1 - alpha)
                    if feat_loss is not None:
                        self._ema_feat.mul_(alpha).add_(feat_loss.detach().clamp(min=1e-8), alpha=1 - alpha)

        # ---- Loss combination ----
        if self.loss_norm:
            # Normalize all losses to unit scale, then combine with kd_weight
            norm_cls = cls_loss / self._ema_cls.clamp(min=1e-8)
            total = norm_cls
            if self.use_logit_kd:
                norm_kd = kd_loss / self._ema_kd.clamp(min=1e-8)
                total = total + self.kd_weight * norm_kd
            if feat_loss is not None:
                norm_feat = feat_loss / self._ema_feat.clamp(min=1e-8)
                total = total + self.kd_weight * norm_feat
        elif self.use_uncertainty:
            # Homoscedastic uncertainty MTL (learnable log-variances)
            prec_cls = torch.exp(-self.log_var_cls)
            total    = 0.5 * prec_cls * cls_loss + 0.5 * self.log_var_cls
            if self.use_logit_kd:
                prec_kd = torch.exp(-self.log_var_kd)
                total  += 0.5 * prec_kd * kd_loss + 0.5 * self.log_var_kd
            if feat_loss is not None:
                prec_feat = torch.exp(-self.log_var_feat)
                total    += 0.5 * prec_feat * feat_loss + 0.5 * self.log_var_feat
        else:
            # Fixed equal weighting: total = CE + kd_weight * KD_loss(es)
            total = cls_loss
            if self.use_logit_kd:
                total = total + self.kd_weight * kd_loss
            if feat_loss is not None:
                total = total + self.kd_weight * feat_loss

        # ---- Teacher-student agreement (monitoring only) ----
        with torch.no_grad():
            t_pred = teacher_logits.argmax(dim=-1)
            s_pred = student_2cls.argmax(dim=-1)
            agree  = (t_pred == s_pred).float().mean()

        loss_dict = {
            'loss/cls':   cls_loss,
            'loss/kd':    kd_loss,
            'loss/total': total,
            'kd/teacher_student_agreement': agree,
        }
        if self.loss_norm:
            loss_dict['loss_norm/ema_cls'] = self._ema_cls.squeeze()
            if self.use_logit_kd:
                loss_dict['loss_norm/ema_kd'] = self._ema_kd.squeeze()
            if feat_loss is not None:
                loss_dict['loss_norm/ema_feat'] = self._ema_feat.squeeze()
            loss_dict['loss_norm/ratio'] = (self._ema_kd.squeeze() / self._ema_cls.clamp(min=1e-8).squeeze()) if self.use_logit_kd else torch.tensor(0.0)
        if self.use_uncertainty:
            loss_dict['sigma/cls'] = torch.exp(-self.log_var_cls).sqrt().squeeze()
            if self.use_logit_kd:
                loss_dict['sigma/kd'] = torch.exp(-self.log_var_kd).sqrt().squeeze()
        if feat_loss is not None:
            loss_dict['loss/feat_kd'] = feat_loss
            if self.use_uncertainty:
                loss_dict['sigma/feat'] = torch.exp(-self.log_var_feat).sqrt().squeeze()
        if self.use_selective_kd:
            loss_dict['kd/confident_ratio'] = kd_mask.float().mean()

        return loss_dict, student_out

    # ---------------------------------------------------------------------- #
    # Lightning hooks  (override parent train/val/test steps)
    # ---------------------------------------------------------------------- #

    def training_step(self, batch, batch_idx):
        loss_dict, student_out = self._kd_step(batch)
        epoch_only = {'sigma/cls', 'sigma/kd', 'sigma/feat', 'kd/teacher_student_agreement'}
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v,
                     on_step=(k not in epoch_only),
                     on_epoch=True,
                     prog_bar=(k == 'loss/total'))
        return loss_dict['loss/total']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio, emb, labels = batch
        loss_dict, student_out = self._kd_step(batch)

        # Log pure CE loss for val/test (not homoscedastic total, which can be negative)
        prefix = 'val_quick' if dataloader_idx == 0 else 'test'
        self.log(f'{prefix}/loss', loss_dict['loss/cls'],
                 on_epoch=True, add_dataloader_idx=False)

        # Update confusion matrix metrics (same as parent)
        if self.task_type == 'regression':
            preds_flat  = student_out.squeeze(-1).reshape(-1)
            labels_flat = labels.float().reshape(-1)
            self.val_metrics.update(preds_flat, labels_flat)
        elif self.num_classes == 2:
            preds_flat   = student_out.squeeze(-1).reshape(-1)
            prob         = torch.sigmoid(preds_flat)
            pred_labels  = (prob > self.binary_threshold).long()
            self.val_preds_list.append(pred_labels.detach())
            self.val_metrics.update(pred_labels, (labels > 0).long().reshape(-1))
        else:
            preds_flat  = student_out.reshape(-1, self.num_classes)
            labels_long = labels.long().reshape(-1)
            self.val_preds_list.append(preds_flat.argmax(dim=-1).detach())
            self.val_metrics.update(preds_flat, labels_long)

        return loss_dict['loss/total']

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx=1)

    def configure_optimizers(self):
        # Student params (inherited behavior)
        teacher_prefixes = ('compress_mlp', self.teacher_head_name)
        student_params = [p for n, p in self.named_parameters()
                         if p.requires_grad and not n.startswith(teacher_prefixes)]
        param_groups = [{'params': student_params}]

        # Teacher params (if un-frozen)
        if self.unfreeze_teacher:
            teacher_params = list(self.compress_mlp.parameters()) + \
                           list(self._teacher_head().parameters())
            param_groups.append({'params': teacher_params, 'lr': self.teacher_lr})

        from torch import optim
        optimizer = optim.Adam(param_groups,
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
