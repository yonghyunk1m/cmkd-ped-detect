import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(logits, soft_targets, weight=None):
    """Cross-entropy with soft (non-one-hot) targets.

    Args:
        logits:       [N, C] raw logits
        soft_targets: [N, C] probability targets (sum to 1)
        weight:       [C] optional class weights
    Returns:
        scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if weight is not None:
        # weight per sample = sum of target * class_weight
        w = (soft_targets * weight.unsqueeze(0)).sum(dim=-1)  # [N]
        loss = -(soft_targets * log_probs).sum(dim=-1) * w
    else:
        loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance.
    Ignores padding labels (ignore_index=-1).
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            reduction='none', ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        valid_mask = (targets != self.ignore_index)
        if self.reduction == 'mean':
            return focal_loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss[valid_mask].sum()
        return focal_loss * valid_mask


class LogitKDLoss(nn.Module):
    """Temperature-scaled KL-Divergence for logit-level Knowledge Distillation.

    Expects student as log-probs and teacher as probs (standard KL-Div interface).
    The caller is responsible for projecting student/teacher logits to the same
    number of classes before passing them here.
    """
    def __init__(self, temperature=3.0):
        super().__init__()
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: [N, C] raw logits from student
            teacher_logits: [N, C] raw logits from teacher
        Returns:
            scalar KD loss
        """
        student_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)
        return self.kl(student_log_probs, teacher_probs) * (self.T ** 2)


class FeatureKDLoss(nn.Module):
    """Cosine similarity loss for feature-level Knowledge Distillation.

    Aligns student hidden features with teacher embeddings in a shared
    projected space.  Loss = 1 - mean(cos_sim), range [0, 2].
    """
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, student_proj, teacher_feat):
        """
        Args:
            student_proj: [N, D] projected student features
            teacher_feat: [N, D] teacher embedding (same dim after projection)
        Returns:
            scalar cosine distance loss
        """
        return (1 - self.cos(student_proj, teacher_feat)).mean()


class CRDLoss(nn.Module):
    """Contrastive Representation Distillation (CRD).

    Uses InfoNCE: for each sample, the student feature should be closest to
    the *same-timestep* teacher feature (positive) and far from all other
    teacher features in the batch (negatives).

    Inputs are feature vectors (not logits).  A projection MLP that maps
    student → shared space should be applied *before* calling this loss.

    Reference: Tian et al., "Contrastive Representation Distillation", ICLR 2020.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: [N, D] L2-normalised student features
            teacher_feat: [N, D] L2-normalised teacher features (detached)
        Returns:
            scalar InfoNCE loss
        """
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        # [N, N] similarity matrix
        logits = s @ t.T / self.T
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class RKDLoss(nn.Module):
    """Relational Knowledge Distillation (RKD).

    Transfers *inter-sample relations* rather than absolute feature values.
    Two components:
      - Distance-wise: pairwise Euclidean distance distribution should match
      - Angle-wise: angle formed by triplets of samples should match

    Reference: Park et al., "Relational Knowledge Distillation", CVPR 2019.
    """
    def __init__(self, dist_weight: float = 1.0, angle_weight: float = 2.0):
        super().__init__()
        self.dist_w = dist_weight
        self.angle_w = angle_weight

    @staticmethod
    def _pdist(feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Pairwise Euclidean distance, mean-normalised. NaN-safe.

        Uses sqrt(x + eps) instead of sqrt(x) to keep gradients finite when
        feat[i] == feat[j] (including the diagonal where diff == 0).
        """
        diff = feat.unsqueeze(0) - feat.unsqueeze(1)          # [N, N, D]
        dist = torch.sqrt(diff.pow(2).sum(-1) + eps)          # [N, N], stable at 0
        # Mask out diagonal (where dist ~= sqrt(eps)); guard against empty mask.
        N = feat.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=feat.device)
        off_diag = dist[mask]
        mu = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(1.0, device=feat.device)
        return dist / mu.clamp(min=1e-8)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: [N, D] projected student features
            teacher_feat: [N, D] teacher features (detached)
        Returns:
            scalar RKD loss
        """
        loss = torch.tensor(0.0, device=student_feat.device)
        eps = 1e-6

        # --- Distance-wise ---
        if self.dist_w > 0:
            t_dist = self._pdist(teacher_feat)
            s_dist = self._pdist(student_feat)
            loss = loss + self.dist_w * F.smooth_l1_loss(s_dist, t_dist)

        # --- Angle-wise (subsample to avoid O(N^3) memory) ---
        if self.angle_w > 0 and student_feat.size(0) >= 3:
            N = student_feat.size(0)
            max_n = min(N, 64)  # cap to avoid OOM on [N,N,N] tensor
            if N > max_n:
                idx = torch.randperm(N, device=student_feat.device)[:max_n]
                tf_sub, sf_sub = teacher_feat[idx], student_feat[idx]
            else:
                tf_sub, sf_sub = teacher_feat, student_feat

            td = tf_sub.unsqueeze(0) - tf_sub.unsqueeze(1)  # [M,M,D]
            # F.normalize has its own eps, but zero-length vectors still
            # produce zero-norm gradients; use a larger eps for safety.
            t_norm = F.normalize(td, dim=-1, eps=eps)
            t_angle = (t_norm.unsqueeze(1) * t_norm.unsqueeze(0)).sum(-1)  # [M,M,M]

            sd = sf_sub.unsqueeze(0) - sf_sub.unsqueeze(1)
            s_norm = F.normalize(sd, dim=-1, eps=eps)
            s_angle = (s_norm.unsqueeze(1) * s_norm.unsqueeze(0)).sum(-1)

            loss = loss + self.angle_w * F.smooth_l1_loss(s_angle, t_angle)

        return loss


class WeightedOrdinalEMDLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets, sample_weights=None):
        # 1. Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # 2. Calculate Predicted CDF
        pred_cdf = torch.cumsum(probs, dim=1)
        
        # 3. Calculate Target CDF
        class_range = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        target_cdf = (class_range >= targets.unsqueeze(1)).float()
        
        # 4. MSE calculation (per sample)
        mse_per_sample = F.mse_loss(pred_cdf, target_cdf, reduction='none').mean(dim=1)
        
        if sample_weights is not None:
            return (mse_per_sample * sample_weights).sum() / sample_weights.sum()
        
        return mse_per_sample.mean()