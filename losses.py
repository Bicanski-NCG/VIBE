import torch
import torch.nn.functional as F

def masked_negative_pearson_loss(pred, target, mask, eps=1e-8, zero_center=False):
    mask = mask.unsqueeze(-1)
    pred = pred * mask
    target = target * mask

    if zero_center:
        pred_mean = pred.sum(dim=1) / (mask.sum(dim=1) + eps)
        target_mean = target.sum(dim=1) / (mask.sum(dim=1) + eps)

        pred = pred - pred_mean.unsqueeze(1)
        target = target - target_mean.unsqueeze(1)

    numerator = (pred * target * mask).sum(dim=1)
    denominator = torch.sqrt(((pred**2 * mask).sum(dim=1)) *
                             ((target**2 * mask).sum(dim=1)) + eps)

    corr = numerator / (denominator + eps)
    return -corr.mean()


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _centre(x, mask=None, eps=1e-8):
    """centre over time, keep padded steps at 0."""
    if mask is not None:
        valid = mask.sum(1, keepdim=True).clamp_min(1.)      # (B,1)
        x = x * mask.unsqueeze(-1)
    else:
        valid = x.new_full((x.size(0), 1), x.size(1))
    mean = x.sum(1, keepdim=True) / valid.unsqueeze(-1)
    xc   = x - mean
    if mask is not None:
        xc = xc * mask.unsqueeze(-1)
    return xc, valid.squeeze(1)                               # (B,S,R), (B,)

# ------------------------------------------------------------------
# 1) sample × sample correlation   –   O(B²)
# ------------------------------------------------------------------
def _sample_corr_matrix(x, mask=None, eps=1e-8):
    xc, _ = _centre(x, mask, eps)
    feat  = F.normalize(xc.flatten(1), dim=1)                 # ℓ₂-norm rows
    return feat @ feat.T                                      # (B,B)

def sample_similarity_loss(pred, target, mask=None, eps=1e-8):
    C_pred   = _sample_corr_matrix(pred,   mask, eps)
    C_target = _sample_corr_matrix(target, mask, eps)
    C_pred.diagonal().zero_()                                 # in-place
    C_target.diagonal().zero_()
    return F.mse_loss(C_pred, C_target)

# ------------------------------------------------------------------
# 2) ROI × ROI correlation   –   batched O(R²)
# ------------------------------------------------------------------
def _roi_cov(xc, valid, eps=1e-8):
    """xc: centred (B,S,R); valid: (B,)"""
    # transpose so we can batch-matmul: (B, R, S) @ (B, S, R) → (B,R,R)
    xs = xc.transpose(1,2)
    cov = xs @ xc / (valid.view(-1,1,1) - 1 + eps)
    return cov

def _roi_corr_matrix(x, mask=None, eps=1e-8):
    xc, valid = _centre(x, mask, eps)                         # (B,S,R)
    cov = _roi_cov(xc, valid, eps)                       # (B,R,R)
    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0.) + eps)
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + eps)
    return corr.clamp(-1., 1.)

def roi_similarity_loss(pred, target, mask=None, eps=1e-8):
    C_pred   = _roi_corr_matrix(pred,   mask, eps)
    C_target = _roi_corr_matrix(target, mask, eps)
    eye = torch.eye(C_pred.size(-1), device=pred.device, dtype=pred.dtype)
    C_pred   = C_pred   - eye
    C_target = C_target - eye
    return F.mse_loss(C_pred, C_target)
