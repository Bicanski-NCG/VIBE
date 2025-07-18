import torch
import torch.nn.functional as F


def masked_negative_pearson_loss(pred, target, mask, eps=1e-8, zero_center=True, network_mask=None):
    mask = mask.unsqueeze(-1)
    pred = pred * mask
    target = target * mask
    if network_mask is not None:
        pred = pred[..., network_mask]
        target = target[..., network_mask]
    if zero_center:
        pred_mean = pred.sum(dim=1) / (mask.sum(dim=1) + eps)
        target_mean = target.sum(dim=1) / (mask.sum(dim=1) + eps)
        pred = pred - pred_mean.unsqueeze(1)
        target = target - target_mean.unsqueeze(1)
    numerator = (pred * target * mask).sum(dim=1)
    denominator = torch.sqrt(((pred**2 * mask).sum(dim=1)) *
                             ((target**2 * mask).sum(dim=1)))
    corr = numerator / (denominator + eps)
    return -corr.mean()


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _centre(x, mask=None, eps=1e-8):
    """centre over time, keep padded steps at 0."""
    if mask is not None:
        valid = mask.sum(1, keepdim=True).clamp_min(1.)
        x = x * mask.unsqueeze(-1)
    else:
        valid = x.new_full((x.size(0), 1), x.size(1))
    mean = x.sum(1, keepdim=True) / valid.unsqueeze(-1)
    xc = x - mean
    if mask is not None:
        xc = xc * mask.unsqueeze(-1)
    return xc, valid.squeeze(1)


# ------------------------------------------------------------------
# 1) sample × sample correlation   –   O(B²)
# ------------------------------------------------------------------


def _sample_corr_matrix(x, mask=None, eps=1e-8):
    xc, _ = _centre(x, mask, eps)
    feat = F.normalize(xc.flatten(1), dim=1)
    return feat @ feat.T


def sample_similarity_loss(pred, target, mask=None, eps=1e-8):
    C_pred = _sample_corr_matrix(pred, mask, eps)
    C_target = _sample_corr_matrix(target, mask, eps)
    C_pred.diagonal().zero_()
    C_target.diagonal().zero_()
    return F.mse_loss(C_pred, C_target)


# ------------------------------------------------------------------
# 2) ROI × ROI correlation   –   batched O(R²)
# ------------------------------------------------------------------


def _roi_cov(xc, valid, eps=1e-8):
    """xc: centred (B,S,R); valid: (B,)"""
    xs = xc.transpose(1, 2)
    cov = xs @ xc / (valid.view(-1, 1, 1) - 1 + eps)
    return cov


def _roi_corr_matrix(x, mask=None, eps=1e-8):
    xc, valid = _centre(x, mask, eps)
    cov = _roi_cov(xc, valid, eps)
    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0.) + eps)
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + eps)
    return corr.clamp(-1., 1.)


def roi_similarity_loss(pred, target, mask=None, eps=1e-8):
    C_pred = _roi_corr_matrix(pred, mask, eps)
    C_target = _roi_corr_matrix(target, mask, eps)
    eye = torch.eye(C_pred.size(-1), device=pred.device, dtype=pred.dtype)
    C_pred = C_pred - eye
    C_target = C_target - eye
    return F.mse_loss(C_pred, C_target)


def spatial_regularizer_loss(pred, Laplacian, mask=None):
    if mask is not None:
        # mask: Boolean tensor of shape (R,), True for voxels to include
        m = mask.float()
        mask_mat = m.unsqueeze(1) * m.unsqueeze(0)
        Laplacian = Laplacian * mask_mat
    energy = torch.einsum('...i,ij,...j', pred, Laplacian, pred)
    loss = energy.mean()
    return loss


def temporal_regularizer_loss(pred):
    variation = (pred[:, 1:] - pred[:, :-1])**2
    loss = variation.mean()
    second_order_variation = (
        pred[2:] + pred[:-2] - 2 * pred[1:-1]
    )**2
    loss += second_order_variation.mean()
    return loss


def temporal_regularizer_loss_new(pred, Laplacian):
    L = Laplacian[: pred.shape[1], : pred.shape[1]]
    energy = torch.einsum('...ik,ij,...jk', pred, L, pred)
    loss = energy.mean()
    return loss


def network_specific_temporal_regularizer_loss(pred, Laplacians, masks):
    if Laplacians is None:
        return torch.zeros((1,), device=pred.device)
    loss = 0.0
    for network in Laplacians.keys():
        L = Laplacians[network][: pred.shape[1], : pred.shape[1]]
        activity = pred[..., masks[network]]
        energy = torch.einsum('...ik,ij,...jk', activity, L, activity)
        loss += energy.mean()
    return loss
