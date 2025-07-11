import torch
import torch.nn.functional as F

def masked_negative_pearson_loss(pred, target, mask, eps=1e-8, zero_center=True,network_mask = None):


    mask = mask.unsqueeze(-1)
    pred = pred * mask
    target = target * mask

    if network_mask is not None:
        pred = pred[...,network_mask]
        target = target[...,network_mask]


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

import torch

def masked_covariance_matrix(data, mask, zero_center=True, eps=1e-8):
    if mask.ndim < data.ndim:
        mask = mask.unsqueeze(-1)

    data_masked = data * mask
    
    if zero_center:
        data_mean = data_masked.sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + eps)
        data_centered = data_masked - data_mean
    else:
        data_centered = data_masked

    data_centered_masked = data_centered * mask
    sum_outer_products = torch.matmul(data_centered_masked.transpose(1, 2), data_centered_masked)
    
    return sum_outer_products

def masked_negative_rv_loss(pred, target, mask, eps=1e-8, zero_center=True, network_mask=None):
    # This block ensures the mask has the correct number of dimensions for broadcasting
    # with pred/target's feature dimension, similar to how Pearson does it.
    if mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1) # Add a dimension if mask is (batch_size, time_steps)
                                  # and data is (batch_size, time_steps, features)

    if network_mask is not None:
        pred = pred[..., network_mask]
        target = target[..., network_mask]
        
        # Apply network_mask to the mask itself to match the new 'pred' and 'target' shape
        # Since 'mask' has now been potentially unsqueezed, this is robust.
        mask_for_stats = mask[..., network_mask] 
    else:
        mask_for_stats = mask

    # Now, pred, target, and mask_for_stats should have compatible shapes
    # (batch, time_steps, features) and (batch, time_steps, 1) or (batch, time_steps, features)
    
    Sx_pred = masked_covariance_matrix(pred, mask_for_stats, zero_center, eps)
    Sy_target = masked_covariance_matrix(target, mask_for_stats, zero_center, eps)

    pred_masked = pred * mask_for_stats # This multiplication should now work
    target_masked = target * mask_for_stats
    # ... rest of your loss function
    if zero_center:
        pred_mean = pred_masked.sum(dim=1, keepdim=True) / (mask_for_stats.sum(dim=1, keepdim=True) + eps)
        target_mean = target_masked.sum(dim=1, keepdim=True) / (mask_for_stats.sum(dim=1, keepdim=True) + eps)
        pred_centered = pred_masked - pred_mean
        target_centered = target_masked - target_mean
    else:
        pred_centered = pred_masked
        target_centered = target_masked

    pred_centered_masked = pred_centered * mask_for_stats
    target_centered_masked = target_centered * mask_for_stats

    Sxy = torch.matmul(pred_centered_masked.transpose(1, 2), target_centered_masked)
    Syx = Sxy.transpose(1, 2)

    numerator = torch.einsum('bii->b', torch.matmul(Sxy, Syx))

    denominator_part_x = torch.einsum('bii->b', torch.matmul(Sx_pred, Sx_pred))
    denominator_part_y = torch.einsum('bii->b', torch.matmul(Sy_target, Sy_target))
    
    denominator = torch.sqrt(denominator_part_x * denominator_part_y + eps)

    # Handle cases where denominator might be zero due to no variance or all masked
    # If denominator is near zero, RV is undefined or approaches zero.
    # In such cases, we want the loss to indicate poor correlation, e.g., -0.
    rv_coefficient = torch.where(denominator < eps, torch.tensor(0.0, device=pred.device), numerator / (denominator + eps))

    return -rv_coefficient.mean()
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



def spatial_regularizer_loss(pred,Laplacian):


    energy = torch.einsum('...i,ij,...j',pred,Laplacian,pred)
    
    loss = energy.mean()
    
    return loss


def temporal_regularizer_loss(pred):

    variation = (pred[:,1:]-pred[:,:-1])**2

    loss = variation.mean()

    second_order_variation = (pred[2:]+ pred[:-2]-2*pred[1:-1])**2

    loss+= second_order_variation.mean()

    return loss


def temporal_regularizer_loss_new(pred,Laplacian):

    
    L = Laplacian[:pred.shape[1],:pred.shape[1]]

    energy = torch.einsum('...ik,ij,...jk',pred,L,pred)
    
    loss = energy.mean()
    
    return loss



def network_specific_temporal_regularizer_loss(pred,Laplacians,masks):

    loss = 0.0
    if Laplacians is None:
        return torch.zeros((1,),device=pred.device)

    for network in Laplacians.keys():
        
        L = Laplacians[network][:pred.shape[1],:pred.shape[1]]
    
        activity_in_network = pred[...,masks[network]]

        energy = torch.einsum('...ik,ij,...jk',activity_in_network,L,activity_in_network)
        
        loss += energy.mean()
    
    return loss

