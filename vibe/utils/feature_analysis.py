import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

from vibe.utils.utils import evaluate_corr
from vibe.utils import logger


def feature_single_ablation(model, val_loader, device, base_r):
    """Leave‑one‑block Δr; returns delta_dict."""
    delta_dict = {}
    for name, proj in model.encoder.projections.items():
        W0, b0 = proj[0].weight.data.clone(), proj[0].bias.data.clone()
        proj[0].weight.zero_()
        proj[0].bias.zero_()
        with torch.no_grad():
            r = evaluate_corr(model, val_loader, device=device).mean().item()
        delta_dict[name] = r - base_r
        proj[0].weight.copy_(W0)
        proj[0].bias.copy_(b0)

    abl_table = wandb.Table(data=[[k, v] for k, v in delta_dict.items()],
                            columns=["block", "delta_r"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(delta_dict.keys()), list(delta_dict.values()))
    ax.set_ylabel("Δ r")
    ax.set_xticklabels(delta_dict.keys(), rotation=45, ha="right")
    ax.set_title("Leave‑one‑block Δr (validation)")
    plt.tight_layout()
    wandb.log({"ablate/bar_chart": wandb.Image(fig), "ablate/table": abl_table})
    plt.close(fig)
    return delta_dict


def feature_pairwise_redundancy(model, val_loader, device, base_r, delta_dict, blocks):
    """Δr_AB – additive; logs heat‑map. Only if len(blocks) < cutoff."""
    n=len(blocks)
    red = torch.zeros(n,n)
    for i in range(n):
        for j in range(i+1,n):
            bi,bj=blocks[i],blocks[j]
            Wi,bi_b = model.encoder.projections[bi][0].weight.data.clone(), model.encoder.projections[bi][0].bias.data.clone()
            Wj,bj_b = model.encoder.projections[bj][0].weight.data.clone(), model.encoder.projections[bj][0].bias.data.clone()
            model.encoder.projections[bi][0].weight.zero_()
            model.encoder.projections[bi][0].bias.zero_()
            model.encoder.projections[bj][0].weight.zero_()
            model.encoder.projections[bj][0].bias.zero_()
            r_joint = evaluate_corr(model,val_loader,device=device).mean().item()-base_r
            model.encoder.projections[bi][0].weight.copy_(Wi)
            model.encoder.projections[bi][0].bias.copy_(bi_b)
            model.encoder.projections[bj][0].weight.copy_(Wj)
            model.encoder.projections[bj][0].bias.copy_(bj_b)
            red[i,j]=red[j,i]=r_joint-(delta_dict[bi]+delta_dict[bj])
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(red.numpy(),ax=ax,xticklabels=blocks,yticklabels=blocks,square=True,cmap="rocket")
    ax.set_title("Pairwise redundancy Δr")
    wandb.log({"ablate/redundancy_heatmap":wandb.Image(fig)})
    plt.close(fig)


def run_feature_analyses(model, val_loader, device):
    """
    Runs three lightweight post‑hoc analyses on the *validation* set:
        1. Leave‑one‑block ablation  (Δr)
        2. Permutation importance    (Δr)
        3. Linear SHAP values        (mean |ϕ|)
    Logs results to W&B under keys:
        ablate/<block>, permute/<block>, shap/hidden_<i>
    """
    model.eval().requires_grad_(False)

    blocks = list(model.encoder.projections.keys())
    EXACT_CUTOFF = 6

    base_r = evaluate_corr(model, val_loader, device=device).mean().item()
    wandb.log({"diag/baseline_r": base_r})

    with logger.step("🔹 Leave‑one‑block ablation"):
        delta_dict = feature_single_ablation(model, val_loader, device, base_r)

    if len(blocks) <= EXACT_CUTOFF:
        with logger.step("🔹 Pairwise redundancy"):
            feature_pairwise_redundancy(model, val_loader, device, base_r, delta_dict, blocks)